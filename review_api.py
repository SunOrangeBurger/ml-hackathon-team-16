"""FastAPI development service for the PathSnap Review 1 fixtures."""

import csv
import json
import logging
import os
from functools import lru_cache
from pathlib import Path
from time import perf_counter
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from learned_inference import evaluate_assignments, load_artifact, predict_trip
from route_decoder import decode, load_fixture


ROOT = Path(__file__).resolve().parent
DATA = Path(os.environ.get("PATHSNAP_DATA", ROOT / "Student folder/Student folder"))
OUTPUT = Path(os.environ.get("PATHSNAP_REVIEW_OUTPUT", ROOT / "output/review-1"))
MODE = os.environ.get("PATHSNAP_MODE", "review")
MODEL = Path(os.environ.get("PATHSNAP_MODEL", ROOT / "artifacts/model.joblib"))
FIXTURES = {
    "synthetic": OUTPUT / "synthetic_decoder_fixture.json",
    "00000001": OUTPUT / "00000001/decoder_fixture.json",
}
logger = logging.getLogger("uvicorn.error")
app = FastAPI(title="PathSnap Review 1")
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get(
        "PATHSNAP_UI_ORIGINS", "http://localhost:8080,http://127.0.0.1:8080"
    ).split(","),
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


class PredictionRequest(BaseModel):
    trip_id: str
    model_id: str | None = None


@lru_cache(maxsize=5)
def catalog_artifact(model_id):
    catalog = json.loads((ROOT / "artifacts/models/catalog.json").read_text(encoding="utf-8"))
    if model_id not in {row["id"] for row in catalog}:
        raise HTTPException(404, "Unknown model")
    return load_artifact(ROOT / "artifacts/models" / f"{model_id}.joblib")


@app.get("/models")
def models():
    path = ROOT / "artifacts/models/catalog.json"
    return {"models": json.loads(path.read_text(encoding="utf-8")) if path.exists() else []}


@lru_cache(maxsize=1)
def learned_service():
    if MODE == "student":
        folders = {folder.name: folder for folder in sorted((DATA / "student_test").iterdir())
                   if folder.is_dir()}
    elif MODE == "evaluator":
        manifest = Path(os.environ["PATHSNAP_EVALUATOR_MANIFEST"]).resolve()
        entries = json.loads(manifest.read_text(encoding="utf-8"))
        if not isinstance(entries, dict) or not all(
            key and isinstance(value, str) and value for key, value in entries.items()
        ):
            raise ValueError("Evaluator manifest must map trip IDs to folder paths")
        folders = {key: (manifest.parent / value).resolve() for key, value in entries.items()}
    else:
        raise ValueError("Unknown PATHSNAP_MODE")
    if len(folders) != 10 or len(set(folders.values())) != 10 or not all(
        folder.is_dir() for folder in folders.values()
    ):
        raise ValueError("Expected exactly ten distinct trip directories")
    return load_artifact(MODEL), folders


def ready_service():
    try:
        return learned_service()
    except (OSError, KeyError, ValueError) as error:
        logger.warning("Could not initialize learned service: %s", error)
        raise HTTPException(503, "Learned inference is unavailable") from error


@app.middleware("http")
async def log_request(request: Request, call_next):
    request_id = uuid4().hex[:12]
    request.state.request_id = request_id
    start = perf_counter()
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    logger.info(
        "request=%s client=%s method=%s path=%s trip=%s status=%s duration_ms=%.1f",
        request_id, request.client.host if request.client else "unknown",
        request.method, request.url.path, getattr(request.state, "trip_id", "-"),
        response.status_code, (perf_counter() - start) * 1000,
    )
    return response


@app.get("/health")
def health():
    if MODE != "review":
        ready_service()
        return {"status": "ready", "mode": MODE}
    ready = all(path.is_file() for path in FIXTURES.values())
    if not ready:
        raise HTTPException(503, "Generate the development fixtures with review_one.py")
    return {"status": "ready"}


@app.get("/trips")
def trips():
    if MODE != "review":
        _, folders = ready_service()
        return {"trip_ids": list(folders)}
    health()
    return {"trip_ids": list(FIXTURES)}


@app.post("/predict")
def predict(payload: PredictionRequest, request: Request):
    trip_id = payload.trip_id
    request.state.trip_id = trip_id
    if MODE != "review":
        return predict_learned(trip_id, payload.model_id)
    if trip_id not in FIXTURES:
        raise HTTPException(404, "Unknown development trip")
    try:
        fixture, nodes, arcs = load_fixture(FIXTURES[trip_id], DATA)
        observations = fixture["observations"]
        assignments, sections, warnings = decode(observations, nodes, arcs)
        metrics = {
            "observation_count": len(assignments),
            "matched_count": sum(arc is not None for arc in assignments),
            "route_section_count": len(sections),
        }
        if trip_id == "00000001":
            with (FIXTURES[trip_id].parent / "aligned_reference.csv").open(
                encoding="utf-8", newline=""
            ) as stream:
                labels = [int(row["reference_arc_id"]) for row in csv.DictReader(stream)]
            if len(labels) != len(assignments):
                raise ValueError("Label count does not match fixture")
            metrics["provisional_accuracy"] = sum(a == b for a, b in zip(assignments, labels)) / len(labels)
    except (OSError, ValueError, KeyError) as error:
        logger.warning("request=%s fixture error: %s", request.state.request_id, error)
        raise HTTPException(503, "Development fixture unavailable or invalid") from error
    return prediction_response(trip_id, observations, sections, metrics, warnings)


def predict_learned(trip_id, model_id=None):
    artifact, folders = ready_service()
    if model_id is not None:
        artifact = catalog_artifact(model_id)
    if trip_id not in folders:
        raise HTTPException(404, "Unknown trip")
    try:
        folder = folders[trip_id]
        trip, _, observations, assignments, sections, warnings = predict_trip(folder, artifact)
    except (OSError, ValueError) as error:
        logger.warning("Inference failed for %s: %s", trip_id, error)
        raise HTTPException(503, "Could not run inference for this trip") from error
    metrics = {
        "observation_count": len(assignments),
        "matched_count": sum(arc is not None for arc in assignments),
        "route_section_count": len(sections),
    }
    try:
        _, accuracy = evaluate_assignments(folder, trip, assignments)
        metrics.update(accuracy)
    except (OSError, ValueError) as error:
        logger.warning("Scoring unavailable for %s: %s", trip_id, error)
        warnings.append("Reference scoring unavailable")
    response = prediction_response(trip_id, observations, sections, metrics, warnings)
    if os.environ.get("PATHSNAP_SHOW_REFERENCE") == "1" and len(trip.route):
        response["reference_route"] = {
            "type": "MultiLineString",
            "coordinates": trip.nodes[trip.arcs[trip.route]].tolist(),
        }
    return response


def prediction_response(trip_id, observations, sections, metrics, warnings):
    return {
        "trip_id": trip_id,
        "gps": {"type": "MultiPoint", "coordinates": [
            [row["longitude"], row["latitude"]] for row in observations
        ]},
        "predicted_route": {"type": "MultiLineString", "coordinates": sections},
        "metrics": metrics,
        "warnings": warnings,
    }
