"""FastAPI service for the PathSnap development demo."""

import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from route_reconstruction import decode, load_fixture_graph

LOGGER = logging.getLogger("uvicorn.error").getChild("pathsnap")
DEFAULT_FIXTURE = Path(__file__).parent / "fixtures" / "development_trip.json"
DECODER_FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "route-reconstruction"


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    trip_id: str = Field(min_length=1, max_length=128)


class MultiPoint(BaseModel):
    type: Literal["MultiPoint"]
    coordinates: list[tuple[float, float]]


class MultiLineString(BaseModel):
    type: Literal["MultiLineString"]
    coordinates: list[list[tuple[float, float]]]


class PredictionResponse(BaseModel):
    trip_id: str
    gps: MultiPoint
    predicted_route: MultiLineString
    metrics: dict[str, float | int] | None
    warnings: list[str]


class FixtureFile(BaseModel):
    trip_id: str
    gps: MultiPoint
    predicted_route: MultiLineString
    metrics: dict[str, float | int] | None = None
    warnings: list[str] = Field(default_factory=list)


class Service:
    def __init__(self, mode: str, fixture_path: Path):
        self.mode = mode
        self.fixture_path = fixture_path
        self.fixture: FixtureFile | None = None
        self.decoder_fixtures: dict[str, dict] = {}
        self.data_dir = Path(os.getenv("PATHSNAP_DATA", "Student folder/Student folder"))
        self.failure: str | None = None

    def initialize(self) -> None:
        if self.mode == "decoder":
            try:
                paths = [DECODER_FIXTURE_DIR / "synthetic_decoder_fixture.json"]
                if real_fixture := os.getenv("PATHSNAP_DECODER_FIXTURE"):
                    paths.append(Path(real_fixture))
                fixtures = {}
                for path in paths:
                    fixture = json.loads(path.read_text(encoding="utf-8"))
                    trip_id = fixture["trip_id"]
                    if fixture["schema_version"] != 1 or not isinstance(trip_id, str) or not trip_id or trip_id in fixtures:
                        raise ValueError(f"Invalid decoder fixture: {path}")
                    load_fixture_graph(fixture, self.data_dir)
                    fixtures[trip_id] = fixture
                self.decoder_fixtures = fixtures
            except (OSError, ValueError, TypeError, KeyError):
                LOGGER.exception("Failed to initialize development decoder")
                self.failure = "Development decoder fixtures could not be loaded."
            return
        if self.mode != "fixture":
            self.failure = "Artifact-backed inference is not configured."
            return

        try:
            data = json.loads(self.fixture_path.read_text(encoding="utf-8"))
            self.fixture = FixtureFile.model_validate(data)
        except (OSError, ValueError, ValidationError):
            LOGGER.exception("Failed to load development prediction fixture")
            self.failure = "Development fixture could not be loaded."

    @property
    def ready(self) -> bool:
        return (self.fixture is not None or bool(self.decoder_fixtures)) and self.failure is None

    def decode_trip(self, trip_id: str) -> PredictionResponse:
        fixture = self.decoder_fixtures[trip_id]
        observations = fixture["observations"]
        graph = load_fixture_graph(fixture, self.data_dir)
        result = decode(observations, graph)
        sections = [
            section["geometry"] for section in result["route_sections"]
            if len(section["geometry"]) >= 2
        ]
        return PredictionResponse(
            trip_id=trip_id,
            gps=MultiPoint(
                type="MultiPoint",
                coordinates=[(row["longitude"], row["latitude"]) for row in observations],
            ),
            predicted_route=MultiLineString(type="MultiLineString", coordinates=sections),
            metrics={
                "observation_count": len(observations),
                "matched_count": sum(row["arc_id"] is not None for row in result["assignments"]),
                "route_section_count": len(sections),
            },
            warnings=[
                "Development scores are used for decoding.",
                *(f'Observation {item["observation_index"]}: {item["message"]}' for item in result["warnings"]),
            ],
        )


def create_app(
    mode: str | None = None,
    fixture_path: Path | None = None,
    cors_origins: list[str] | None = None,
) -> FastAPI:
    service = Service(
        mode=mode or os.getenv("PATHSNAP_MODE", "fixture"),
        fixture_path=fixture_path or Path(os.getenv("PATHSNAP_FIXTURE_PATH", DEFAULT_FIXTURE)),
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        service.initialize()
        app.state.service = service
        yield

    app = FastAPI(title="PathSnap API", version="1.0.0", lifespan=lifespan)
    origins = cors_origins
    if origins is None:
        origins = [
            origin.strip()
            for origin in os.getenv("PATHSNAP_CORS_ORIGINS", "").split(",")
            if origin.strip()
        ]
    if origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_methods=["GET", "POST"],
            allow_headers=["Content-Type", "X-Request-ID"],
        )

    @app.middleware("http")
    async def log_request(request: Request, call_next):
        request.state.request_id = str(uuid.uuid4())
        request.state.trip_id = None
        started = time.perf_counter()
        status_code = 500
        try:
            response = await call_next(request)
            status_code = response.status_code
            response.headers["X-Request-ID"] = request.state.request_id
            return response
        finally:
            elapsed_ms = (time.perf_counter() - started) * 1000
            LOGGER.info(
                "request_id=%s client=%s trip_id=%s status=%s duration_ms=%.2f method=%s path=%s",
                request.state.request_id,
                request.client.host if request.client else "-",
                request.state.trip_id or "-",
                status_code,
                elapsed_ms,
                request.method,
                request.url.path,
            )

    def error_response(
        request: Request,
        status_code: int,
        code: str,
        message: str,
        details: list[dict] | None = None,
    ) -> JSONResponse:
        error = {"code": code, "message": message}
        if details is not None:
            error["details"] = details
        return JSONResponse(
            status_code=status_code,
            content={
                "error": error,
                "request_id": getattr(request.state, "request_id", None),
            },
            headers={"X-Request-ID": getattr(request.state, "request_id", "")},
        )

    @app.exception_handler(RequestValidationError)
    async def invalid_request_handler(request: Request, exc: RequestValidationError):
        return error_response(
            request,
            422,
            "invalid_request",
            "Request body does not match the prediction schema.",
            [
                {"location": error["loc"], "message": error["msg"], "type": error["type"]}
                for error in exc.errors()
            ],
        )

    @app.get("/health")
    async def health(request: Request):
        current: Service = request.app.state.service
        if not current.ready:
            return JSONResponse(
                status_code=503,
                content={
                    "status": "not_ready",
                    "mode": current.mode,
                    "detail": current.failure or "Prediction service is unavailable.",
                },
            )
        return {"status": "ready", "mode": current.mode}

    @app.get("/trips")
    async def trips(request: Request):
        current: Service = request.app.state.service
        if not current.ready:
            return error_response(
                request, 503, "inference_unavailable", "Prediction service is not ready."
            )
        return {"trip_ids": list(current.decoder_fixtures) if current.mode == "decoder" else [current.fixture.trip_id]}

    @app.post("/predict", response_model=PredictionResponse)
    def predict(body: PredictionRequest, request: Request):
        request.state.trip_id = body.trip_id
        current: Service = request.app.state.service
        if not current.ready:
            return error_response(
                request, 503, "inference_unavailable", "Prediction service is not ready."
            )
        if current.mode == "decoder" and body.trip_id in current.decoder_fixtures:
            try:
                return current.decode_trip(body.trip_id)
            except (OSError, ValueError, TypeError, KeyError):
                LOGGER.exception("Failed to decode development trip %s", body.trip_id)
                return error_response(
                    request, 503, "inference_unavailable", "Development trip could not be decoded."
                )
        if current.fixture is None or body.trip_id != current.fixture.trip_id:
            return error_response(
                request, 404, "unknown_trip", f"Unknown trip_id: {body.trip_id}"
            )
        return PredictionResponse(
            trip_id=current.fixture.trip_id,
            gps=current.fixture.gps,
            predicted_route=current.fixture.predicted_route,
            metrics=current.fixture.metrics,
            warnings=current.fixture.warnings,
        )

    return app


app = create_app()
