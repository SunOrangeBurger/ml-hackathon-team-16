"""FastAPI service for the PathSnap development fixture."""

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


LOGGER = logging.getLogger("pathsnap.api")
DEFAULT_FIXTURE = Path(__file__).parent / "fixtures" / "development_trip.json"


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
        self.failure: str | None = None

    def initialize(self) -> None:
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
        return self.fixture is not None and self.failure is None


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
                "request_id=%s trip_id=%s status=%s duration_ms=%.2f method=%s path=%s",
                request.state.request_id,
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
        return {"trip_ids": [current.fixture.trip_id]}

    @app.post("/predict", response_model=PredictionResponse)
    async def predict(body: PredictionRequest, request: Request):
        request.state.trip_id = body.trip_id
        current: Service = request.app.state.service
        if not current.ready:
            return error_response(
                request, 503, "inference_unavailable", "Prediction service is not ready."
            )
        if body.trip_id != current.fixture.trip_id:
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
