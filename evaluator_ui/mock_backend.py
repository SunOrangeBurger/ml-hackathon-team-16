"""
FastAPI Development Stub / Mock Server conforming to documentation/03-fastapi-backend.md.

Use this on Laptop 1 for Review 1 demonstrations to prove cross-laptop connectivity
even before the final ML model and HMM decoder are fully integrated.

Run with:
    python mock_backend.py --host 0.0.0.0 --port 8000
"""

import argparse
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

from evaluator_ui.fixtures.sample_trips import MOCK_TRIPS

app = FastAPI(
    title="Route Reconstruction Inference API",
    description="Backend inference service for GPS trajectory map-matching and route reconstruction.",
    version="1.0.0"
)

# Enable CORS for cross-origin / cross-laptop requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictRequest(BaseModel):
    trip_id: str

class GeoJSONGeometry(BaseModel):
    type: str
    coordinates: List[Any]

class PredictResponse(BaseModel):
    trip_id: str
    gps: GeoJSONGeometry
    predicted_route: GeoJSONGeometry
    metrics: Optional[Dict[str, Any]] = None
    warnings: List[str] = []

@app.get("/health")
def get_health():
    """Reports readiness of backend service."""
    return {"status": "ready", "version": "1.0.0"}

@app.get("/trips", response_model=List[str])
def get_trips():
    """Returns trip IDs as strings, preserving leading zeros."""
    return sorted(list(MOCK_TRIPS.keys()))

@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    """
    Accepts trip_id, performs trajectory map-matching (or returns development fixture),
    and returns GPS points, predicted route geometry, aggregate metrics, and warnings.
    """
    tid = str(request.trip_id)
    if tid not in MOCK_TRIPS:
        raise HTTPException(
            status_code=404,
            detail=f"Trip ID '{tid}' not found in manifest. Available trips: {list(MOCK_TRIPS.keys())}"
        )

    trip = MOCK_TRIPS[tid]
    return {
        "trip_id": tid,
        "gps": trip["gps"],
        "predicted_route": trip["predicted_route"],
        "metrics": trip["metrics"],
        "warnings": trip["warnings"]
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run FastAPI Mock Inference Backend")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host address to bind (0.0.0.0 for cross-laptop)")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on (default 8000)")
    args = parser.parse_args()

    print(f"============================================================")
    print(f" Starting FastAPI Mock Server for Hackathon Review 1")
    print(f" Listening on: http://{args.host}:{args.port}")
    print(f" Available endpoints:")
    print(f"   GET  /health")
    print(f"   GET  /trips")
    print(f"   POST /predict")
    print(f"============================================================")
    uvicorn.run(app, host=args.host, port=args.port)
