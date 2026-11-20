"""
Sample development and synthetic fixtures conforming to:
documentation/03-fastapi-backend.md
documentation/04-evaluator-ui.md

Contract:
- trip_id: string (preserves leading zeros)
- gps: GeoJSON MultiPoint (pairs are [longitude, latitude])
- predicted_route: GeoJSON MultiLineString (pairs are [longitude, latitude])
- metrics: object or null
- warnings: string array
"""

# Base sample coordinates around city road corridor
BASE_LAT = 12.9716
BASE_LON = 77.5946

def generate_evaluator_trip(tid: str, offset_idx: int) -> dict:
    """Generates a realistic reserved evaluator trip fixture with server-side ground truth hidden."""
    dlat = offset_idx * 0.003
    dlon = offset_idx * 0.003
    gps_coords = [
        [round(BASE_LON + dlon + i * 0.0012, 6), round(BASE_LAT + dlat + i * 0.0015, 6)]
        for i in range(12)
    ]
    # Projected road route slightly smoother
    route_coords = [
        [round(BASE_LON + dlon + i * 0.0012 - 0.0001, 6), round(BASE_LAT + dlat + i * 0.0015 - 0.0001, 6)]
        for i in range(12)
    ]
    return {
        "trip_id": tid,
        "title": f"Reserved Evaluator Trip {tid}",
        "description": f"Official reserved evaluation trip #{offset_idx}. Ground-truth labels withheld server-side.",
        "gps": {
            "type": "MultiPoint",
            "coordinates": gps_coords
        },
        "predicted_route": {
            "type": "MultiLineString",
            "coordinates": [route_coords]
        },
        "reference_route": None,  # Ground truth strictly hidden on server for evaluator trips!
        "metrics": None,          # Evaluator mode: server computes aggregate results internally
        "warnings": [
            f"Evaluator Mode: Trip {tid} ground-truth reference geometry is withheld on server."
        ]
    }

MOCK_TRIPS = {
    # Development Triplet
    "00000001": {
        "trip_id": "00000001",
        "title": "Trip 00000001 - Continuous Corridor (Training Dev)",
        "description": "Standard urban trajectory with clear road alignment and high candidate confidence.",
        "gps": {
            "type": "MultiPoint",
            "coordinates": [
                [77.5900, 12.9700],
                [77.5912, 12.9715],
                [77.5925, 12.9732],
                [77.5938, 12.9750],
                [77.5952, 12.9768],
                [77.5966, 12.9785],
                [77.5980, 12.9802],
                [77.5995, 12.9818],
                [77.6010, 12.9835],
                [77.6025, 12.9852]
            ]
        },
        "predicted_route": {
            "type": "MultiLineString",
            "coordinates": [
                [
                    [77.5898, 12.9698],
                    [77.5910, 12.9712],
                    [77.5922, 12.9730],
                    [77.5935, 12.9748],
                    [77.5950, 12.9765],
                    [77.5963, 12.9782],
                    [77.5978, 12.9800],
                    [77.5992, 12.9815],
                    [77.6008, 12.9832],
                    [77.6022, 12.9850]
                ]
            ]
        },
        "reference_route": {
            "type": "MultiLineString",
            "coordinates": [
                [
                    [77.5898, 12.9698],
                    [77.5910, 12.9712],
                    [77.5922, 12.9730],
                    [77.5935, 12.9748],
                    [77.5950, 12.9765],
                    [77.5963, 12.9782],
                    [77.5978, 12.9800],
                    [77.5992, 12.9815],
                    [77.6008, 12.9832],
                    [77.6022, 12.9850]
                ]
            ]
        },
        "metrics": {
            "top1_accuracy": 0.954,
            "macro_f1": 0.941,
            "route_continuity": 1.000,
            "candidate_coverage": 0.985,
            "arc_edit_distance": 1
        },
        "warnings": []
    },
    "00000002": {
        "trip_id": "00000002",
        "title": "Trip 00000002 - Route Discontinuity (Tunnel / Signal Loss)",
        "description": "Trajectory exhibiting a 92-second timestamp gap. Demonstrates that Viterbi emits segmented MultiLineString without connecting across gaps.",
        "gps": {
            "type": "MultiPoint",
            "coordinates": [
                [77.6050, 12.9860],
                [77.6062, 12.9875],
                [77.6075, 12.9890],
                # 90-second outage / signal loss
                [77.6130, 12.9950],
                [77.6145, 12.9968],
                [77.6160, 12.9985]
            ]
        },
        "predicted_route": {
            "type": "MultiLineString",
            "coordinates": [
                [
                    [77.6048, 12.9858],
                    [77.6060, 12.9872],
                    [77.6073, 12.9888]
                ],
                [
                    [77.6128, 12.9948],
                    [77.6142, 12.9965],
                    [77.6158, 12.9982]
                ]
            ]
        },
        "reference_route": {
            "type": "MultiLineString",
            "coordinates": [
                [
                    [77.6048, 12.9858],
                    [77.6060, 12.9872],
                    [77.6073, 12.9888],
                    [77.6090, 12.9905],
                    [77.6110, 12.9925],
                    [77.6128, 12.9948],
                    [77.6142, 12.9965],
                    [77.6158, 12.9982]
                ]
            ]
        },
        "metrics": {
            "top1_accuracy": 0.867,
            "macro_f1": 0.842,
            "route_continuity": 0.720,
            "candidate_coverage": 0.910,
            "arc_edit_distance": 4
        },
        "warnings": [
            "Discontinuity warning: 92-second timestamp gap detected between obs 3 and 4.",
            "Viterbi decoder restarted sequence; route emitted as disconnected MultiLineString."
        ]
    },
    "synthetic": {
        "trip_id": "synthetic",
        "title": "Trip synthetic - Shared Backend Development Fixture",
        "description": "Standard synthetic development fixture shared across Workstream 02 (Decoder) and 03 (FastAPI).",
        "gps": {
            "type": "MultiPoint",
            "coordinates": [
                [8.00005, 49.0],
                [8.00015, 49.00003]
            ]
        },
        "predicted_route": {
            "type": "MultiLineString",
            "coordinates": [
                [
                    [8.0, 49.0],
                    [8.0001, 49.0],
                    [8.0002, 49.0]
                ]
            ]
        },
        "reference_route": None,
        "metrics": None,
        "warnings": [
            "Development fixture only; this response matches backend/fixtures/development_trip.json."
        ]
    }
}

# Add the 10 Reserved Student Evaluator Trips (00000091 through 00000100)
for idx, num in enumerate(range(91, 101), start=1):
    tid_str = f"{num:08d}"
    MOCK_TRIPS[tid_str] = generate_evaluator_trip(tid_str, idx)

# Aliases for 3-digit convenience
MOCK_TRIPS["001"] = {**MOCK_TRIPS["00000001"], "trip_id": "001"}
MOCK_TRIPS["002"] = {**MOCK_TRIPS["00000002"], "trip_id": "002"}
MOCK_TRIPS["003"] = {**MOCK_TRIPS["00000091"], "trip_id": "003"}
