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

MOCK_TRIPS = {
    "001": {
        "trip_id": "001",
        "title": "Trip 001 - Continuous Corridor (Training Dev)",
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
    "002": {
        "trip_id": "002",
        "title": "Trip 002 - Route Discontinuity (Tunnel / Signal Loss)",
        "description": "Trajectory exhibiting a 92-second timestamp gap. Demonstrates that Viterbi emits segmented MultiLineString without connecting across gaps.",
        "gps": {
            "type": "MultiPoint",
            "coordinates": [
                [77.6050, 12.9860],
                [77.6062, 12.9875],
                [77.6075, 12.9890],
                # Gap period
                [77.6130, 12.9950],
                [77.6145, 12.9968],
                [77.6160, 12.9985]
            ]
        },
        "predicted_route": {
            "type": "MultiLineString",
            # Two separate line segments
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
    "003": {
        "trip_id": "003",
        "title": "Trip 003 - Reserved Evaluator Trip",
        "description": "Representative reserved evaluation trip. Server-side ground truth is intentionally withheld (null metrics).",
        "gps": {
            "type": "MultiPoint",
            "coordinates": [
                [77.5850, 12.9620],
                [77.5862, 12.9635],
                [77.5878, 12.9650],
                [77.5895, 12.9668],
                [77.5910, 12.9685],
                [77.5925, 12.9702]
            ]
        },
        "predicted_route": {
            "type": "MultiLineString",
            "coordinates": [
                [
                    [77.5848, 12.9618],
                    [77.5860, 12.9632],
                    [77.5875, 12.9648],
                    [77.5892, 12.9665],
                    [77.5908, 12.9682],
                    [77.5923, 12.9700]
                ]
            ]
        },
        "reference_route": None,
        "metrics": None,
        "warnings": [
            "Evaluator mode: Ground truth is hidden server-side. Pointwise metrics are unavailable."
        ]
    }
}
