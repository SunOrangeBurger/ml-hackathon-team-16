from fastapi.testclient import TestClient

from backend.app import create_app


def test_fixture_endpoints_return_expected_contract():
    with TestClient(create_app()) as client:
        assert client.get("/health").json() == {"status": "ready", "mode": "fixture"}
        assert client.get("/trips").json() == {"trip_ids": ["synthetic"]}

        response = client.post("/predict", json={"trip_id": "synthetic"})

    assert response.status_code == 200
    assert response.headers["X-Request-ID"]
    assert response.json() == {
        "trip_id": "synthetic",
        "gps": {
            "type": "MultiPoint",
            "coordinates": [[8.00005, 49.0], [8.00015, 49.00003]],
        },
        "predicted_route": {
            "type": "MultiLineString",
            "coordinates": [[[8.0, 49.0], [8.0001, 49.0], [8.0002, 49.0]]],
        },
        "metrics": None,
        "warnings": [
            "Development fixture only; this response is not a live model prediction."
        ],
    }


def test_unknown_trip_has_documented_error_shape():
    with TestClient(create_app()) as client:
        response = client.post("/predict", json={"trip_id": "missing"})

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "unknown_trip"
    assert response.json()["request_id"] == response.headers["X-Request-ID"]


def test_invalid_request_has_documented_error_shape():
    with TestClient(create_app()) as client:
        response = client.post("/predict", json={"trip_id": ""})

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_request"
    assert response.json()["error"]["details"]


def test_unconfigured_artifact_mode_is_not_ready():
    with TestClient(create_app(mode="artifact")) as client:
        health = client.get("/health")
        prediction = client.post("/predict", json={"trip_id": "synthetic"})

    assert health.status_code == 503
    assert health.json()["status"] == "not_ready"
    assert prediction.status_code == 503
    assert prediction.json()["error"]["code"] == "inference_unavailable"


def test_missing_fixture_reports_not_ready(tmp_path):
    missing = tmp_path / "missing.json"
    with TestClient(create_app(fixture_path=missing)) as client:
        response = client.get("/health")

    assert response.status_code == 503
    assert response.json()["status"] == "not_ready"
