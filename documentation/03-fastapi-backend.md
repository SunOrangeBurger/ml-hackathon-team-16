# FastAPI backend and deployment

The challenge brief schedules Review 1 before lunch and describes a final evaluation. The Review 1 checklists below are proposed team milestones. The brief gives no separate checklist for that checkpoint.

## Review 1 requirements

- [x] Run a FastAPI stub with a health check, a trip list and a prediction endpoint backed by development fixtures.
- [x] Define a valid response and representative error payload for the UI owner.
- [ ] Demonstrate a request from the second laptop and its corresponding log entry on the receiving laptop.

The fixture service is implemented in [`backend/app.py`](../backend/app.py) and returns the synthetic decoder example. It is an integration stub, not model inference. The current checkout does not contain trained model or evaluator artifacts.

### Run the development service

Install dependencies and run from the project root:

```powershell
python -m pip install -r requirements.txt
uvicorn backend.app:app --host 0.0.0.0 --port 8000
```

`0.0.0.0` allows Laptop 2 to connect. Set `PATHSNAP_CORS_ORIGINS` to a comma-separated list of the UI's exact origins if it runs in a browser, for example `http://localhost:5173`. `PATHSNAP_FIXTURE_PATH` can point to another response fixture with the same schema.

```powershell
curl http://localhost:8000/health
curl http://localhost:8000/trips
curl -H "Content-Type: application/json" -d '{"trip_id":"synthetic"}' http://localhost:8000/predict
```

The API also exposes its OpenAPI document at `/openapi.json` and interactive docs at `/docs`.

### Contract and errors

- `GET /health` returns `{"status":"ready","mode":"fixture"}` or HTTP 503 with `status: "not_ready"`.
- `GET /trips` returns `{"trip_ids":["synthetic"]}`. IDs remain strings.
- `POST /predict` accepts `{"trip_id":"synthetic"}` and responds with the fields in the documented response shape below.
- Error responses use `{"error":{"code":"...","message":"..."},"request_id":"..."}`. Invalid requests return 422 with `error.details`; unknown IDs return 404 with `unknown_trip`; unavailable inference returns 503 with `inference_unavailable`.
- Every response includes `X-Request-ID`; Laptop 1 logs the request ID, trip ID (when provided), status and duration.
- GeoJSON coordinates are `[longitude, latitude]`. Metrics are `null` in the fixture; the fixture warning identifies that the response is not a live prediction.

Example fixture response:

```json
{
  "trip_id": "synthetic",
  "gps": {"type": "MultiPoint", "coordinates": [[8.00005, 49.0], [8.00015, 49.00003]]},
  "predicted_route": {"type": "MultiLineString", "coordinates": [[[8.0, 49.0], [8.0001, 49.0], [8.0002, 49.0]]]},
  "metrics": null,
  "warnings": ["Development fixture only; this response is not a live model prediction."]
}
```

Set `PATHSNAP_MODE=artifact` to ensure the service reports not-ready until an artifact-backed inference provider is integrated. It intentionally does not fall back to fixture predictions in that mode.

Run the focused API tests with `python -m pytest tests/test_backend.py`.

## Final evaluation requirements

- [ ] Host the classifier, preprocessing, decoder and road network on Laptop 1.
- [ ] Support selection and live inference for any of the ten reserved evaluator trips during final evaluation.
- [ ] Calculate accuracy server-side using internal reference data and expose only aggregate results. Show receiving-laptop logs during the demo and capture successful responses for the report.
- [ ] Provide dependencies, configuration and startup and connectivity steps in the project README.

## Detailed tasks

1. Define the proposed contract. `GET /health` reports readiness, `GET /trips` returns trip IDs as strings, and `POST /predict` accepts a JSON object with `trip_id`. Return `trip_id`, `gps` as GeoJSON MultiPoint, `predicted_route` as GeoJSON MultiLineString, `metrics` as an aggregate object or null, and `warnings` as a string array. GeoJSON pairs use longitude then latitude.
2. Resolve IDs through a configured trip manifest. Validate request and response schemas. Agree on JSON errors for unknown IDs, invalid requests and unavailable inference. Keep development and evaluator manifests explicit.
3. Load versioned model, preprocessing and decoder assets on startup. Validate compatibility and return a readiness failure if assets cannot load. Measure cold start and request latency.
4. For each request, load observations and graph, produce ordered candidate probabilities and run the decoder. Convert route sections into GeoJSON. Pass predictions to a separate evaluator function which loads references and computes the defined accuracy and denominators. Keep evaluator reference geometry and labels within that function.
5. Log request ID, trip ID, status and duration on Laptop 1. Configure the bind host, port and UI origin where needed. Verify connectivity from Laptop 2 and exercise unknown IDs, unavailable artifacts and trips with gaps.
6. Rehearse with development trips. During final evaluation, accept any reserved ID through the UI and demonstrate inference and scoring on demand.

## Handoff

Share the OpenAPI schema and fixture response with the [evaluator UI](04-evaluator-ui.md). Integrate the [model artifacts](01-data-and-model-training.md) and [decoder](02-route-reconstruction.md) through their agreed interfaces.
