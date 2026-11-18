# FastAPI backend and deployment

The challenge brief schedules Review 1 before lunch and describes a final evaluation. The Review 1 checklists below are proposed team milestones. The brief gives no separate checklist for that checkpoint.

## Review 1 requirements

- [x] Run a FastAPI stub with a health check, a trip list and a prediction endpoint backed by development fixtures.
- [x] Provide a valid response and representative error payload for the UI in the [README](../README.md).
- [ ] Demonstrate a request from the second laptop and its corresponding log entry on the receiving laptop.

## Final evaluation requirements

The backup branch supports trained inference in student and evaluator modes. Student mode was exercised end to end. Evaluator mode uses the local manifest described in the [README](../README.md#reserved-evaluation-mode).

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
