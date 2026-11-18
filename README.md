# PathSnap backup implementation

The evaluator includes five saved classifier choices. Run `python build_model_catalog.py`
to rebuild their artifacts and aggregate scores from the supplied training, student-test
and evaluator folders. Models are ordered by student-test decoded accuracy.
The displayed combined accuracy weights each eligible observation equally across
student-test and organizer evaluator trips. Each split is also shown separately.
These measurements use provisional geometric reference alignment.

This branch combines the Workstream 01 audit and distance scores with a directed Viterbi decoder, a FastAPI development service, and an evaluator UI. It uses a synthetic fixture and training trip `00000001`.

The branch also includes the trained model and a learned path from GPS observations to a decoded route. See [measured development results](documentation/development-results.md).

## Run locally

From the repository root, place the supplied dataset at `Student folder/Student folder/`, or pass its actual location to `review_one.py` and set `PATHSNAP_DATA` to the same location for the API.

```powershell
uv venv --python 3.11
uv pip install -r requirements.txt
& .\.venv\Scripts\python.exe review_one.py
& .\.venv\Scripts\python.exe -m uvicorn review_api:app --host 127.0.0.1 --port 8000
```

In another terminal:

```powershell
python -m http.server 8080 --bind 127.0.0.1 --directory evaluator_ui
```

Open [the evaluator UI](http://127.0.0.1:8080) and select `synthetic` or `00000001`. The API terminal prints the request ID, client address, trip, status, and duration. Open `output/review-1/00000001/map.html` for the Folium audit map.

## Two-laptop demonstration

On Laptop 1, generate the fixtures as above. Set the allowed UI origin and bind the API to the LAN:

```powershell
$env:PATHSNAP_UI_ORIGINS = 'http://127.0.0.1:8080'
& .\.venv\Scripts\python.exe -m uvicorn review_api:app --host 0.0.0.0 --port 8000
```

On Laptop 2, serve the `evaluator_ui` directory with the same `http.server` command. Open the UI, enter `http://LAPTOP_1_LAN_IP:8000` in **API address**, and click **Load trips**. Both laptops need to be on the same reachable network, and TCP port 8000 must be reachable on Laptop 1. Select both trips and match each visible result to the request ID and client address in Laptop 1's terminal.

## API examples

`GET /health` replies with `{"status":"ready"}` after fixture generation. `GET /trips` replies with `{"trip_ids":["synthetic","00000001"]}`.

```powershell
$body = @{trip_id = 'synthetic'} | ConvertTo-Json
Invoke-RestMethod http://127.0.0.1:8000/predict -Method Post -ContentType 'application/json' -Body $body
```

The successful response has this shape:

```json
{
  "trip_id": "synthetic",
  "gps": {"type": "MultiPoint", "coordinates": [[8.00005, 49.0], [8.00015, 49.00003]]},
  "predicted_route": {"type": "MultiLineString", "coordinates": [[[8.00005, 49.0], [8.0001, 49.0], [8.00015, 49.0]]]},
  "metrics": {"observation_count": 2, "matched_count": 2, "route_section_count": 1},
  "warnings": []
}
```

An unknown ID returns HTTP 404 with `{"detail":"Unknown development trip"}`. Coordinates are GeoJSON longitude, latitude pairs. A real-trip response also includes aggregate provisional accuracy. The API reads provisional labels after decoding and returns only the aggregate.

See the [Workstream 01 review results](documentation/review-1-data-and-model.md) and the [decision log](documentation/progress-decisions.md).

## Run the trained pipeline on student-test trips

Use the same UI server as above. Set the API mode in the terminal that starts Uvicorn:

```powershell
$env:PATHSNAP_MODE = 'student'
& .\.venv\Scripts\python.exe -m uvicorn review_api:app --host 127.0.0.1 --port 8000
```

The API exposes the ten student-test trips in this mode. It uses `artifacts/model.joblib` by default. To point at a locally trained model, set `PATHSNAP_MODEL` to its path. Use model artifacts produced by this repository. References are loaded for scoring after decoding and the response exposes aggregate metrics.

## Rebuild and evaluate

```powershell
$env:OMP_NUM_THREADS = '2'
$env:OPENBLAS_NUM_THREADS = '2'
& .\.venv\Scripts\python.exe prepare_training.py
& .\.venv\Scripts\python.exe train_models.py --sample 300 --trials 6
& .\.venv\Scripts\python.exe plot_results.py
& .\.venv\Scripts\python.exe evaluate_routes.py --model output/model/model.joblib
```

Run artifacts are written to `output/model/` and `output/evaluation/`. Open `output/evaluation/student-test-map.html` for the Folium overlay of GPS points, candidate roads, decoded route, and the development reference.

## Reserved evaluation mode

After organizers provide the evaluation data, create a local JSON object mapping the ten displayed trip IDs to ten distinct directories. Relative paths resolve against the manifest directory. Each directory should contain the supplied `.track`, `.nodes`, `.arcs` and `.route` files named with that directory's basename. Store the manifest and evaluator data on Laptop 1.

Set `PATHSNAP_MODE=evaluator` and `PATHSNAP_EVALUATOR_MANIFEST` to the manifest path before starting the API with the LAN command above. On Laptop 2, verify all ten IDs appear. Select a trip, inspect the predicted route and aggregate results, and confirm the corresponding request in Laptop 1's terminal. The evaluator-mode contract was exercised with development directories; confirm the organizer's file layout and scoring expectations when the reserved package arrives.
