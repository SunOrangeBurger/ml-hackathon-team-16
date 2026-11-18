# PathSnap

Review 1 connects the candidate-score fixtures, directed route decoder, FastAPI service and Streamlit evaluator.

## Run the Review 1 demo

Use Python 3.11 or later in a virtual environment on each laptop. From the repository root, install dependencies:

```powershell
python -m pip install -r requirements.txt -r requirements-ui.txt
python -m pytest -q
```

On Laptop 1, start the API in decoder mode:

```powershell
$env:PATHSNAP_MODE = 'decoder'
python -m uvicorn backend.app:app --host 0.0.0.0 --port 8000
```

On Laptop 2, start the evaluator:

```powershell
python -m streamlit run evaluator_ui/app.py
```

Open the evaluator at http://localhost:8501. Select **Live Backend API (Laptop 1)**, enter the receiving laptop's LAN address as `http://<laptop-1-ip>:8000`, and select `synthetic`. Use **Run Prediction / Decode** to show the GPS points, connected route and match counts. Check Laptop 1's terminal for the prediction request ID, client address, trip ID, status and duration. For a single-machine run, use `http://127.0.0.1:8000` instead. The two-laptop run requires both machines to be on a reachable network and port 8000 to be reachable on Laptop 1.

## Add the training trip

Run these commands on Laptop 1 before starting Uvicorn. Use the supplied dataset's root directory containing `train` and `student_test`.

```powershell
$env:PATHSNAP_DATA = 'Student folder/Student folder'
python review_one.py --data-dir "$env:PATHSNAP_DATA"
$env:PATHSNAP_DECODER_FIXTURE = 'output/review-1/00000001/decoder_fixture.json'
```

Trip `00000001` then appears alongside `synthetic`. The API decodes development distance scores and reports observation, match and route-section counts. [Workstream 01 results](documentation/review-1-data-and-model.md) describe the provisional labels and baseline accuracy. Use the evaluator's HTML download or the [Folium export command](evaluator_ui/README.md) to save a map.

The [backend guide](documentation/03-fastapi-backend.md) documents the original precomputed fixture mode and final-evaluation requirements. The trained pipeline was developed on the backup implementation branch. The official reserved-data demonstration requires the organizer's dataset.
