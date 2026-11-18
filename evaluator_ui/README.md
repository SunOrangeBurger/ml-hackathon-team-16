# Evaluator UI and Route Reconstruction Visualizations

This module implements the complete **Evaluator UI and Folium Visualizations** for the Machine Learning Hackathon (Module 04).

See the [project quick start](../README.md) for the integrated Review 1 run. The physical two-laptop demonstration is a separate verification step.

---

## 🎯 Review 1 Requirements Coverage

| Requirement | Implementation & Where to Verify |
| :--- | :--- |
| **Demonstrate selectable trip, GPS trace, predicted route & results** | Select trips (`001`, `002`, `003`) from sidebar. UI loads GeoJSON MultiPoint (GPS) and MultiLineString (route) from agreed API fixture and calculates metrics. |
| **Show distinct layers and legend** | Top-right layer toggle: `Raw GPS Points`, `Predicted Route (HMM)`, `Reference Ground Truth`, `Trip Endpoints`. Floating on-map legend shows symbols and color keys. |
| **Confirm coordinate order** | Tested & verified in `tests/test_ui_and_renderer.py`. GeoJSON `[lon, lat]` is properly flipped to Leaflet/Folium `[lat, lon]`. |
| **Exercise loading and error states** | Visual spinner on request, connection timeout detection, stale response rejection, and an interactive "Retry Request" button with network troubleshooting hints. |
| **Confirm Laptop 2 can call API on Laptop 1** | API Client with configurable base URL (e.g., `http://<laptop-1-ip>:8000`) and a dedicated **"Ping API"** button measuring round-trip latency. Included `mock_backend.py` allows testing cross-laptop calling right away. |

---

## 🚀 Quick Start Guide

### 1. Install Dependencies
```bash
pip install -r requirements-ui.txt
```

### 2. Launch the Evaluator UI (Laptop 2)
```bash
streamlit run evaluator_ui/app.py
```
Open your browser at `http://localhost:8501`.

---

## 🧪 Demonstrating Review 1 (Two Modes)

### Mode A: Offline Development Fixtures (Standalone on Laptop 2)
1. In the sidebar, select **"Offline Development Fixtures (Review 1 Demo)"**.
2. Pick any trip:
   - **Trip 001**: Clean urban corridor with 95.4% accuracy and continuous route.
   - **Trip 002**: Demonstrates a 92-second timestamp gap. The predicted route is rendered as two disconnected line strings without joining across the gap.
   - **Trip 003**: Evaluator trip where ground truth is withheld server-side (metrics display clear "Unavailable" badge).
3. Toggle the map layers in the top-right corner to show/hide GPS fixes and route overlays.

### Mode B: Live Cross-Laptop Demonstration (Laptop 1 ↔ Laptop 2)

#### Step 1: Start Backend on Laptop 1
On Laptop 1, start the merged FastAPI backend in decoder mode:
```bash
PATHSNAP_MODE=decoder python -m uvicorn backend.app:app --host 0.0.0.0 --port 8000
```
For PowerShell and the optional training trip, follow the [project quick start](../README.md).
> **Note**: Find Laptop 1's local IP address by running `ipconfig` (Windows) or `ifconfig` (Linux/Mac). Example IP: `192.168.1.15`.
> Ensure Laptop 1's firewall permits inbound TCP on port 8000.

#### Step 2: Connect from Laptop 2
1. On Laptop 2, open the Streamlit Evaluator UI.
2. In the sidebar, select **"Live Backend API (Laptop 1)"**.
3. Enter the FastAPI Base URL: `http://192.168.1.15:8000`.
4. Click **"🔌 Ping API"**. You will see:
   ```
   Connected! (12.4 ms) Online (ready)
   ```
5. Select `synthetic` and click **"Run Prediction / Decode"**. Check the route and counts in the UI and the corresponding request in Laptop 1's terminal.

---

## 🗺️ Standalone Folium Map Generation (For Reports)

To generate standalone interactive `.html` maps for report submissions:
```bash
# Generate map from fixture
python evaluator_ui/generate_folium_map.py --trip 001 --output trip_001_map.html

# Generate map from live backend on Laptop 1
python evaluator_ui/generate_folium_map.py --trip synthetic --api http://192.168.1.15:8000 --output synthetic_live.html
```

---

## 🧪 Running Unit Tests
```bash
python -m unittest evaluator_ui/tests/test_ui_and_renderer.py
```
All tests verify coordinate order transformations, schema compatibility, discontinuity rendering, and API client validation.
