"""
Streamlit Evaluator UI for Route Reconstruction Hackathon.
Satisfies both Review 1 and Final Evaluation requirements:
- documentation/04-evaluator-ui.md
- documentation/03-fastapi-backend.md

Run with:
    python -m streamlit run evaluator_ui/app.py
"""

import streamlit as st
import streamlit.components.v1 as components
import json
import time

from evaluator_ui.api_client import BackendAPIClient
from evaluator_ui.map_renderer import get_leaflet_map_html
from evaluator_ui.fixtures.sample_trips import MOCK_TRIPS

# Page configuration
st.set_page_config(
    page_title="Route Reconstruction Evaluator",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 10px;
    }
    .metric-val {
        font-size: 22px;
        font-weight: 700;
        color: #0f172a;
    }
    .metric-lbl {
        font-size: 12px;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .badge-online {
        background-color: #dcfce7;
        color: #15803d;
        padding: 4px 8px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 12px;
    }
    .badge-evaluator {
        background-color: #ede9fe;
        color: #6d28d9;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 12px;
        border: 1px solid #ddd6fe;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Session State
if "api_url" not in st.session_state:
    st.session_state.api_url = "http://localhost:8000"
if "use_mock" not in st.session_state:
    st.session_state.use_mock = True
if "active_trip_id" not in st.session_state:
    st.session_state.active_trip_id = "00000001"
if "last_result" not in st.session_state:
    st.session_state.last_result = None
if "last_latency" not in st.session_state:
    st.session_state.last_latency = 0.0
if "fetch_error" not in st.session_state:
    st.session_state.fetch_error = None

# --- SIDEBAR: API & CONFIGURATION ---
st.sidebar.title("⚙️ Evaluator Controls")
st.sidebar.markdown("---")

# Mode Selection
mode = st.sidebar.radio(
    "Data Source Mode",
    ["Development & Reserved Fixtures (Demo / Rehearsal)", "Live Backend API (Laptop 1)"],
    index=0 if st.session_state.use_mock else 1,
    help="Switch between local fixtures (including all 10 reserved evaluator trips) and live FastAPI server on Laptop 1."
)
st.session_state.use_mock = (mode == "Development & Reserved Fixtures (Demo / Rehearsal)")

# API Configuration
st.sidebar.subheader("🌐 Backend Connection")
api_url_input = st.sidebar.text_input(
    "FastAPI Base URL",
    value=st.session_state.api_url,
    help="e.g. http://localhost:8000 or http://192.168.1.15:8000"
)
st.session_state.api_url = api_url_input.strip()

client = BackendAPIClient(base_url=st.session_state.api_url)

# Connectivity Check
col_ping1, col_ping2 = st.sidebar.columns([1, 1])
with col_ping1:
    test_btn = st.button("🔌 Ping API", use_container_width=True)

if test_btn:
    with st.spinner("Pinging /health on Laptop 1..."):
        ok, msg, latency = client.check_health()
        if ok:
            st.sidebar.success(f"Connected! ({latency:.1f} ms)\n{msg}")
        else:
            st.sidebar.error(f"Failed: {msg}")

# Fetch Trips
try:
    available_trips = client.get_trips(use_mock=st.session_state.use_mock)
except Exception as e:
    st.sidebar.warning(f"Could not load trips from API: {str(e)}")
    available_trips = ["00000001", "00000002", "synthetic"]

st.sidebar.markdown("---")
st.sidebar.subheader("🚗 Trip Selection")

# Format helper for trip labels
def format_trip_label(tid: str) -> str:
    if tid in MOCK_TRIPS and "title" in MOCK_TRIPS[tid]:
        return MOCK_TRIPS[tid]["title"]
    if tid.isdigit() and int(tid) >= 91:
        return f"Trip {tid} (Reserved Evaluator #{int(tid)-90})"
    return f"Trip {tid}"

selected_trip = st.sidebar.selectbox(
    "Select Trip ID (Development / Reserved)",
    options=available_trips,
    format_func=format_trip_label,
    index=0 if st.session_state.active_trip_id not in available_trips else available_trips.index(st.session_state.active_trip_id),
    help="Preserves string IDs with leading zeros."
)

if selected_trip in MOCK_TRIPS:
    st.sidebar.caption(f"ℹ️ {MOCK_TRIPS[selected_trip].get('description', '')}")

fetch_trigger = st.sidebar.button("🚀 Request Prediction / Decode", type="primary", use_container_width=True)

# Update state and fetch prediction
if selected_trip != st.session_state.active_trip_id or fetch_trigger or st.session_state.last_result is None:
    st.session_state.active_trip_id = selected_trip
    st.session_state.fetch_error = None
    with st.spinner(f"Requesting prediction for Trip '{selected_trip}'..."):
        try:
            result, latency = client.predict_trip(selected_trip, use_mock=st.session_state.use_mock)
            st.session_state.last_result = result
            st.session_state.last_latency = latency
        except Exception as err:
            st.session_state.fetch_error = str(err)
            st.session_state.last_result = None

# --- MAIN CONTENT AREA ---
st.title("🗺️ Route Reconstruction Evaluator UI")
st.caption("Machine Learning Hackathon • Map-Matching & Viterbi Route Reconstruction Evaluation")

# Status Bar
col_status1, col_status2, col_status3 = st.columns([2, 1, 1])
with col_status1:
    is_eval_trip = str(st.session_state.active_trip_id).isdigit() and int(st.session_state.active_trip_id) >= 91
    badge_type = '<span class="badge-evaluator">🔒 Final Evaluation Reserved Trip</span>' if is_eval_trip else '<span class="badge-online">✔ Development Mode</span>'
    st.markdown(f"**Trip:** `{st.session_state.active_trip_id}` &nbsp; {badge_type}", unsafe_allow_html=True)
with col_status2:
    if st.session_state.use_mock:
        st.markdown('<span class="badge-online">✔ Offline Fixtures Ready</span>', unsafe_allow_html=True)
    else:
        st.markdown(f'<span class="badge-online">Live API ({st.session_state.last_latency:.1f} ms)</span>', unsafe_allow_html=True)
with col_status3:
    if st.button("🔄 Refresh View", use_container_width=True):
        st.rerun()

st.markdown("---")

# Handle Error State
if st.session_state.fetch_error:
    st.error(f"❌ Error during prediction retrieval: {st.session_state.fetch_error}")
    st.markdown("""
    **Connection & Diagnostic Steps:**
    1. Verify that Laptop 1 is running `uvicorn backend.app:app` or `mock_backend.py` with `--host 0.0.0.0`.
    2. Ensure Laptop 1 and Laptop 2 are connected to the same Wi-Fi / hotspot.
    3. Check Windows Firewall on Laptop 1 for inbound port permissions.
    4. You can also switch back to **Development & Reserved Fixtures** in the sidebar to rehearse offline.
    """)
    if st.button("🔁 Retry Connection"):
        st.rerun()
    st.stop()

# Render Result
trip_data = st.session_state.last_result
if trip_data:
    warnings = trip_data.get("warnings", [])
    if warnings:
        with st.expander("⚠️ Server Warnings & Discontinuities", expanded=True):
            for w in warnings:
                st.warning(f"• {w}")

    # Layout: Map on the left (wide), Metrics on the right
    col_map, col_metrics = st.columns([3, 1])

    with col_map:
        st.subheader("Interactive Trajectory & Route Map")
        map_html = get_leaflet_map_html(trip_data, height=580)
        components.html(map_html, height=600)

        st.download_button(
            label="💾 Download Interactive HTML Map",
            data=map_html,
            file_name=f"trip_{st.session_state.active_trip_id}_map.html",
            mime="text/html"
        )

    with col_metrics:
        st.subheader("Evaluation Metrics")
        metrics = trip_data.get("metrics")

        if metrics is None:
            st.info("""
            🔒 **Reserved Evaluator Mode**

            Ground-truth geometry and labels are strictly withheld on the server to prevent data leakage during official evaluation.
            
            Aggregate sequence scoring is calculated server-side.
            """)
        else:
            acc = metrics.get("top1_accuracy")
            f1 = metrics.get("macro_f1")
            cont = metrics.get("route_continuity")
            cov = metrics.get("candidate_coverage")
            dist = metrics.get("arc_edit_distance")

            if acc is not None:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-lbl">Pointwise Top-1 Acc</div>
                    <div class="metric-val">{(acc * 100):.1f}%</div>
                </div>
                """, unsafe_allow_html=True)
            if f1 is not None:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-lbl">Macro F1 Score</div>
                    <div class="metric-val">{f1:.3f}</div>
                </div>
                """, unsafe_allow_html=True)
            if cont is not None:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-lbl">Route Continuity</div>
                    <div class="metric-val">{(cont * 100):.1f}%</div>
                </div>
                """, unsafe_allow_html=True)
            if cov is not None:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-lbl">Candidate Coverage</div>
                    <div class="metric-val">{(cov * 100):.1f}%</div>
                </div>
                """, unsafe_allow_html=True)
            if dist is not None:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-lbl">Arc Edit Distance</div>
                    <div class="metric-val">{dist}</div>
                </div>
                """, unsafe_allow_html=True)

        # Telemetry Summary
        gps_coords = trip_data.get("gps", {}).get("coordinates", [])
        pred_routes = trip_data.get("predicted_route", {}).get("coordinates", [])
        num_segments = len(pred_routes)

        st.markdown("---")
        st.markdown(f"**GPS Points:** `{len(gps_coords)}` fixes")
        st.markdown(f"**Route Sections:** `{num_segments}` segment(s)")
        if num_segments > 1:
            st.markdown("🚨 *Route has discontinuities across time gaps.*")
        st.markdown(f"**Round-trip Latency:** `{st.session_state.last_latency:.1f} ms`")

    # Debug / Raw Payload Expander
    with st.expander("🔍 View Raw GeoJSON API Payload"):
        st.json(trip_data)

st.markdown("---")
st.caption("Team 16 • Machine Learning Hackathon • Evaluator Interface")
