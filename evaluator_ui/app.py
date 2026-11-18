"""
Streamlit Evaluator UI for Hackathon Review 1 and Final Evaluation.
Conforms to documentation/04-evaluator-ui.md.

Run with:
    streamlit run evaluator_ui/app.py
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
        font-size: 24px;
        font-weight: 700;
        color: #0f172a;
    }
    .metric-lbl {
        font-size: 13px;
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
    .badge-offline {
        background-color: #fee2e2;
        color: #b91c1c;
        padding: 4px 8px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 12px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Session State
if "api_url" not in st.session_state:
    st.session_state.api_url = "http://localhost:8000"
if "use_mock" not in st.session_state:
    st.session_state.use_mock = True
if "active_trip_id" not in st.session_state:
    st.session_state.active_trip_id = "001"
if "last_result" not in st.session_state:
    st.session_state.last_result = None
if "fetch_error" not in st.session_state:
    st.session_state.fetch_error = None

# --- SIDEBAR: API & CONFIGURATION ---
st.sidebar.title("⚙️ Evaluator Controls")
st.sidebar.markdown("---")

# Mode Selection
mode = st.sidebar.radio(
    "Data Source Mode",
    ["Offline Development Fixtures (Review 1 Demo)", "Live Backend API (Laptop 1)"],
    index=0 if st.session_state.use_mock else 1,
    help="Switch between local agreed fixtures and live FastAPI server on Laptop 1."
)
st.session_state.use_mock = (mode == "Offline Development Fixtures (Review 1 Demo)")

# API Configuration
st.sidebar.subheader("🌐 Backend Connection")
api_url_input = st.sidebar.text_input(
    "FastAPI Base URL",
    value=st.session_state.api_url,
    help="e.g. http://localhost:8000 or http://192.168.1.50:8000"
)
st.session_state.api_url = api_url_input.strip()

client = BackendAPIClient(base_url=st.session_state.api_url)

# Connectivity Check
col_ping1, col_ping2 = st.sidebar.columns([1, 1])
with col_ping1:
    test_btn = st.button("🔌 Ping API", use_container_width=True)

if test_btn:
    with st.spinner("Pinging /health..."):
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
    available_trips = ["001", "002", "003"]

st.sidebar.markdown("---")
st.sidebar.subheader("🚗 Trip Selection")

selected_trip = st.sidebar.selectbox(
    "Select Trip ID",
    options=available_trips,
    index=0 if st.session_state.active_trip_id not in available_trips else available_trips.index(st.session_state.active_trip_id),
    help="Preserves string IDs with leading zeros (e.g. 001)."
)

# Show description if using mock data
if selected_trip in MOCK_TRIPS:
    st.sidebar.caption(f"ℹ️ {MOCK_TRIPS[selected_trip].get('description', '')}")

fetch_trigger = st.sidebar.button("🚀 Run Prediction / Decode", type="primary", use_container_width=True)

# Clear error or fetch new result when selection changes or button clicked
if selected_trip != st.session_state.active_trip_id or fetch_trigger or st.session_state.last_result is None:
    st.session_state.active_trip_id = selected_trip
    st.session_state.fetch_error = None
    with st.spinner(f"Requesting inference for Trip '{selected_trip}'..."):
        try:
            # Simulate real network timing for smooth UX
            if st.session_state.use_mock:
                time.sleep(0.15)
            result = client.predict_trip(selected_trip, use_mock=st.session_state.use_mock)
            st.session_state.last_result = result
        except Exception as err:
            st.session_state.fetch_error = str(err)
            st.session_state.last_result = None

# --- MAIN CONTENT AREA ---
st.title("🗺️ Route Reconstruction Evaluator UI")
st.caption("Review 1 Milestone • Map-Matching & Viterbi Sequence Decoder Verification")

# Top Status Banner
col_status1, col_status2, col_status3 = st.columns([2, 1, 1])
with col_status1:
    st.markdown(f"**Active Selection:** Trip `{st.session_state.active_trip_id}` | Mode: `{mode}`")
with col_status2:
    if st.session_state.use_mock:
        st.markdown('<span class="badge-online">✔ Mock Fixture Ready</span>', unsafe_allow_html=True)
    else:
        st.markdown(f'<span class="badge-online">Live API: {st.session_state.api_url}</span>', unsafe_allow_html=True)
with col_status3:
    if st.button("🔄 Refresh Data", use_container_width=True):
        st.rerun()

st.markdown("---")

# Handle Error State
if st.session_state.fetch_error:
    st.error(f"❌ Error during prediction retrieval: {st.session_state.fetch_error}")
    st.markdown("""
    **Troubleshooting Checklist:**
    1. Verify that Laptop 1 is running `uvicorn` / `mock_backend.py` with `--host 0.0.0.0`.
    2. Check that Laptop 2 and Laptop 1 are connected to the same Wi-Fi / Hotspot.
    3. Ensure Windows Defender Firewall allows inbound traffic on the designated port.
    4. Or toggle back to **Offline Development Fixtures** in the sidebar.
    """)
    if st.button("🔁 Retry Request"):
        st.rerun()
    st.stop()

# If Result is Available
trip_data = st.session_state.last_result
if trip_data:
    # Check for Discontinuities / Warnings
    warnings = trip_data.get("warnings", [])
    if warnings:
        with st.expander("⚠️ Server Warnings & Discontinuities Detected", expanded=True):
            for w in warnings:
                st.warning(f"• {w}")

    # Layout: Map on the left (wide), Metrics on the right
    col_map, col_metrics = st.columns([3, 1])

    with col_map:
        st.subheader("Interactive Trajectory & Route Map")
        map_html = get_leaflet_map_html(trip_data, height=580)
        components.html(map_html, height=600)

        # Export map option
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
            st.info("🔒 **Metrics Unavailable**\n\nReserved Evaluator Trip. Ground truth is withheld on server to prevent data leakage.")
        else:
            acc = metrics.get("top1_accuracy")
            f1 = metrics.get("macro_f1")
            cont = metrics.get("route_continuity")
            cov = metrics.get("candidate_coverage")
            dist = metrics.get("arc_edit_distance")

            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-lbl">Pointwise Top-1 Acc</div>
                <div class="metric-val">{(acc * 100):.1f}%</div>
            </div>
            <div class="metric-card">
                <div class="metric-lbl">Macro F1 Score</div>
                <div class="metric-val">{f1:.3f}</div>
            </div>
            <div class="metric-card">
                <div class="metric-lbl">Route Continuity</div>
                <div class="metric-val">{(cont * 100):.1f}%</div>
            </div>
            <div class="metric-card">
                <div class="metric-lbl">Candidate Coverage</div>
                <div class="metric-val">{(cov * 100):.1f}%</div>
            </div>
            <div class="metric-card">
                <div class="metric-lbl">Arc Edit Distance</div>
                <div class="metric-val">{dist}</div>
            </div>
            """, unsafe_allow_html=True)

        # Summary Information
        gps_coords = trip_data.get("gps", {}).get("coordinates", [])
        pred_routes = trip_data.get("predicted_route", {}).get("coordinates", [])
        num_segments = len(pred_routes)

        st.markdown("---")
        st.markdown(f"**GPS Points:** `{len(gps_coords)}` fixes")
        st.markdown(f"**Route Sections:** `{num_segments}` segment(s)")
        if num_segments > 1:
            st.markdown("🚨 *Route has discontinuities across time gaps.*")

    # Debug / Raw Payload Expander
    with st.expander("🔍 View Raw GeoJSON API Payload"):
        st.json(trip_data)

st.markdown("---")
st.caption("Team 16 • Machine Learning Hackathon • Review 1 Milestone")
