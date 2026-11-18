"""
Reproducible Folium Map Generator.
Generates an interactive HTML map for any development or evaluated trip.

Usage:
    python generate_folium_map.py --trip 001 --output trip_001_map.html
    python generate_folium_map.py --trip 002 --output trip_002_gap_map.html
"""

import argparse
import sys
import os

# Add root directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from evaluator_ui.map_renderer import export_folium_html
from evaluator_ui.fixtures.sample_trips import MOCK_TRIPS
from evaluator_ui.api_client import BackendAPIClient

def main():
    parser = argparse.ArgumentParser(description="Generate Folium / Leaflet Route Reconstruction Map")
    parser.add_argument("--trip", type=str, default="001", help="Trip ID to render (e.g. 001, 002)")
    parser.add_argument("--output", type=str, default="route_map.html", help="Output HTML file path")
    parser.add_argument("--api", type=str, default=None, help="Optional FastAPI backend URL (e.g. http://localhost:8000)")
    args = parser.parse_args()

    trip_id = str(args.trip)
    print(f"Generating map for Trip '{trip_id}'...")

    if args.api:
        client = BackendAPIClient(base_url=args.api)
        try:
            trip_data = client.predict_trip(trip_id)
        except (ConnectionError, ValueError) as error:
            parser.error(str(error))
    else:
        if trip_id not in MOCK_TRIPS:
            parser.error(f"Unknown local fixture: {trip_id}")
        trip_data = MOCK_TRIPS[trip_id]

    out_file = export_folium_html(trip_data, args.output)
    print(f"✔ Interactive HTML map successfully saved to: {os.path.abspath(out_file)}")

if __name__ == "__main__":
    main()
