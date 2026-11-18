"""
Unit tests for Evaluator UI components, map renderer, and API client.
"""

import unittest
import sys
import os

# Ensure package is on python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from evaluator_ui.fixtures.sample_trips import MOCK_TRIPS
from evaluator_ui.map_renderer import flip_geojson_coordinates, get_leaflet_map_html
from evaluator_ui.api_client import BackendAPIClient

class TestEvaluatorUI(unittest.TestCase):

    def test_coordinate_flip(self):
        """Verifies that GeoJSON [longitude, latitude] flips to Leaflet [latitude, longitude]."""
        geojson_point = [77.5900, 12.9700]  # [lon, lat]
        leaflet_point = flip_geojson_coordinates(geojson_point)
        self.assertEqual(leaflet_point, [12.9700, 77.5900])  # [lat, lon]

    def test_sample_fixtures_schema(self):
        """Checks that all sample fixtures adhere to the agreed contract."""
        for tid, trip in MOCK_TRIPS.items():
            self.assertIn("trip_id", trip)
            self.assertEqual(str(trip["trip_id"]), tid)
            self.assertIn("gps", trip)
            self.assertEqual(trip["gps"]["type"], "MultiPoint")
            self.assertIn("predicted_route", trip)
            self.assertEqual(trip["predicted_route"]["type"], "MultiLineString")
            self.assertIn("warnings", trip)

    def test_map_renderer_html_generation(self):
        """Verifies map renderer outputs valid HTML with required layer definitions."""
        trip = MOCK_TRIPS["001"]
        html = get_leaflet_map_html(trip)
        self.assertIn("L.map", html)
        self.assertIn("Raw GPS Points", html)
        self.assertIn("Predicted Route (HMM)", html)
        self.assertIn("Trip: 001", html)

    def test_discontinuity_rendering(self):
        """Checks that trip 002 with gaps generates multiple line segments without joining them."""
        trip = MOCK_TRIPS["002"]
        html = get_leaflet_map_html(trip)
        self.assertIn("Route contains gaps", html)
        self.assertEqual(len(trip["predicted_route"]["coordinates"]), 2)

    def test_api_client_mock_mode(self):
        """Verifies that BackendAPIClient operates properly in mock mode."""
        client = BackendAPIClient("http://localhost:8000")
        trips = client.get_trips(use_mock=True)
        self.assertIn("001", trips)
        self.assertIn("002", trips)
        self.assertIn("003", trips)

        res = client.predict_trip("001", use_mock=True)
        self.assertEqual(res["trip_id"], "001")
        self.assertIsNotNone(res["metrics"])

    def test_evaluator_trip_null_metrics(self):
        """Verifies that evaluator trip handles null metrics correctly."""
        trip = MOCK_TRIPS["003"]
        self.assertIsNone(trip["metrics"])

if __name__ == "__main__":
    unittest.main()
