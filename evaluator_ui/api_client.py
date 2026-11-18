"""
API Client for Evaluator UI to communicate with FastAPI backend on Laptop 1.
Conforms to schemas defined in documentation/03-fastapi-backend.md.
"""

import time
import requests
from typing import Dict, Any, List, Tuple
from evaluator_ui.fixtures.sample_trips import MOCK_TRIPS

class BackendAPIClient:
    def __init__(self, base_url: str = "http://localhost:8000", timeout: float = 6.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def check_health(self) -> Tuple[bool, str, float]:
        """
        Calls GET /health.
        Returns: (is_healthy, status_message, latency_ms)
        """
        start = time.perf_counter()
        try:
            resp = requests.get(f"{self.base_url}/health", timeout=self.timeout)
            latency_ms = (time.perf_counter() - start) * 1000
            if resp.status_code == 200:
                data = resp.json()
                status = data.get("status", "ready") if isinstance(data, dict) else "ready"
                return True, f"Online ({status})", latency_ms
            return False, f"Server returned HTTP {resp.status_code}", latency_ms
        except requests.exceptions.ConnectionError:
            return False, f"Connection refused to {self.base_url}. Ensure server is running on Laptop 1 with 0.0.0.0 host binding and Windows Firewall allows inbound connections on port.", 0.0
        except requests.exceptions.Timeout:
            return False, f"Request timed out after {self.timeout}s.", 0.0
        except Exception as e:
            return False, f"Connection error: {str(e)}", 0.0

    def get_trips(self, use_mock: bool = False) -> List[str]:
        """
        Fetches trip list from GET /trips.
        Returns string trip IDs preserving leading zeros.
        """
        if use_mock:
            return sorted(list(MOCK_TRIPS.keys()))

        try:
            resp = requests.get(f"{self.base_url}/trips", timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()
            if isinstance(data, list):
                return [str(t) for t in data]
            if isinstance(data, dict):
                trip_ids = data.get("trip_ids", data.get("trips"))
                if isinstance(trip_ids, list):
                    return [str(trip_id) for trip_id in trip_ids]
            raise ValueError("Unexpected response format from GET /trips")
        except Exception as e:
            raise RuntimeError(f"Failed to fetch trips from {self.base_url}/trips: {str(e)}")

    def predict_trip(self, trip_id: str, use_mock: bool = False) -> Dict[str, Any]:
        """
        Sends selected ID to POST /predict.
        Validates request and response schemas. Rejects stale responses.
        """
        trip_id_str = str(trip_id)
        if use_mock:
            if trip_id_str in MOCK_TRIPS:
                return MOCK_TRIPS[trip_id_str]
            # Fallback for unexpected ID in mock mode
            return {
                "trip_id": trip_id_str,
                "gps": {"type": "MultiPoint", "coordinates": [[77.5900, 12.9700], [77.5950, 12.9750]]},
                "predicted_route": {"type": "MultiLineString", "coordinates": [[[77.5898, 12.9698], [77.5948, 12.9748]]]},
                "metrics": None,
                "warnings": ["Generated from fallback mock."]
            }

        payload = {"trip_id": trip_id_str}
        try:
            resp = requests.post(f"{self.base_url}/predict", json=payload, timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()

            # Prevent stale responses when user quickly changes selection
            ret_id = str(data.get("trip_id", ""))
            if ret_id and ret_id != trip_id_str:
                raise ValueError(f"Stale response rejected: Requested trip '{trip_id_str}' but received '{ret_id}'.")

            return data
        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"Prediction request failed for trip {trip_id_str}: {str(e)}")
