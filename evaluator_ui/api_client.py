"""
API Client for Evaluator UI to communicate with FastAPI backend on Laptop 1.
Conforms to schemas defined in:
- documentation/03-fastapi-backend.md
- documentation/04-evaluator-ui.md
"""

import time
import requests
from typing import Dict, Any, List, Tuple
from evaluator_ui.fixtures.sample_trips import MOCK_TRIPS

class BackendAPIClient:
    def __init__(self, base_url: str = "http://localhost:8000", timeout: float = 8.0):
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
                mode = data.get("mode", "production") if isinstance(data, dict) else ""
                return True, f"Online ({status}, mode={mode})", latency_ms
            elif resp.status_code == 503:
                data = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {}
                detail = data.get("detail", "Prediction service is initializing or not ready.")
                return False, f"Server 503 Not Ready: {detail}", latency_ms
            return False, f"Server returned HTTP {resp.status_code}", latency_ms
        except requests.exceptions.ConnectionError:
            return False, f"Connection refused to {self.base_url}. Ensure Laptop 1 backend is listening on 0.0.0.0 and firewall permits port.", 0.0
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
            # Return ordered list: dev trips first, then the 10 reserved evaluator trips
            dev_ids = ["00000001", "00000002", "synthetic"]
            eval_ids = [f"{num:08d}" for num in range(91, 101)]
            return dev_ids + eval_ids

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
        except requests.exceptions.HTTPError as err:
            try:
                err_json = resp.json()
                msg = err_json.get("error", {}).get("message", str(err))
                raise RuntimeError(f"Server error: {msg}")
            except Exception:
                raise RuntimeError(f"HTTP error {resp.status_code} from {self.base_url}/trips")
        except Exception as e:
            raise RuntimeError(f"Failed to fetch trips from {self.base_url}/trips: {str(e)}")

    def predict_trip(self, trip_id: str, use_mock: bool = False) -> Tuple[Dict[str, Any], float]:
        """
        Sends selected ID to POST /predict.
        Validates request and response schemas. Rejects stale responses.
        Returns: (response_dict, latency_ms)
        """
        trip_id_str = str(trip_id)
        if use_mock:
            time.sleep(0.08)  # Smooth UX simulation
            if trip_id_str in MOCK_TRIPS:
                return MOCK_TRIPS[trip_id_str], 80.0
            # Fallback for unexpected ID in mock mode
            return {
                "trip_id": trip_id_str,
                "gps": {"type": "MultiPoint", "coordinates": [[77.5900, 12.9700], [77.5950, 12.9750]]},
                "predicted_route": {"type": "MultiLineString", "coordinates": [[[77.5898, 12.9698], [77.5948, 12.9748]]]},
                "metrics": None,
                "warnings": ["Generated from fallback mock."]
            }, 80.0

        payload = {"trip_id": trip_id_str}
        start = time.perf_counter()
        try:
            resp = requests.post(f"{self.base_url}/predict", json=payload, timeout=self.timeout)
            latency_ms = (time.perf_counter() - start) * 1000

            if not resp.ok:
                try:
                    err_data = resp.json()
                    err_msg = err_data.get("error", {}).get("message", resp.text)
                    code = err_data.get("error", {}).get("code", f"http_{resp.status_code}")
                    raise ValueError(f"[{code}] {err_msg}")
                except Exception as parse_err:
                    if isinstance(parse_err, ValueError):
                        raise
                    resp.raise_for_status()

            data = resp.json()

            # Stale response prevention: verify returned trip_id matches selected
            ret_id = str(data.get("trip_id", ""))
            if ret_id and ret_id != trip_id_str:
                raise ValueError(f"Stale response rejected: Requested trip '{trip_id_str}' but received '{ret_id}'.")

            return data, latency_ms
        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"Prediction request failed for trip {trip_id_str}: {str(e)}")
