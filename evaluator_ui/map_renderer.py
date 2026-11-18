"""
Interactive map renderer supporting:
1. Native Leaflet.js embedding (works out of the box in Streamlit via st.components.v1.html)
2. Standalone Folium-compatible HTML export

Complies with GeoJSON specs:
- GeoJSON format: [longitude, latitude]
- Leaflet / Folium format: [latitude, longitude]
"""

import json
from html import escape
from typing import Dict, Any, List

def flip_geojson_coordinates(coords: List[float]) -> List[float]:
    """Flips [longitude, latitude] -> [latitude, longitude]."""
    if len(coords) >= 2:
        return [coords[1], coords[0]]
    return coords


def get_leaflet_map_html(trip_data: Dict[str, Any], height: int = 600) -> str:
    """
    Renders an interactive Leaflet map matching the exact layers and styling
    required by the hackathon evaluation guidelines.
    """
    trip_id = escape(str(trip_data.get("trip_id", "Unknown")))
    gps = trip_data.get("gps", {}).get("coordinates", [])
    pred_segments = trip_data.get("predicted_route", {}).get("coordinates", [])
    ref_segments = trip_data.get("reference_route", {}).get("coordinates", []) if trip_data.get("reference_route") else []

    # Coordinate flip: GeoJSON [lon, lat] -> Leaflet [lat, lon]
    gps_latlng = [flip_geojson_coordinates(pt) for pt in gps]

    pred_latlng_segments = []
    for seg in pred_segments:
        pred_latlng_segments.append([flip_geojson_coordinates(pt) for pt in seg])

    ref_latlng_segments = []
    for seg in ref_segments:
        ref_latlng_segments.append([flip_geojson_coordinates(pt) for pt in seg])

    # Calculate bounding box
    all_points = gps_latlng[:]
    for seg in pred_latlng_segments:
        all_points.extend(seg)
    for seg in ref_latlng_segments:
        all_points.extend(seg)

    if all_points:
        lats = [p[0] for p in all_points]
        lngs = [p[1] for p in all_points]
        bounds = [[min(lats), min(lngs)], [max(lats), max(lngs)]]
        center = [(min(lats) + max(lats)) / 2, (min(lngs) + max(lngs)) / 2]
    else:
        bounds = [[12.96, 77.58], [12.98, 77.60]]
        center = [12.97, 77.59]

    # Check for discontinuities (multiple segments in predicted route)
    has_discontinuity = len(pred_latlng_segments) > 1

    gps_json = json.dumps(gps_latlng)
    pred_json = json.dumps(pred_latlng_segments)
    ref_json = json.dumps(ref_latlng_segments)
    bounds_json = json.dumps(bounds)
    center_json = json.dumps(center)
    gap_json = json.dumps(has_discontinuity)

    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <style>
        html, body {{
            height: 100%;
            margin: 0;
            padding: 0;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background-color: #f8fafc;
        }}
        #map {{
            width: 100%;
            height: {height}px;
            border-radius: 8px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        }}
        .legend {{
            background: white;
            padding: 10px 14px;
            line-height: 22px;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.15);
            font-size: 13px;
        }}
        .legend-title {{
            font-weight: 700;
            margin-bottom: 6px;
            font-size: 14px;
            color: #1e293b;
            border-bottom: 1px solid #e2e8f0;
            padding-bottom: 4px;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            margin-bottom: 5px;
            color: #334155;
            font-size: 12px;
        }}
        .legend-color {{
            width: 22px;
            height: 8px;
            margin-right: 8px;
            display: inline-block;
            border-radius: 2px;
        }}
        .legend-circle {{
            width: 10px;
            height: 10px;
            margin-right: 8px;
            display: inline-block;
            border-radius: 50%;
        }}
    </style>
</head>
<body>
    <div id="map"></div>
    <script>
        var map = L.map('map').setView({center_json}, 14);

        // OpenStreetMap Default Tiles
        var osmLayer = L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
            maxZoom: 19,
            attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
        }}).addTo(map);

        // CartoDB Positron Light Tiles
        var lightLayer = L.tileLayer('https://{{s}}.basemaps.cartocdn.com/light_all/{{z}}/{{x}}/{{y}}{{r}}.png', {{
            maxZoom: 19,
            attribution: '&copy; CartoDB'
        }});

        // Layer Groups
        var gpsLayerGroup = L.layerGroup();
        var predictedRouteGroup = L.layerGroup();
        var referenceRouteGroup = L.layerGroup();
        var markersGroup = L.layerGroup();

        var gpsPoints = {gps_json};
        var predSegments = {pred_json};
        var refSegments = {ref_json};
        var bounds = {bounds_json};

        // 1. Render GPS Points (CircleMarkers with details)
        gpsPoints.forEach(function(pt, idx) {{
            var marker = L.circleMarker(pt, {{
                radius: 5,
                fillColor: "#0284c7",
                color: "#ffffff",
                weight: 1.5,
                opacity: 1,
                fillOpacity: 0.95
            }}).bindPopup("<b>GPS Fix #" + (idx + 1) + "</b><br>Lat: " + pt[0].toFixed(6) + "<br>Lon: " + pt[1].toFixed(6));
            gpsLayerGroup.addLayer(marker);
        }});
        gpsLayerGroup.addTo(map);

        // 2. Render Predicted Route (MultiLineString - respecting gaps)
        predSegments.forEach(function(segment, sIdx) {{
            var polyline = L.polyline(segment, {{
                color: "#2563eb",
                weight: 5,
                opacity: 0.85,
                lineCap: 'round',
                lineJoin: 'round'
            }}).bindPopup("<b>Predicted Route Section " + (sIdx + 1) + "</b>");
            predictedRouteGroup.addLayer(polyline);
        }});
        predictedRouteGroup.addTo(map);

        // 3. Render Reference Route (if available in dev mode)
        if (refSegments.length > 0) {{
            refSegments.forEach(function(segment, sIdx) {{
                var polyline = L.polyline(segment, {{
                    color: "#10b981",
                    weight: 3.5,
                    dashArray: '6, 6',
                    opacity: 0.9
                }}).bindPopup("<b>Ground Truth Reference</b>");
                referenceRouteGroup.addLayer(polyline);
            }});
            referenceRouteGroup.addTo(map);
        }}

        // 4. Start & End Markers
        if (gpsPoints.length > 0) {{
            var startPt = gpsPoints[0];
            var endPt = gpsPoints[gpsPoints.length - 1];

            var startIcon = L.divIcon({{
                className: 'custom-div-icon',
                html: "<div style='background-color:#16a34a;color:white;padding:3px 7px;border-radius:12px;font-size:11px;font-weight:bold;border:2px solid white;box-shadow:0 2px 4px rgba(0,0,0,0.3)'>START</div>",
                iconSize: [50, 20],
                iconAnchor: [25, 10]
            }});

            var endIcon = L.divIcon({{
                className: 'custom-div-icon',
                html: "<div style='background-color:#dc2626;color:white;padding:3px 7px;border-radius:12px;font-size:11px;font-weight:bold;border:2px solid white;box-shadow:0 2px 4px rgba(0,0,0,0.3)'>END</div>",
                iconSize: [45, 20],
                iconAnchor: [22, 10]
            }});

            markersGroup.addLayer(L.marker(startPt, {{ icon: startIcon }}).bindPopup("<b>Trip Start Point</b>"));
            markersGroup.addLayer(L.marker(endPt, {{ icon: endIcon }}).bindPopup("<b>Trip End Point</b>"));
            markersGroup.addTo(map);
        }}

        // Layer Controls
        var baseMaps = {{
            "OpenStreetMap": osmLayer,
            "Light Canvas": lightLayer
        }};

        var overlayMaps = {{
            "Raw GPS Points": gpsLayerGroup,
            "Predicted Route (HMM)": predictedRouteGroup,
            "Trip Endpoints": markersGroup
        }};

        if (refSegments.length > 0) {{
            overlayMaps["Reference Ground Truth"] = referenceRouteGroup;
        }}

        L.control.layers(baseMaps, overlayMaps, {{ collapsed: false, position: 'topright' }}).addTo(map);

        // Map Legend
        var legend = L.control({{ position: 'bottomleft' }});
        legend.onAdd = function (map) {{
            var div = L.DomUtil.create('div', 'legend');
            var html = '<div class="legend-title">Trip: {trip_id}</div>';
            html += '<div class="legend-item"><span class="legend-circle" style="background:#0284c7;border:1px solid white"></span> GPS Observations</div>';
            html += '<div class="legend-item"><span class="legend-color" style="background:#2563eb"></span> Predicted Route</div>';
            if (refSegments.length > 0) {{
                html += '<div class="legend-item"><span class="legend-color" style="background:#10b981;border-bottom:2px dashed #059669"></span> Reference Route</div>';
            }}
            if ({gap_json}) {{
                html += '<div class="legend-item" style="color:#b91c1c;font-weight:600;">&#9888; Route contains gaps</div>';
            }}
            div.innerHTML = html;
            return div;
        }};
        legend.addTo(map);

        // Auto-fit bounds
        if (bounds && bounds.length === 2 && bounds[0][0] !== bounds[1][0]) {{
            map.fitBounds(bounds, {{ padding: [35, 35] }});
        }}
    </script>
</body>
</html>
"""
    return html_content


def export_folium_html(trip_data: Dict[str, Any], output_path: str) -> str:
    """Exports interactive HTML map file suitable for report attachments and submission."""
    html_content = get_leaflet_map_html(trip_data, height=750)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    return output_path
