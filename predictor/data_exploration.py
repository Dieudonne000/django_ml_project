from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


BASE_DIR = Path(__file__).resolve().parents[1]
GEOJSON_PATH = BASE_DIR / "dummy-data" / "rwanda_districts.geojson"

DISTRICT_COORDS: dict[str, tuple[float, float]] = {
    "Nyarugenge": (-1.9536, 29.8739),
    "Gasabo": (-1.8883, 30.0911),
    "Kicukiro": (-1.9958, 30.0811),
    "Nyanza": (-2.3517, 29.7506),
    "Gisagara": (-2.5867, 29.7),
    "Nyaruguru": (-2.57, 29.5124),
    "Huye": (-2.5947, 29.7394),
    "Nyamagabe": (-2.43, 29.49),
    "Ruhango": (-2.2167, 29.7833),
    "Muhanga": (-2.0833, 29.75),
    "Kamonyi": (-2.0, 29.8333),
    "Karongi": (-2.1333, 29.3667),
    "Rutsiro": (-2.0833, 29.25),
    "Rubavu": (-1.6833, 29.3167),
    "Nyabihu": (-1.7333, 29.5),
    "Ngororero": (-1.8833, 29.5833),
    "Rusizi": (-2.4833, 29.0),
    "Nyamasheke": (-2.35, 29.1333),
    "Rulindo": (-1.7167, 29.85),
    "Gakenke": (-1.6833, 29.7833),
    "Musanze": (-1.4833, 29.6333),
    "Burera": (-1.4333, 29.85),
    "Gicumbi": (-1.5833, 30.0833),
    "Rwamagana": (-1.95, 30.4333),
    "Nyagatare": (-1.3167, 30.3167),
    "Gatsibo": (-1.5833, 30.4167),
    "Kayonza": (-1.8667, 30.65),
    "Kirehe": (-2.2667, 30.6667),
    "Ngoma": (-2.15, 30.4333),
    "Bugesera": (-2.1833, 30.25),
}

DISTRICT_BOUNDARIES: dict[str, list[tuple[float, float]]] = {
    "Nyarugenge": [(-1.92, 29.83), (-1.92, 29.92), (-1.97, 29.92), (-1.97, 29.86), (-1.96, 29.83), (-1.92, 29.83)],
    "Gasabo": [(-1.82, 30.0), (-1.82, 30.15), (-1.92, 30.15), (-1.92, 30.05), (-1.91, 30.0), (-1.82, 30.0)],
    "Kicukiro": [(-1.97, 30.02), (-1.97, 30.14), (-2.05, 30.14), (-2.05, 30.05), (-2.01, 30.02), (-1.97, 30.02)],
    "Nyanza": [(-2.28, 29.68), (-2.28, 29.8), (-2.42, 29.8), (-2.42, 29.68), (-2.28, 29.68)],
    "Gisagara": [(-2.5, 29.62), (-2.5, 29.78), (-2.66, 29.78), (-2.66, 29.62), (-2.5, 29.62)],
    "Nyaruguru": [(-2.48, 29.42), (-2.48, 29.6), (-2.66, 29.6), (-2.66, 29.42), (-2.48, 29.42)],
    "Huye": [(-2.5, 29.65), (-2.5, 29.82), (-2.68, 29.82), (-2.68, 29.65), (-2.5, 29.65)],
    "Nyamagabe": [(-2.33, 29.38), (-2.33, 29.58), (-2.53, 29.58), (-2.53, 29.38), (-2.33, 29.38)],
    "Ruhango": [(-2.13, 29.7), (-2.13, 29.85), (-2.3, 29.85), (-2.3, 29.7), (-2.13, 29.7)],
    "Muhanga": [(-2.0, 29.67), (-2.0, 29.82), (-2.17, 29.82), (-2.17, 29.67), (-2.0, 29.67)],
    "Kamonyi": [(-1.92, 29.75), (-1.92, 29.9), (-2.08, 29.9), (-2.08, 29.75), (-1.92, 29.75)],
    "Karongi": [(-2.05, 29.27), (-2.05, 29.47), (-2.22, 29.47), (-2.22, 29.27), (-2.05, 29.27)],
    "Rutsiro": [(-1.98, 29.15), (-1.98, 29.35), (-2.18, 29.35), (-2.18, 29.15), (-1.98, 29.15)],
    "Rubavu": [(-1.6, 29.22), (-1.6, 29.4), (-1.77, 29.4), (-1.77, 29.22), (-1.6, 29.22)],
    "Nyabihu": [(-1.63, 29.4), (-1.63, 29.6), (-1.83, 29.6), (-1.83, 29.4), (-1.63, 29.4)],
    "Ngororero": [(-1.8, 29.48), (-1.8, 29.68), (-1.97, 29.68), (-1.97, 29.48), (-1.8, 29.48)],
    "Rusizi": [(-2.38, 28.88), (-2.38, 29.1), (-2.58, 29.1), (-2.58, 28.88), (-2.38, 28.88)],
    "Nyamasheke": [(-2.25, 29.02), (-2.25, 29.22), (-2.45, 29.22), (-2.45, 29.02), (-2.25, 29.02)],
    "Rulindo": [(-1.63, 29.77), (-1.63, 29.93), (-1.8, 29.93), (-1.8, 29.77), (-1.63, 29.77)],
    "Gakenke": [(-1.58, 29.68), (-1.58, 29.88), (-1.78, 29.88), (-1.78, 29.68), (-1.58, 29.68)],
    "Musanze": [(-1.38, 29.53), (-1.38, 29.73), (-1.58, 29.73), (-1.58, 29.53), (-1.38, 29.53)],
    "Burera": [(-1.33, 29.73), (-1.33, 29.95), (-1.53, 29.95), (-1.53, 29.73), (-1.33, 29.73)],
    "Gicumbi": [(-1.47, 29.97), (-1.47, 30.18), (-1.68, 30.18), (-1.68, 29.97), (-1.47, 29.97)],
    "Rwamagana": [(-1.85, 30.33), (-1.85, 30.55), (-2.05, 30.55), (-2.05, 30.33), (-1.85, 30.33)],
    "Nyagatare": [(-1.17, 30.17), (-1.17, 30.47), (-1.43, 30.47), (-1.43, 30.17), (-1.17, 30.17)],
    "Gatsibo": [(-1.47, 30.28), (-1.47, 30.55), (-1.7, 30.55), (-1.7, 30.28), (-1.47, 30.28)],
    "Kayonza": [(-1.75, 30.5), (-1.75, 30.8), (-2.0, 30.8), (-2.0, 30.5), (-1.75, 30.5)],
    "Kirehe": [(-2.13, 30.5), (-2.13, 30.82), (-2.4, 30.82), (-2.4, 30.5), (-2.13, 30.5)],
    "Ngoma": [(-2.03, 30.3), (-2.03, 30.55), (-2.27, 30.55), (-2.27, 30.3), (-2.03, 30.3)],
    "Bugesera": [(-2.07, 30.1), (-2.07, 30.38), (-2.3, 30.38), (-2.3, 30.1), (-2.07, 30.1)],
}


def dataset_exploration(df: pd.DataFrame) -> str:
    return df.head().to_html(
        classes="table table-bordered table-striped table-sm",
        float_format="%.2f",
        justify="center",
        index=False,
    )


def data_exploration(df: pd.DataFrame) -> str:
    # Show descriptive stats to match the lab intent ("Data Description").
    return df.describe(include="all").to_html(
        classes="table table-bordered table-striped table-sm",
        float_format="%.2f",
        justify="center",
    )


def _flatten_geometry_points(geometry: dict) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    geometry_type = geometry.get("type")
    coordinates = geometry.get("coordinates", [])

    if geometry_type == "Polygon":
        for ring in coordinates:
            for lon, lat in ring:
                points.append((lon, lat))
    elif geometry_type == "MultiPolygon":
        for polygon in coordinates:
            for ring in polygon:
                for lon, lat in ring:
                    points.append((lon, lat))

    return points


def _geometry_label_point(geometry: dict) -> tuple[float, float] | None:
    points = _flatten_geometry_points(geometry)
    if not points:
        return None

    lons = [point[0] for point in points]
    lats = [point[1] for point in points]
    return (sum(lons) / len(lons), sum(lats) / len(lats))


def rwanda_clients_map(df: pd.DataFrame) -> str:
    """
    Display Rwanda map with district boundaries and number of vehicle clients
    in each district.

    Uses GeoJSON if available, otherwise falls back to approximate district
    boundaries and district center markers.
    """
    district_counts = (
        df.assign(district=df["district"].astype(str).str.strip())
        .groupby("district")
        .size()
        .reset_index(name="clients")
        .rename(columns={"district": "district_name"})
    )
    district_counts["district_norm"] = district_counts["district_name"].str.strip().str.lower()

    if GEOJSON_PATH.exists():
        with GEOJSON_PATH.open("r", encoding="utf-8") as f:
            geojson = json.load(f)

        features = geojson.get("features", [])
        if not features:
            return (
                "<div class='alert alert-warning mb-0'>"
                "GeoJSON has no features. Check <code>dummy-data/rwanda_districts.geojson</code>."
                "</div>"
            )

        sample_props = features[0].get("properties", {})
        district_property = next(
            (
                key
                for key in ["district", "shapeName", "ADM2_EN", "NAME_2", "name"]
                if key in sample_props
            ),
            None,
        )
        if district_property is None:
            return (
                "<div class='alert alert-warning mb-0'>"
                "No district name field found in GeoJSON properties."
                "</div>"
            )

        for feature in features:
            props = feature.setdefault("properties", {})
            props["district_norm"] = str(props.get(district_property, "")).strip().lower()
            props["id"] = props["district_norm"]

        centroid_rows: list[dict[str, object]] = []
        for feature in features:
            props = feature.get("properties", {})
            district_norm = props.get("district_norm", "")
            label_point = _geometry_label_point(feature.get("geometry", {}))
            if label_point is None:
                continue
            centroid_rows.append(
                {
                    "district_norm": district_norm,
                    "lat": label_point[1],
                    "lon": label_point[0],
                }
            )

        centroid_df = pd.DataFrame(centroid_rows)
        label_df = centroid_df.merge(
            district_counts[["district_name", "district_norm", "clients"]],
            on="district_norm",
            how="left",
        )
        label_df["clients"] = label_df["clients"].fillna(0).astype(int)
        label_df["district_name"] = label_df["district_name"].fillna(
            label_df["district_norm"].str.title()
        )
        label_df["text"] = (
            label_df["district_name"] + "<br>Sales: " + label_df["clients"].astype(str)
        )

        fig = px.choropleth_mapbox(
            district_counts,
            geojson=geojson,
            locations="district_norm",
            featureidkey="properties.id",
            color="clients",
            color_continuous_scale="Blues",
            mapbox_style="carto-positron",
            center={"lat": -1.94, "lon": 30.06},
            zoom=7.8,
            opacity=0.65,
            title="Rwanda Vehicle Clients Distribution by District",
            labels={"clients": "Total Clients"},
            hover_name="district_name",
        )
        fig.add_trace(
            go.Scattermapbox(
                lat=label_df["lat"],
                lon=label_df["lon"],
                mode="text",
                text=label_df["text"],
                textfont={"size": 10, "color": "#111827"},
                hoverinfo="skip",
                showlegend=False,
            )
        )
        fig.update_layout(
            margin={"r": 0, "t": 60, "l": 0, "b": 0},
            height=800,
            dragmode="zoom",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            title={"x": 0.5, "xanchor": "center", "font": {"size": 18, "color": "#2C3E50"}},
        )
        fig.update_mapboxes(center={"lat": -1.94, "lon": 30.06}, zoom=7.8)
        fig.update_traces(
            marker_line_width=1,
            marker_line_color="darkblue",
            selector={"type": "choroplethmapbox"},
        )
        return fig.to_html(
            full_html=False,
            include_plotlyjs=True,
            config={"scrollZoom": True, "responsive": True},
        )

    counts_by_district = district_counts.set_index("district_name")["clients"].to_dict()
    districts = [d for d in DISTRICT_COORDS if d in counts_by_district]
    if not districts:
        return (
            "<div class='alert alert-warning mb-0'>"
            "No valid district names found in dataset for Rwanda map."
            "</div>"
        )

    lats = [DISTRICT_COORDS[d][0] for d in districts]
    lons = [DISTRICT_COORDS[d][1] for d in districts]
    clients = [int(counts_by_district[d]) for d in districts]

    fig = go.Figure()
    for district, coords in DISTRICT_BOUNDARIES.items():
        if district not in counts_by_district:
            continue
        fig.add_trace(
            go.Scattermapbox(
                lat=[c[0] for c in coords],
                lon=[c[1] for c in coords],
                mode="lines",
                line={"width": 1, "color": "rgba(0, 90, 170, 0.45)"},
                fill="toself",
                fillcolor="rgba(0, 123, 255, 0.08)",
                hoverinfo="skip",
                showlegend=False,
            )
        )

    fig.add_trace(
        go.Scattermapbox(
            lat=lats,
            lon=lons,
            mode="markers",
            marker={
                "size": [max(12, c * 0.6) for c in clients],
                "color": clients,
                "colorscale": "Blues",
                "showscale": True,
                "colorbar": {"title": "Clients"},
            },
            text=[f"<b>{d}</b><br>Clients: {c}" for d, c in zip(districts, clients)],
            hovertemplate="%{text}<extra></extra>",
            showlegend=False,
        )
    )
    fig.update_layout(
        title="Vehicle Clients per District (Approximate Boundaries)",
        margin={"r": 0, "t": 40, "l": 0, "b": 0},
        height=620,
        mapbox={
            "style": "open-street-map",
            "center": {"lat": -1.95, "lon": 29.9},
            "zoom": 7,
        },
    )
    return fig.to_html(full_html=False, include_plotlyjs=True)
