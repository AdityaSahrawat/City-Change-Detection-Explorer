# 🛰️ City Change Detection Explorer

A geospatial tool for detecting and quantifying land-cover changes in urban areas using multispectral Sentinel-2 imagery — currently focused on **Hisar, Haryana, India**.

The system computes spectral indices (NDVI, NDWI, NDBI) for two dates, differences them, detects changed regions, and quantifies the change area in **hectares and km²**.

---

## 📁 Project Structure

```
City-Change-Detection-Explorer/
├── be/                          # Python backend — FastAPI + GIS processing
│   ├── app/
│   │   ├── api.py               # FastAPI application — 5 REST endpoints
│   │   ├── processing.py        # Sentinel-2 band extraction, index computation,
│   │   │                        # temporal change detection, polygon vectorisation
│   │   ├── schemas.py           # Pydantic request / response models
│   │   ├── store.py             # In-memory analysis store (PostGIS-ready abstraction)
│   │   └── load.py              # GeoJSON boundary validation helper
│   ├── data/
│   │   ├── hisar.geojson        # City boundary polygon for Hisar
│   │   ├── raw/                 # Raw Sentinel-2 .SAFE archives (not tracked in git)
│   │   └── extracted/           # Processed output rasters
│   ├── main.py                  # uvicorn entry point
│   └── pyproject.toml           # uv-managed dependencies
│
└── fe/                          # Next.js 16 frontend — interactive explorer
    ├── app/
    │   ├── components/
    │   │   ├── ChangeMap.tsx    # MapLibre GL map — change polygon overlay
    │   │   ├── StatsCards.tsx   # Area statistics cards (ha / km² per class)
    │   │   ├── AnalysisList.tsx # Sidebar list of past analyses
    │   │   └── RegionDetail.tsx # Click-to-inspect polygon detail panel
    │   ├── layout.tsx
    │   ├── page.tsx             # Main explorer page — wires all components
    │   └── globals.css          # Dark-mode design system
    └── package.json
```

---

## 🔬 How It Works

### Temporal Change Detection Pipeline

```
Date 1 Sentinel-2 .SAFE
        ↓
  classify_scene()
  B03 (Green) · B04 (Red) · B08 (NIR) · B11 (SWIR)
        ↓
  NDVI = (NIR - Red) / (NIR + Red)
  NDWI = (Green - NIR) / (Green + NIR)
  NDBI = (SWIR - NIR) / (SWIR + NIR)

Date 2 Sentinel-2 .SAFE
        ↓
  (same pipeline)

        ↓
  compute_change()
  Δ NDVI = NDVI₂ - NDVI₁
  Δ NDWI = NDWI₂ - NDWI₁
  Δ NDBI = NDBI₂ - NDBI₁
        ↓
  Threshold → binary raster mask per class
  scipy.ndimage.binary_opening (noise removal)
        ↓
  rasterio.features.shapes() → GeoJSON polygons
        ↓
  Area calculation (ha, km²) per polygon
        ↓
  GeoJSON FeatureCollection → FastAPI → Next.js + MapLibre GL
```

### Spectral Indices

| Index | Formula | Detects |
|-------|---------|---------|
| **NDVI** | `(NIR − Red) / (NIR + Red)` | Vegetation presence/loss |
| **NDWI** | `(Green − NIR) / (Green + NIR)` | Water body change |
| **NDBI** | `(SWIR − NIR) / (SWIR + NIR)` | Built-up / urban expansion |

**Bands used:**
- `B03` (Green, 10 m) — NDWI
- `B04` (Red, 10 m) — NDVI
- `B08` (NIR, 10 m) — NDVI, NDWI, NDBI
- `B11` (SWIR, 20 m) — upsampled to 10 m for NDBI

### Spatial Quantification

Every detected change region is vectorised and quantified:

```
Built-up increase · 2019 → 2025
  Changed area: 12.4 km²
  Pixel count:  124,000
```

Clicking any polygon on the map shows:

```
Region #17
  Change type:  Built-up increase
  Area:         3.42 ha  (0.034 km²)
  Before:       S2B_MSIL2A_20190301…
  After:        S2B_MSIL2A_20250302…
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.11+ with **[uv](https://docs.astral.sh/uv/)** installed
- Node.js 18+ with **pnpm**
- A Sentinel-2 L2A `.SAFE` archive ([Copernicus Browser](https://browser.dataspace.copernicus.eu/))

---

### Backend Setup

```bash
cd be

# Install all dependencies (managed by uv)
uv sync

# Start the FastAPI server
uv run uvicorn app.api:app --reload --port 8000
```

Open [http://localhost:8000/docs](http://localhost:8000/docs) for the interactive API explorer.

**Download Sentinel-2 data** and place the `.SAFE` folder under `be/data/raw/`.

> **Tip:** Set `date1_safe_path` to `"synthetic"` in the API request to auto-generate a synthetic
> baseline from the date-2 scene — useful for testing the full pipeline without two real scenes.

---

### Frontend Setup

```bash
cd fe

# Install dependencies
pnpm install

# Start dev server
pnpm dev
```

Open [http://localhost:3000](http://localhost:3000).

---

## 🌐 API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/analyses` | Trigger a new analysis (returns 202 immediately) |
| `GET` | `/analyses` | List all analyses |
| `GET` | `/analyses/{id}` | Get analysis metadata + status |
| `GET` | `/analyses/{id}/changes` | GeoJSON FeatureCollection of change polygons |
| `GET` | `/analyses/{id}/statistics` | Quantified statistics (ha, km² per class) |

**Example — start an analysis:**
```bash
curl -X POST http://localhost:8000/analyses \
  -H "Content-Type: application/json" \
  -d '{
    "date1_safe_path": "synthetic",
    "date2_safe_path": "data/raw/S2B_MSIL2A_20260302T053639_N0512_R005_T43REN_20260302T093017.SAFE",
    "aoi_geojson_path": "data/hisar.geojson"
  }'
```

**Example — get statistics:**
```bash
curl http://localhost:8000/analyses/{id}/statistics
# {
#   "total_changed_area_ha": 48.2,
#   "total_changed_area_km2": 0.482,
#   "changes_by_class": {
#     "built_up_increase": { "area_ha": 18.7, "area_km2": 0.187, "pixel_count": 18700 },
#     "vegetation_decrease": { "area_ha": 22.1, ... }
#   },
#   "date1": "synthetic-baseline",
#   "date2": "S2B_MSIL2A_20260302…"
# }
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Satellite data | Sentinel-2 L2A (ESA Copernicus) |
| Geospatial processing | `rasterio`, `geopandas`, `shapely` |
| Change detection | `numpy`, `scipy.ndimage` |
| Image processing | `scikit-image` |
| REST API | **FastAPI** + **uvicorn** |
| Frontend framework | Next.js 16 (React 19) |
| Map rendering | **MapLibre GL JS** |
| Styling | Vanilla CSS (dark-mode design system) |
| Package management | **uv** (Python), **pnpm** (Node) |
| Language | Python 3.11+, TypeScript |

---

## 🗺️ Area of Interest — Hisar, Haryana

The city boundary is defined in `be/data/hisar.geojson`:

- **Latitude:** 28.97° N – 29.46° N
- **Longitude:** 75.43° E – 76.12° E

To analyse a different city, replace `hisar.geojson` with a new boundary polygon.

---

## 🔮 Roadmap

- [ ] PostGIS persistence — swap `store.py` for SQLAlchemy + GeoAlchemy2
- [ ] Support for additional cities via dynamic AOI selection
- [ ] Export results as GeoTIFF / COG for GIS tools
- [ ] Automated Sentinel-2 download via Copernicus CDSE API
- [ ] Time-series chart: area per class over multiple dates

---

## 📄 License

Educational and research purposes. Sentinel-2 data is provided freely by the
[European Space Agency (ESA)](https://www.esa.int/) under the Copernicus Open Access policy.
