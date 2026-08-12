"""
FastAPI application — City Change Detection Explorer API.

Endpoints:
  POST   /analyses                       Trigger a new analysis
  GET    /analyses                       List all analyses
  GET    /analyses/{id}                  Get analysis metadata
  GET    /analyses/{id}/changes          GeoJSON FeatureCollection of change polygons
  GET    /analyses/{id}/statistics       Quantified change statistics
"""
from __future__ import annotations

import logging
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any

import json
import os
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .processing import run_analysis
from .schemas import (
    AnalysisDetail,
    AnalysisRequest,
    AnalysisSummary,
    ChangesResponse,
    Statistics,
)
from .store import AnalysisRecord, store

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="City Change Detection Explorer API",
    description=(
        "Geospatial change detection API using Sentinel-2 multispectral imagery. "
        "Computes NDVI / NDWI / NDBI differences between two dates, vectorises "
        "change regions, and returns spatial quantification in hectares / km²."
    ),
    version="0.1.0",
)

# ── CORS — allow Next.js dev server ───────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs("data/extracted", exist_ok=True)
app.mount("/extracted", StaticFiles(directory="data/extracted"), name="extracted")

_executor = ThreadPoolExecutor(max_workers=2)


# ── Background worker ──────────────────────────────────────────────────────────

def _run_and_store(record: AnalysisRecord) -> None:
    """Run analysis in background thread and update the store."""
    try:
        result = run_analysis(
            record.date1_safe_path,
            record.date2_safe_path,
            record.aoi_geojson_path,
        )
        record.statistics = result["statistics"]
        record.changes_geojson = result["changes_geojson"]
        record.date1_image_url = result.get("date1_image_url")
        record.date2_image_url = result.get("date2_image_url")
        record.image_coords = result.get("image_coords")
        record.status = "complete"
        logger.info("Analysis %s complete — %d features", record.id, len(result["changes_geojson"]["features"]))
    except Exception as exc:
        record.status = "error"
        record.error = str(exc)
        logger.exception("Analysis %s failed: %s", record.id, exc)
    finally:
        store.update(record)


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/aoi")
def get_aoi() -> dict[str, Any]:
    """Get the city boundary GeoJSON (hisar.geojson) to highlight AOI."""
    aoi_path = "data/hisar.geojson"
    if not os.path.exists(aoi_path):
        raise HTTPException(status_code=404, detail="AOI GeoJSON file not found")
    with open(aoi_path, "r", encoding="utf-8") as f:
        return json.load(f)


@app.get("/aoi-mask")
def get_aoi_mask() -> dict[str, Any]:
    """Get the inverted mask GeoJSON for hisar.geojson to mask area outside AOI."""
    mask_path = "data/hisar_mask.geojson"
    if not os.path.exists(mask_path):
        raise HTTPException(status_code=404, detail="AOI mask GeoJSON file not found")
    with open(mask_path, "r", encoding="utf-8") as f:
        return json.load(f)


@app.post("/analyses", response_model=AnalysisSummary, status_code=202)
def create_analysis(
    request: AnalysisRequest,
    background_tasks: BackgroundTasks,
) -> AnalysisSummary:
    analysis_id = str(uuid.uuid4())
    record = AnalysisRecord(
        id=analysis_id,
        created_at=datetime.now(tz=timezone.utc),
        date1_safe_path=request.date1_safe_path,
        date2_safe_path=request.date2_safe_path,
        aoi_geojson_path=request.aoi_geojson_path,
        status="pending",
    )
    store.add(record)
    background_tasks.add_task(_run_and_store, record)
    return _record_to_summary(record)


@app.get("/analyses", response_model=list[AnalysisSummary])
def list_analyses() -> list[AnalysisSummary]:
    """List all analyses (most recent first)."""
    return [_record_to_summary(r) for r in store.list_all()]


@app.get("/analyses/{analysis_id}", response_model=AnalysisDetail)
def get_analysis(analysis_id: str) -> AnalysisDetail:
    """Get metadata and statistics for a specific analysis."""
    record = _get_or_404(analysis_id)
    detail = AnalysisDetail(
        **_record_to_summary(record).model_dump(),
        statistics=_stats_from_record(record),
        date1_image_url=record.date1_image_url,
        date2_image_url=record.date2_image_url,
        image_coords=record.image_coords,
    )
    return detail


@app.get("/analyses/{analysis_id}/changes", response_model=ChangesResponse, response_model_by_alias=True)
def get_changes(analysis_id: str) -> ChangesResponse:
    """
    Get GeoJSON FeatureCollection of all change polygons.

    Each feature has properties:
      region_id, change_class, change_direction,
      area_ha, area_km², date1, date2
    """
    record = _get_or_404(analysis_id)
    _assert_complete(record)

    geojson = record.changes_geojson or {"type": "FeatureCollection", "features": []}
    return ChangesResponse.model_validate({
        "type": "FeatureCollection",
        "features": geojson.get("features", []),
        "analysis_id": analysis_id,
        "total_features": len(geojson.get("features", [])),
    })


@app.get("/analyses/{analysis_id}/statistics", response_model=Statistics)
def get_statistics(analysis_id: str) -> Statistics:
    """
    Get spatial quantification for an analysis.

    Returns total changed area (ha, km²) and per-class breakdown.
    """
    record = _get_or_404(analysis_id)
    _assert_complete(record)
    stats = _stats_from_record(record)
    if stats is None:
        raise HTTPException(status_code=500, detail="Statistics not computed")
    return stats


# ── Helpers ────────────────────────────────────────────────────────────────────

def _get_or_404(analysis_id: str) -> AnalysisRecord:
    record = store.get(analysis_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Analysis '{analysis_id}' not found")
    return record


def _assert_complete(record: AnalysisRecord) -> None:
    if record.status == "pending":
        raise HTTPException(status_code=202, detail="Analysis is still running")
    if record.status == "error":
        raise HTTPException(status_code=500, detail=f"Analysis failed: {record.error}")


def _record_to_summary(record: AnalysisRecord) -> AnalysisSummary:
    return AnalysisSummary(
        id=record.id,
        created_at=record.created_at,
        date1_safe_path=record.date1_safe_path,
        date2_safe_path=record.date2_safe_path,
        aoi_geojson_path=record.aoi_geojson_path,
        status=record.status,
        error=record.error,
    )


def _stats_from_record(record: AnalysisRecord) -> Statistics | None:
    if record.statistics is None:
        return None
    s = record.statistics
    return Statistics(
        total_changed_area_ha=s["total_changed_area_ha"],
        total_changed_area_km2=s["total_changed_area_km2"],
        total_changed_pixels=s["total_changed_pixels"],
        changes_by_class=s["changes_by_class"],
        date1=s["date1"],
        date2=s["date2"],
    )
