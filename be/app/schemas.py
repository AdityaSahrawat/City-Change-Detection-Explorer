"""
Pydantic schemas for the Change Detection API.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field


class AnalysisRequest(BaseModel):
    date1_safe_path: str  # relative or absolute path to date-1 .SAFE folder
    date2_safe_path: str  # relative or absolute path to date-2 .SAFE folder
    aoi_geojson_path: str = "data/hisar.geojson"


class ClassStats(BaseModel):
    area_ha: float
    area_km2: float
    pixel_count: int


class Statistics(BaseModel):
    total_changed_area_ha: float
    total_changed_area_km2: float
    total_changed_pixels: int
    changes_by_class: dict[str, ClassStats]
    date1: str
    date2: str


class AnalysisSummary(BaseModel):
    id: str
    created_at: datetime
    date1_safe_path: str
    date2_safe_path: str
    aoi_geojson_path: str
    status: str  # "pending" | "complete" | "error"
    error: Optional[str] = None


class AnalysisDetail(AnalysisSummary):
    statistics: Optional[Statistics] = None
    date1_image_url: Optional[str] = None
    date2_image_url: Optional[str] = None
    image_coords: Optional[list[list[float]]] = None


class GeoJSONGeometry(BaseModel):
    """GeoJSON geometry object."""

    geometry_type: str = Field(..., alias="type")
    coordinates: Any

    model_config = {"populate_by_name": True}


class ChangeProperties(BaseModel):
    """Properties attached to each change polygon feature."""

    region_id: int
    change_class: str  # "water" | "vegetation" | "built_up" | "soil"
    change_direction: str  # "increase" | "decrease"
    area_ha: float
    area_km2: float
    date1: str
    date2: str
    ndvi_diff: Optional[float] = None
    ndwi_diff: Optional[float] = None
    ndbi_diff: Optional[float] = None


class ChangeFeature(BaseModel):
    """GeoJSON Feature wrapping a change polygon."""

    feature_type: str = Field(default="Feature", alias="type")
    geometry: GeoJSONGeometry
    properties: ChangeProperties

    model_config = {"populate_by_name": True}


class ChangesResponse(BaseModel):
    """GeoJSON FeatureCollection of all detected change polygons."""

    collection_type: str = Field(default="FeatureCollection", alias="type")
    features: list[ChangeFeature]
    analysis_id: str
    total_features: int

    model_config = {"populate_by_name": True}
