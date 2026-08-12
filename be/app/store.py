"""
In-memory analysis store.

Stores AnalysisRecord objects keyed by UUID.
This can be swapped for a PostGIS-backed implementation by replacing
`AnalysisStore` with a class that uses SQLAlchemy + GeoAlchemy2.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional


@dataclass
class AnalysisRecord:
    id: str
    created_at: datetime
    date1_safe_path: str
    date2_safe_path: str
    aoi_geojson_path: str
    status: str = "pending"  # "pending" | "complete" | "error"
    error: Optional[str] = None
    statistics: Optional[dict[str, Any]] = None
    changes_geojson: Optional[dict[str, Any]] = None  # GeoJSON FeatureCollection
    date1_image_url: Optional[str] = None
    date2_image_url: Optional[str] = None
    image_coords: Optional[list[list[float]]] = None


class AnalysisStore:
    """Thread-safe in-memory store for analysis results."""

    def __init__(self) -> None:
        self._data: dict[str, AnalysisRecord] = {}
        self._lock = threading.Lock()

    def add(self, record: AnalysisRecord) -> None:
        with self._lock:
            self._data[record.id] = record

    def get(self, analysis_id: str) -> Optional[AnalysisRecord]:
        with self._lock:
            return self._data.get(analysis_id)

    def list_all(self) -> list[AnalysisRecord]:
        with self._lock:
            return sorted(self._data.values(), key=lambda r: r.created_at, reverse=True)

    def update(self, record: AnalysisRecord) -> None:
        with self._lock:
            self._data[record.id] = record


# Singleton store shared across the application
store = AnalysisStore()
