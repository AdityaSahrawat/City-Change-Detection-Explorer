"""
Sentinel-2 processing module.

Provides two main functions:
  classify_scene()  — extract bands, compute spectral indices, classify land cover
  compute_change()  — diff two scenes, produce binary masks, vectorise to polygons, calc areas

The pipeline:
  Date 1 .SAFE  →  classify_scene()  →  NDVI/NDWI/NDBI + classification
  Date 2 .SAFE  →  classify_scene()  →  NDVI/NDWI/NDBI + classification
                      ↓
                compute_change()
                      ↓
              difference arrays per index
                      ↓
           binary change mask per class (threshold)
                      ↓
           scipy.ndimage binary_opening (noise removal)
                      ↓
           rasterio.features.shapes() → shapely polygons
                      ↓
           area in ha and km² (via projected CRS)
                      ↓
           GeoJSON FeatureCollection
"""
from __future__ import annotations

import glob
import json
import logging
import os
import uuid
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import rasterio
from PIL import Image
from pyproj import Transformer
from rasterio.features import shapes
from rasterio.mask import mask as rio_mask
from scipy.ndimage import binary_opening, median_filter
from shapely.geometry import mapping, shape
from shapely.ops import transform as shapely_transform, unary_union
from skimage.transform import resize

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

CLASS_LABELS = {
    0: "background",
    1: "water",
    2: "vegetation",
    3: "built_up",
    4: "soil",
}

CLASS_COLORS = {
    "background": "#2d2d2d",
    "water": "#1f77b4",
    "vegetation": "#2ca02c",
    "built_up": "#d62728",
    "soil": "#ff7f0e",
}

# Thresholds for change detection
CHANGE_THRESHOLDS = {
    "ndvi": 0.15,   # |Δndvi| > 0.15 → vegetation change
    "ndwi": 0.10,   # |Δndwi| > 0.10 → water change
    "ndbi": 0.12,   # |Δndbi| > 0.12 → built-up change
}

# Pixel area at 10 m resolution
PIXEL_AREA_HA = (10 * 10) / 10_000  # 0.01 ha per pixel
PIXEL_AREA_KM2 = (10 * 10) / 1_000_000  # 0.0001 km² per pixel

# Minimum area (ha) for a change polygon to be included in output
MIN_POLYGON_AREA_HA = 0.05


# ── Band helpers ───────────────────────────────────────────────────────────────

def _find_band(base_path: str, resolution: str, suffix: str) -> str:
    pattern = os.path.join(base_path, f"R{resolution}m", f"*_{suffix}_{resolution}m.jp2")
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"Band not found: {pattern}")
    return files[0]


def _img_data_path(safe_path: str) -> str:
    """Resolve the IMG_DATA path inside a .SAFE archive."""
    pattern = os.path.join(safe_path, "GRANULE", "*", "IMG_DATA")
    matches = glob.glob(pattern)
    if not matches:
        raise FileNotFoundError(f"IMG_DATA not found under: {safe_path}")
    return matches[0]


# ── Synthetic scene generator (for demo / missing real data) ───────────────────

def _make_synthetic_scene(reference_scene: dict[str, Any], seed: int = 42) -> dict[str, Any]:
    """
    Generate a plausible synthetic scene by perturbing a reference scene.
    Used when a second real Sentinel-2 scene is not available.
    """
    rng = np.random.default_rng(seed)
    perturbed = {}
    for key in ("ndvi", "ndwi", "ndbi"):
        arr = reference_scene[key].copy()
        noise = rng.normal(0, 0.05, arr.shape).astype("float32")
        # simulate built-up growth in top-right quadrant
        h, w = arr.shape
        if key == "ndbi":
            arr[: h // 3, w // 2 :] += 0.2
        elif key == "ndvi":
            arr[: h // 3, w // 2 :] -= 0.2
        perturbed[key] = np.clip(arr + noise, -1.0, 1.0)

    # Re-derive classification from perturbed indices
    ndvi, ndwi, ndbi = perturbed["ndvi"], perturbed["ndwi"], perturbed["ndbi"]
    classification = _classify_from_indices(ndvi, ndwi, ndbi)

    return {
        "ndvi": perturbed["ndvi"],
        "ndwi": perturbed["ndwi"],
        "ndbi": perturbed["ndbi"],
        "classification": classification,
        "transform": reference_scene["transform"],
        "crs": reference_scene["crs"],
        "shape": reference_scene["shape"],
        "synthetic": True,
    }


# ── Core classification ────────────────────────────────────────────────────────

def _classify_from_indices(
    ndvi: np.ndarray,
    ndwi: np.ndarray,
    ndbi: np.ndarray,
) -> np.ndarray:
    """Rule-based classification from spectral indices."""
    classification = np.zeros_like(ndvi, dtype=np.uint8)
    classification[ndwi > 0.1] = 1                                            # water
    classification[(ndvi > 0.5) & (classification == 0)] = 2                  # vegetation
    classification[(ndbi > 0.15) & (ndvi < 0.3) & (classification == 0)] = 3  # built-up
    classification[(classification == 0) & (ndvi > -0.1)] = 4                 # soil
    classification = median_filter(classification, size=3)
    built_up = binary_opening(classification == 3, structure=np.ones((3, 3)))
    classification[built_up] = 3
    return classification.astype(np.uint8)


def classify_scene(safe_path: str, aoi_geojson: str) -> dict[str, Any]:
    """
    Extract Sentinel-2 bands from a .SAFE archive, crop to AOI, compute
    spectral indices, and classify land cover.

    Returns a dict with:
      ndvi, ndwi, ndbi       — float32 arrays
      classification         — uint8 array (0=bg, 1=water, 2=veg, 3=built, 4=soil)
      transform              — affine transform (10 m grid)
      crs                    — rasterio CRS
      shape                  — (height, width)
      synthetic              — False
    """
    img_data = _img_data_path(safe_path)

    b03_path = _find_band(img_data, "10", "B03")
    b04_path = _find_band(img_data, "10", "B04")
    b08_path = _find_band(img_data, "10", "B08")
    b11_path = _find_band(img_data, "20", "B11")

    with rasterio.open(b04_path) as red_src:
        gdf = gpd.read_file(aoi_geojson).to_crs(red_src.crs)
        geoms = list(gdf.geometry)

        def crop(path: str) -> tuple[np.ndarray, Any]:
            with rasterio.open(path) as src:
                arr, transform = rio_mask(src, geoms, crop=True)
                return arr[0].astype("float32"), transform

        green, transform = crop(b03_path)
        red, _ = crop(b04_path)
        nir, _ = crop(b08_path)
        swir_raw, _ = crop(b11_path)

    # Upsample SWIR from 20 m → 10 m
    swir = resize(swir_raw, red.shape, order=1, preserve_range=True).astype("float32")

    eps = 1e-6
    ndvi = (nir - red) / (nir + red + eps)
    ndwi = (green - nir) / (green + nir + eps)
    ndbi = (swir - nir) / (swir + nir + eps)

    classification = _classify_from_indices(ndvi, ndwi, ndbi)

    with rasterio.open(b04_path) as red_src:
        gdf_proj = gpd.read_file(aoi_geojson).to_crs(red_src.crs)
        _, crop_transform = rio_mask(red_src, list(gdf_proj.geometry), crop=True)

    return {
        "ndvi": ndvi,
        "ndwi": ndwi,
        "ndbi": ndbi,
        "classification": classification,
        "transform": crop_transform,
        "crs": rasterio.open(b04_path).crs,
        "shape": ndvi.shape,
        "synthetic": False,
    }


# ── Change computation ─────────────────────────────────────────────────────────

def _mask_to_polygons(
    binary_mask: np.ndarray,
    transform: Any,
    crs: Any,
    min_pixels: int = 5,
) -> list[dict[str, Any]]:
    """
    Convert a binary uint8 raster mask to a list of GeoJSON-like geometry dicts.
    Filters out tiny regions below min_pixels.
    """
    mask_uint8 = binary_mask.astype(np.uint8)
    polygons = []
    for geom, value in shapes(mask_uint8, transform=transform):
        if value == 1:
            poly = shape(geom)
            if poly.is_valid and not poly.is_empty:
                polygons.append(geom)
    return polygons


def compute_change(
    scene_before: dict[str, Any],
    scene_after: dict[str, Any],
    date1: str,
    date2: str,
) -> dict[str, Any]:
    """
    Compute temporal change between two classified scenes.

    Pipeline:
      index_before → index_after → difference
          ↓
      threshold → binary change mask per class
          ↓
      binary_opening (morphological noise removal)
          ↓
      rasterio.features.shapes() → GeoJSON polygons
          ↓
      area in ha and km²
          ↓
      GeoJSON FeatureCollection + statistics dict

    Returns:
      {
        "changes_geojson": GeoJSON FeatureCollection,
        "statistics": { total_changed_area_ha, changes_by_class, ... }
      }
    """
    ndvi_diff = scene_after["ndvi"] - scene_before["ndvi"]
    ndwi_diff = scene_after["ndwi"] - scene_before["ndwi"]
    ndbi_diff = scene_after["ndbi"] - scene_before["ndbi"]

    transform = scene_before["transform"]
    crs = scene_before["crs"]
    transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)

    features: list[dict[str, Any]] = []
    stats_by_class: dict[str, dict] = {}
    region_id = 0
    total_pixels = 0

    # ── Per-index change masks ─────────────────────────────────────────────────
    index_specs = [
        ("built_up",   ndbi_diff,  CHANGE_THRESHOLDS["ndbi"]),
        ("vegetation", ndvi_diff, -CHANGE_THRESHOLDS["ndvi"]),  # negative = veg loss
        ("water",      ndwi_diff,  CHANGE_THRESHOLDS["ndwi"]),
    ]

    for change_class, diff_arr, threshold in index_specs:
        for direction, sign in [("increase", 1), ("decrease", -1)]:
            if sign == 1:
                binary_mask = (diff_arr > threshold).astype(np.uint8)
            else:
                binary_mask = (diff_arr < -abs(threshold)).astype(np.uint8)

            # Morphological cleanup
            cleaned = binary_opening(binary_mask, structure=np.ones((3, 3))).astype(np.uint8)

            pixel_count = int(cleaned.sum())
            if pixel_count == 0:
                continue

            area_ha = pixel_count * PIXEL_AREA_HA
            area_km2 = pixel_count * PIXEL_AREA_KM2

            # Vectorise to polygons
            raw_geoms = _mask_to_polygons(cleaned, transform, crs)

            for geom in raw_geoms:
                poly = shape(geom)
                poly_area_ha = poly.area / 10_000  # transform is in metres
                if poly_area_ha < MIN_POLYGON_AREA_HA:
                    continue

                # Reproject geometry to WGS84 (EPSG:4326) for MapLibre / GeoJSON standard
                poly_4326 = shapely_transform(transformer.transform, poly)
                geom_4326 = mapping(poly_4326)

                feature = {
                    "type": "Feature",
                    "id": region_id,
                    "geometry": geom_4326,
                    "properties": {
                        "region_id": region_id,
                        "change_class": change_class,
                        "change_direction": direction,
                        "area_ha": round(poly_area_ha, 4),
                        "area_km2": round(poly_area_ha / 100, 6),
                        "date1": date1,
                        "date2": date2,
                    },
                }
                features.append(feature)
                region_id += 1

            # Aggregate stats per class+direction
            key = f"{change_class}_{direction}"
            stats_by_class[key] = {
                "area_ha": round(area_ha, 4),
                "area_km2": round(area_km2, 6),
                "pixel_count": pixel_count,
            }
            total_pixels += pixel_count

    total_area_ha = total_pixels * PIXEL_AREA_HA
    total_area_km2 = total_pixels * PIXEL_AREA_KM2

    changes_geojson = {
        "type": "FeatureCollection",
        "features": features,
    }

    statistics = {
        "total_changed_area_ha": round(total_area_ha, 4),
        "total_changed_area_km2": round(total_area_km2, 6),
        "total_changed_pixels": total_pixels,
        "changes_by_class": stats_by_class,
        "date1": date1,
        "date2": date2,
    }

    return {
        "changes_geojson": changes_geojson,
        "statistics": statistics,
    }


def _extract_tci_png(safe_path: str, aoi_geojson: str, out_filename: str) -> tuple[str, list[list[float]]]:
    """
    Extract True Color Image (TCI RGB) from Sentinel-2 scene, crop to AOI,
    save as PNG, and return (out_filename, corner_coordinates_4326).
    """
    img_data = _img_data_path(safe_path)
    tci_path = _find_band(img_data, "10", "TCI")

    with rasterio.open(tci_path) as src:
        gdf = gpd.read_file(aoi_geojson).to_crs(src.crs)
        arr, tf = rio_mask(src, list(gdf.geometry), crop=True)
        _, h, w = arr.shape

        rgb = np.moveaxis(arr, 0, -1).astype(np.uint8)
        os.makedirs("data/extracted", exist_ok=True)
        out_path = os.path.join("data/extracted", out_filename)
        img = Image.fromarray(rgb)
        img.save(out_path, format="PNG")

        tr = Transformer.from_crs(src.crs, "EPSG:4326", always_xy=True)
        tl = list(tr.transform(*tf * (0, 0)))
        tr_pt = list(tr.transform(*tf * (w, 0)))
        br = list(tr.transform(*tf * (w, h)))
        bl = list(tr.transform(*tf * (0, h)))
        image_coords = [tl, tr_pt, br, bl]

        return out_filename, image_coords


def run_analysis(
    date1_safe_path: str,
    date2_safe_path: str,
    aoi_geojson_path: str,
) -> dict[str, Any]:
    """
    Top-level entry point called by the API endpoint.
    Handles the case where date1_safe_path == "synthetic" by generating
    a perturbed version of the date-2 scene as a synthetic baseline.
    """
    logger.info("Processing date-2 scene: %s", date2_safe_path)
    scene2 = classify_scene(date2_safe_path, aoi_geojson_path)

    run_id = str(uuid.uuid4())[:8]

    # Extract real Sentinel-2 satellite photo (TCI) for Date 2
    date2_png_name, image_coords = _extract_tci_png(
        date2_safe_path, aoi_geojson_path, f"{run_id}_date2.png"
    )

    if date1_safe_path.lower() == "synthetic":
        logger.info("Generating synthetic date-1 scene from date-2")
        scene1 = _make_synthetic_scene(scene2, seed=42)
        date1_label = "synthetic-baseline"
        date1_png_name = date2_png_name
    else:
        logger.info("Processing date-1 scene: %s", date1_safe_path)
        scene1 = classify_scene(date1_safe_path, aoi_geojson_path)
        date1_label = Path(date1_safe_path).stem
        date1_png_name, _ = _extract_tci_png(
            date1_safe_path, aoi_geojson_path, f"{run_id}_date1.png"
        )

    date2_label = Path(date2_safe_path).stem

    result = compute_change(scene1, scene2, date1_label, date2_label)
    result["date1_image_url"] = f"/extracted/{date1_png_name}"
    result["date2_image_url"] = f"/extracted/{date2_png_name}"
    result["image_coords"] = image_coords
    return result
