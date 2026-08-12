"use client";

import { useEffect, useRef, useState } from "react";
import * as maplibregl from "maplibre-gl";
import type { GeoJSONSource, MapGeoJSONFeature } from "maplibre-gl";
import type * as GeoJSON from "geojson";

const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

interface ChangeFeatureProps {
  region_id: number;
  change_class: string;
  change_direction: string;
  area_ha: number;
  area_km2: number;
  date1: string;
  date2: string;
}

interface ChangesGeoJSON {
  type: "FeatureCollection";
  features: Array<{
    type: "Feature";
    geometry: GeoJSON.Geometry;
    properties: ChangeFeatureProps;
  }>;
}

interface ChangeMapProps {
  geojson: ChangesGeoJSON | null;
  opacity?: number; // 0–1, default 0.65
  date1ImageUrl?: string | null;
  date2ImageUrl?: string | null;
  imageCoords?: number[][] | null;
  onRegionClick?: (props: ChangeFeatureProps) => void;
}

const CLASS_COLORS: Record<string, string> = {
  built_up:   "#fc8181",
  vegetation: "#68d391",
  water:      "#63b3ed",
  soil:       "#f6ad55",
};

const LEGEND_ITEMS = [
  { key: "built_up",   label: "Built-up increase", color: CLASS_COLORS.built_up },
  { key: "vegetation", label: "Vegetation change",  color: CLASS_COLORS.vegetation },
  { key: "water",      label: "Water change",       color: CLASS_COLORS.water },
  { key: "soil",       label: "Soil / bare land",   color: CLASS_COLORS.soil },
];

type BasemapStyle = "satellite" | "streets" | "dark";

const BASEMAP_TILES: Record<BasemapStyle, { url: string[]; attribution: string }> = {
  satellite: {
    url: [
      "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    ],
    attribution: "© Esri World Imagery",
  },
  streets: {
    url: [
      "https://a.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}@2x.png",
      "https://b.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}@2x.png",
    ],
    attribution: "© OpenStreetMap contributors © CARTO",
  },
  dark: {
    url: [
      "https://a.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}@2x.png",
      "https://b.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}@2x.png",
    ],
    attribution: "© OpenStreetMap contributors © CARTO",
  },
};

const DEFAULT_CENTER: [number, number] = [75.72, 29.15]; // Hisar, Haryana
const DEFAULT_ZOOM = 11;

export default function ChangeMap({
  geojson,
  opacity = 0.65,
  date1ImageUrl,
  date2ImageUrl,
  imageCoords,
  onRegionClick,
}: ChangeMapProps) {
  const container1Ref = useRef<HTMLDivElement>(null); // Base Map (Date 1 Sentinel-2 Scene)
  const container2Ref = useRef<HTMLDivElement>(null); // Overlay Map (Date 2 Sentinel-2 Scene + Changes)
  const map1Ref       = useRef<maplibregl.Map | null>(null);
  const map2Ref       = useRef<maplibregl.Map | null>(null);
  const isSyncingRef  = useRef(false);

  const [ready, setReady]           = useState(false);
  const [swipePos, setSwipePos]     = useState(50); // 0..100% split screen
  const [isDragging, setIsDragging] = useState(false);
  const [basemap, setBasemap]       = useState<BasemapStyle>("satellite");

  // ── Helper to sync viewports between Map 1 and Map 2 ────────────────────────
  const syncMaps = (source: maplibregl.Map, target: maplibregl.Map) => {
    if (isSyncingRef.current) return;
    isSyncingRef.current = true;
    target.jumpTo({
      center: source.getCenter(),
      zoom: source.getZoom(),
      bearing: source.getBearing(),
      pitch: source.getPitch(),
    });
    isSyncingRef.current = false;
  };

  // ── Init dual maps ─────────────────────────────────────────────────────────
  useEffect(() => {
    if (!container1Ref.current || !container2Ref.current || map1Ref.current) return;

    const labelsTiles = [
      "https://a.basemaps.cartocdn.com/rastertiles/voyager_only_labels/{z}/{x}/{y}@2x.png",
      "https://b.basemaps.cartocdn.com/rastertiles/voyager_only_labels/{z}/{x}/{y}@2x.png",
    ];

    // Map 1: Date 1 Sentinel-2 Satellite Image + Labels
    const map1 = new maplibregl.Map({
      container: container1Ref.current,
      style: {
        version: 8,
        sources: {
          "basemap-source-1": {
            type: "raster",
            tiles: BASEMAP_TILES.satellite.url,
            tileSize: 256,
            attribution: BASEMAP_TILES.satellite.attribution,
          },
          "labels-source-1": {
            type: "raster",
            tiles: labelsTiles,
            tileSize: 256,
          },
        },
        layers: [
          { id: "basemap-layer-1", type: "raster", source: "basemap-source-1" },
          { id: "labels-layer-1", type: "raster", source: "labels-source-1", paint: { "raster-opacity": 0.85 } },
        ],
      },
      center: DEFAULT_CENTER,
      zoom: DEFAULT_ZOOM,
    });

    map1.addControl(new maplibregl.NavigationControl(), "bottom-left");

    // Map 2: Date 2 Sentinel-2 Satellite Image + Labels + Changes
    const map2 = new maplibregl.Map({
      container: container2Ref.current,
      style: {
        version: 8,
        sources: {
          "basemap-source-2": {
            type: "raster",
            tiles: BASEMAP_TILES.satellite.url,
            tileSize: 256,
          },
          "labels-source-2": {
            type: "raster",
            tiles: labelsTiles,
            tileSize: 256,
          },
        },
        layers: [
          { id: "basemap-layer-2", type: "raster", source: "basemap-source-2" },
          { id: "labels-layer-2", type: "raster", source: "labels-source-2", paint: { "raster-opacity": 0.85 } },
        ],
      },
      center: DEFAULT_CENTER,
      zoom: DEFAULT_ZOOM,
      interactive: true,
    });

    // Sync viewport movements
    map1.on("move", () => syncMaps(map1, map2));
    map2.on("move", () => syncMaps(map2, map1));

    map2.on("load", async () => {
      // ── Add AOI boundary & Mask (removes map area outside hisar.geojson) ──
      try {
        const [aoiRes, maskRes] = await Promise.all([
          fetch(`${API_BASE}/aoi`),
          fetch(`${API_BASE}/aoi-mask`),
        ]);

        if (aoiRes.ok) {
          const aoiGeoJSON = await aoiRes.json();
          [map1, map2].forEach((m, idx) => {
            m.addSource("hisar-aoi", {
              type: "geojson",
              data: aoiGeoJSON,
            });

            m.addLayer({
              id: `hisar-aoi-fill-${idx}`,
              type: "fill",
              source: "hisar-aoi",
              paint: {
                "fill-color": "#06b6d4",
                "fill-opacity": 0.04,
              },
            });

            m.addLayer({
              id: `hisar-aoi-outline-${idx}`,
              type: "line",
              source: "hisar-aoi",
              paint: {
                "line-color": "#00f2fe",
                "line-width": 3,
                "line-opacity": 0.95,
              },
            });
          });
        }

        if (maskRes.ok) {
          const maskGeoJSON = await maskRes.json();
          [map1, map2].forEach((m, idx) => {
            m.addSource("hisar-mask", {
              type: "geojson",
              data: maskGeoJSON,
            });

            // Mask layer blocks out everything outside hisar.geojson
            m.addLayer({
              id: `hisar-mask-fill-${idx}`,
              type: "fill",
              source: "hisar-mask",
              paint: {
                "fill-color": "#090d16",
                "fill-opacity": 0.92,
              },
            });
          });
        }
      } catch (e) {
        console.error("Failed to load AOI/Mask GeoJSON:", e);
      }

      // Add changes source to map2
      map2.addSource("changes", {
        type: "geojson",
        data: { type: "FeatureCollection", features: [] },
      });

      // Change fill layer
      map2.addLayer(
        {
          id: "changes-fill",
          type: "fill",
          source: "changes",
          paint: {
            "fill-color": [
              "match", ["get", "change_class"],
              "built_up",   CLASS_COLORS.built_up,
              "vegetation", CLASS_COLORS.vegetation,
              "water",      CLASS_COLORS.water,
              "soil",       CLASS_COLORS.soil,
              "#ff7800",
            ],
            "fill-opacity": opacity,
          },
        },
        "labels-layer-2"
      );

      // Change outline layer
      map2.addLayer(
        {
          id: "changes-outline",
          type: "line",
          source: "changes",
          paint: {
            "line-color": [
              "match", ["get", "change_class"],
              "built_up",   "#e53e3e",
              "vegetation", "#2f855a",
              "water",      "#2b6cb0",
              "soil",       "#dd6b20",
              "#dd6b20",
            ],
            "line-width": 1.8,
            "line-opacity": Math.min(1, opacity + 0.3),
          },
        },
        "labels-layer-2"
      );

      // Hover highlight layer
      map2.addLayer(
        {
          id: "changes-hover",
          type: "fill",
          source: "changes",
          paint: {
            "fill-color": "#ffffff",
            "fill-opacity": [
              "case",
              ["boolean", ["feature-state", "hovered"], false],
              0.35, 0,
            ],
          },
        },
        "labels-layer-2"
      );

      map2.on("click", "changes-fill", (e) => {
        const f = e.features?.[0] as MapGeoJSONFeature | undefined;
        if (f && onRegionClick) onRegionClick(f.properties as ChangeFeatureProps);
      });

      let hoveredId: string | number | null = null;
      map2.on("mousemove", "changes-fill", (e) => {
        map2.getCanvas().style.cursor = "pointer";
        const f = e.features?.[0];
        if (f && f.id !== undefined && f.id !== null) {
          if (hoveredId !== null && hoveredId !== f.id) {
            map2.setFeatureState({ source: "changes", id: hoveredId }, { hovered: false });
          }
          hoveredId = f.id;
          map2.setFeatureState({ source: "changes", id: hoveredId }, { hovered: true });
        }
      });

      map2.on("mouseleave", "changes-fill", () => {
        map2.getCanvas().style.cursor = "";
        if (hoveredId !== null) {
          map2.setFeatureState({ source: "changes", id: hoveredId }, { hovered: false });
          hoveredId = null;
        }
      });

      map1Ref.current = map1;
      map2Ref.current = map2;
      setReady(true);
    });

    return () => {
      map1.remove();
      map2.remove();
      map1Ref.current = null;
      map2Ref.current = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // ── Add/Update real Sentinel-2 satellite image overlays on Map 1 and Map 2 ─
  useEffect(() => {
    if (!ready || !imageCoords || imageCoords.length < 4) return;

    // Date 1 Sentinel-2 TCI image on Map 1
    if (map1Ref.current && date1ImageUrl) {
      const url1 = `${API_BASE}${date1ImageUrl}`;
      if (map1Ref.current.getSource("s2-date1")) {
        (map1Ref.current.getSource("s2-date1") as maplibregl.ImageSource).updateImage({
          url: url1,
          coordinates: imageCoords as [ [number, number], [number, number], [number, number], [number, number] ],
        });
      } else {
        map1Ref.current.addSource("s2-date1", {
          type: "image",
          url: url1,
          coordinates: imageCoords as [ [number, number], [number, number], [number, number], [number, number] ],
        });
        map1Ref.current.addLayer(
          {
            id: "s2-date1-layer",
            type: "raster",
            source: "s2-date1",
            paint: { "raster-opacity": 0.95 },
          },
          "labels-layer-1"
        );
      }
    }

    // Date 2 Sentinel-2 TCI image on Map 2
    if (map2Ref.current && date2ImageUrl) {
      const url2 = `${API_BASE}${date2ImageUrl}`;
      if (map2Ref.current.getSource("s2-date2")) {
        (map2Ref.current.getSource("s2-date2") as maplibregl.ImageSource).updateImage({
          url: url2,
          coordinates: imageCoords as [ [number, number], [number, number], [number, number], [number, number] ],
        });
      } else {
        map2Ref.current.addSource("s2-date2", {
          type: "image",
          url: url2,
          coordinates: imageCoords as [ [number, number], [number, number], [number, number], [number, number] ],
        });
        map2Ref.current.addLayer(
          {
            id: "s2-date2-layer",
            type: "raster",
            source: "s2-date2",
            paint: { "raster-opacity": 0.95 },
          },
          "changes-fill"
        );
      }
    }
  }, [ready, date1ImageUrl, date2ImageUrl, imageCoords]);

  // ── Switch basemap style for both maps ─────────────────────────────────────
  const switchBasemap = (style: BasemapStyle) => {
    setBasemap(style);
    if (!ready) return;
    [map1Ref.current, map2Ref.current].forEach((map, idx) => {
      if (!map) return;
      const srcName = idx === 0 ? "basemap-source-1" : "basemap-source-2";
      const source = map.getSource(srcName) as maplibregl.RasterTileSource | undefined;
      if (source && source.setTiles) {
        source.setTiles(BASEMAP_TILES[style].url);
      }
    });
  };

  // ── Sync GeoJSON data & fit bounds ─────────────────────────────────────────
  useEffect(() => {
    if (!ready || !map2Ref.current) return;
    const source = map2Ref.current.getSource("changes") as GeoJSONSource | undefined;
    if (!source) return;

    const data = geojson ?? { type: "FeatureCollection", features: [] };
    source.setData(data as GeoJSON.FeatureCollection);

    if (geojson?.features.length) {
      const bounds = new maplibregl.LngLatBounds();
      for (const f of geojson.features) {
        const g = f.geometry as GeoJSON.Polygon | GeoJSON.MultiPolygon;
        const flat = g.type === "Polygon" ? g.coordinates.flat() : g.coordinates.flat(2);
        for (const [lng, lat] of flat as [number, number][]) {
          if (typeof lng === "number" && typeof lat === "number" && lng >= -180 && lng <= 180 && lat >= -90 && lat <= 90) {
            bounds.extend([lng, lat]);
          }
        }
      }
      if (!bounds.isEmpty()) {
        map1Ref.current?.fitBounds(bounds, { padding: 60, maxZoom: 14, duration: 800 });
        map2Ref.current.fitBounds(bounds, { padding: 60, maxZoom: 14, duration: 800 });
      }
    }
  }, [geojson, ready]);

  // ── Sync opacity ───────────────────────────────────────────────────────────
  useEffect(() => {
    if (!ready || !map2Ref.current) return;
    map2Ref.current.setPaintProperty("changes-fill",    "fill-opacity", opacity);
    map2Ref.current.setPaintProperty("changes-outline", "line-opacity", Math.min(1, opacity + 0.3));
  }, [opacity, ready]);

  // ── Handle slider drag ──────────────────────────────────────────────────────
  const handleMouseDown = () => setIsDragging(true);

  useEffect(() => {
    if (!isDragging) return;

    const handleMouseMove = (e: MouseEvent) => {
      if (!container1Ref.current) return;
      const rect = container1Ref.current.getBoundingClientRect();
      const x = Math.max(0, Math.min(e.clientX - rect.left, rect.width));
      const pct = Math.round((x / rect.width) * 100);
      setSwipePos(pct);
    };

    const handleMouseUp = () => setIsDragging(false);

    window.addEventListener("mousemove", handleMouseMove);
    window.addEventListener("mouseup", handleMouseUp);
    return () => {
      window.removeEventListener("mousemove", handleMouseMove);
      window.removeEventListener("mouseup", handleMouseUp);
    };
  }, [isDragging]);

  // ── Render ─────────────────────────────────────────────────────────────────
  return (
    <>
      {/* Base Map 1: Date 1 Clean Baseline Satellite */}
      <div
        ref={container1Ref}
        style={{
          position: "absolute",
          inset: 0,
        }}
      />

      {/* Overlay Map 2: Date 2 Satellite + Change Detection Layer (Clipped to right of split handle) */}
      <div
        ref={container2Ref}
        style={{
          position: "absolute",
          inset: 0,
          clipPath: `polygon(${swipePos}% 0, 100% 0, 100% 100%, ${swipePos}% 100%)`,
        }}
      />

      {/* Loading overlay */}
      {!ready && (
        <div className="map-placeholder" style={{ position: "absolute", inset: 0 }}>
          <div className="map-placeholder-icon">🛰️</div>
          <div className="map-placeholder-text">Loading Sentinel-2 Imagery & Hisar Mask…</div>
        </div>
      )}

      {/* ── Hisar AOI Boundary Badge ── */}
      {ready && (
        <div
          style={{
            position: "absolute",
            bottom: "24px",
            left: "50%",
            transform: "translateX(-50%)",
            background: "rgba(15, 23, 42, 0.9)",
            color: "#00f2fe",
            padding: "6px 14px",
            borderRadius: "20px",
            fontSize: "12px",
            fontWeight: 700,
            letterSpacing: "0.5px",
            backdropFilter: "blur(8px)",
            border: "1.5px solid #00f2fe",
            boxShadow: "0 0 12px rgba(0, 242, 254, 0.4)",
            zIndex: 35,
            display: "flex",
            alignItems: "center",
            gap: "8px",
          }}
        >
          <span style={{ width: 8, height: 8, borderRadius: "50%", background: "#00f2fe", boxShadow: "0 0 8px #00f2fe" }} />
          Analysis Boundary: Hisar AOI (hisar.geojson)
        </div>
      )}

      {/* ── Before ↔ After Split-Screen Compare Swipe Handle ── */}
      {ready && (
        <div
          style={{
            position: "absolute",
            top: 0,
            bottom: 0,
            left: `${swipePos}%`,
            width: "4px",
            background: "#3b82f6",
            boxShadow: "0 0 12px rgba(59, 130, 246, 0.9)",
            cursor: "col-resize",
            zIndex: 40,
            transform: "translateX(-2px)",
          }}
          onMouseDown={handleMouseDown}
        >
          {/* Draggable handle pill */}
          <div
            style={{
              position: "absolute",
              top: "50%",
              left: "50%",
              transform: "translate(-50%, -50%)",
              width: "40px",
              height: "40px",
              borderRadius: "50%",
              background: "#3b82f6",
              color: "#ffffff",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              fontSize: "16px",
              fontWeight: "bold",
              boxShadow: "0 2px 12px rgba(0, 0, 0, 0.6)",
              userSelect: "none",
            }}
          >
            ↔
          </div>

          {/* Labels for Before / After */}
          <div
            style={{
              position: "absolute",
              top: "16px",
              right: "14px",
              background: "rgba(15, 23, 42, 0.9)",
              color: "#e2e8f0",
              padding: "5px 10px",
              borderRadius: "6px",
              fontSize: "12px",
              fontWeight: 700,
              whiteSpace: "nowrap",
              backdropFilter: "blur(8px)",
              border: "1px solid rgba(255, 255, 255, 0.15)",
              boxShadow: "0 4px 12px rgba(0, 0, 0, 0.4)",
            }}
          >
            ← 2016 Sentinel-2 Image (Date 1)
          </div>
          <div
            style={{
              position: "absolute",
              top: "16px",
              left: "14px",
              background: "rgba(15, 23, 42, 0.9)",
              color: "#38bdf8",
              padding: "5px 10px",
              borderRadius: "6px",
              fontSize: "12px",
              fontWeight: 700,
              whiteSpace: "nowrap",
              backdropFilter: "blur(8px)",
              border: "1px solid rgba(56, 189, 248, 0.35)",
              boxShadow: "0 4px 12px rgba(0, 0, 0, 0.4)",
            }}
          >
            2026 Sentinel-2 Image + Changes (Date 2) →
          </div>
        </div>
      )}

      {/* Legend & Basemap Selector */}
      {ready && (
        <div className="map-legend">
          {/* Basemap selector */}
          <div style={{ marginBottom: 10, paddingBottom: 8, borderBottom: "1px solid rgba(255, 255, 255, 0.1)" }}>
            <div className="map-legend-title">Map View</div>
            <div style={{ display: "flex", gap: 4, marginTop: 4 }}>
              <button
                type="button"
                onClick={() => switchBasemap("satellite")}
                style={{
                  padding: "3px 8px",
                  fontSize: "10px",
                  borderRadius: "4px",
                  border: "none",
                  cursor: "pointer",
                  background: basemap === "satellite" ? "#3b82f6" : "rgba(255, 255, 255, 0.1)",
                  color: basemap === "satellite" ? "#fff" : "#94a3b8",
                  fontWeight: 600,
                }}
              >
                🛰️ Satellite
              </button>
              <button
                type="button"
                onClick={() => switchBasemap("streets")}
                style={{
                  padding: "3px 8px",
                  fontSize: "10px",
                  borderRadius: "4px",
                  border: "none",
                  cursor: "pointer",
                  background: basemap === "streets" ? "#3b82f6" : "rgba(255, 255, 255, 0.1)",
                  color: basemap === "streets" ? "#fff" : "#94a3b8",
                  fontWeight: 600,
                }}
              >
                🗺️ Streets
              </button>
              <button
                type="button"
                onClick={() => switchBasemap("dark")}
                style={{
                  padding: "3px 8px",
                  fontSize: "10px",
                  borderRadius: "4px",
                  border: "none",
                  cursor: "pointer",
                  background: basemap === "dark" ? "#3b82f6" : "rgba(255, 255, 255, 0.1)",
                  color: basemap === "dark" ? "#fff" : "#94a3b8",
                  fontWeight: 600,
                }}
              >
                🌙 Dark
              </button>
            </div>
          </div>

          <div className="map-legend-title">Change classes</div>
          <div className="legend-items">
            {LEGEND_ITEMS.map((item) => (
              <div key={item.key} className="legend-item">
                <div className="legend-swatch" style={{ background: item.color }} />
                <span>{item.label}</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </>
  );
}
