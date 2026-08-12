"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import dynamic from "next/dynamic";
import AnalysisList from "./components/AnalysisList";
import StatsCards from "./components/StatsCards";
import RegionDetail from "./components/RegionDetail";

const ChangeMap = dynamic(() => import("./components/ChangeMap"), {
  ssr: false,
  loading: () => (
    <div className="map-placeholder">
      <div className="map-placeholder-icon">🗺️</div>
      <div className="map-placeholder-text">Loading map…</div>
    </div>
  ),
});

const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

// ── Types ─────────────────────────────────────────────────────────────────────

interface Analysis {
  id: string;
  created_at: string;
  date1_safe_path: string;
  date2_safe_path: string;
  aoi_geojson_path: string;
  status: "pending" | "complete" | "error";
  error?: string | null;
}

interface ClassStats {
  area_ha: number;
  area_km2: number;
  pixel_count: number;
}

interface Statistics {
  total_changed_area_ha: number;
  total_changed_area_km2: number;
  total_changed_pixels: number;
  changes_by_class: Record<string, ClassStats>;
  date1: string;
  date2: string;
}

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

const ALL_CLASSES = ["built_up", "vegetation", "water", "soil"] as const;
type ChangeClass = (typeof ALL_CLASSES)[number];

const CLASS_META: Record<ChangeClass, { label: string; color: string; icon: string }> = {
  built_up:   { label: "Built-up",   color: "#fc8181", icon: "🏗️" },
  vegetation: { label: "Vegetation", color: "#68d391", icon: "🌿" },
  water:      { label: "Water",      color: "#63b3ed", icon: "💧" },
  soil:       { label: "Soil",       color: "#f6ad55", icon: "🪨" },
};

// ── Main page ─────────────────────────────────────────────────────────────────

export default function HomePage() {
  // Form
  const [date1Path, setDate1Path] = useState("data/raw/S2A_MSIL2A_20160604T052652_N0500_R105_T43REN_20231003T092400.SAFE");
  const [date2Path, setDate2Path] = useState("data/raw/S2A_MSIL2A_20260714T053241_N0512_R105_T43REN_20260714T102412.SAFE");
  const [aoiPath, setAoiPath]     = useState("data/hisar.geojson");

  // API state
  const [analyses, setAnalyses]         = useState<Analysis[]>([]);
  const [selectedId, setSelectedId]     = useState<string | null>(null);
  const [statistics, setStatistics]     = useState<Statistics | null>(null);
  const [changesGeoJSON, setChangesGeoJSON] = useState<ChangesGeoJSON | null>(null);
  const [selectedRegion, setSelectedRegion] = useState<ChangeFeatureProps | null>(null);

  // Sentinel-2 satellite image overlays
  const [date1ImageUrl, setDate1ImageUrl] = useState<string | null>(null);
  const [date2ImageUrl, setDate2ImageUrl] = useState<string | null>(null);
  const [imageCoords, setImageCoords]     = useState<number[][] | null>(null);

  // UI state
  const [submitting, setSubmitting] = useState(false);
  const [formError, setFormError]   = useState<string | null>(null);

  // ── Slider / filter state ──────────────────────────────────────────────────
  const [opacity, setOpacity]           = useState(55);       // 0–100
  const [minAreaHa, setMinAreaHa]       = useState(0);        // ha
  const [activeClasses, setActiveClasses] = useState<Set<ChangeClass>>(
    new Set(ALL_CLASSES)
  );

  const pollingRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // ── Filtered GeoJSON (applied before passing to map) ──────────────────────
  const filteredGeoJSON: ChangesGeoJSON | null = changesGeoJSON
    ? {
        type: "FeatureCollection",
        features: changesGeoJSON.features.filter(
          (f) =>
            activeClasses.has(f.properties.change_class as ChangeClass) &&
            f.properties.area_ha >= minAreaHa
        ),
      }
    : null;

  // ── API helpers ────────────────────────────────────────────────────────────
  const fetchAnalyses = useCallback(async () => {
    try {
      const res = await fetch(`${API_BASE}/analyses`);
      if (!res.ok) return;
      setAnalyses(await res.json());
    } catch { /* silent if backend not up */ }
  }, []);

  useEffect(() => { fetchAnalyses(); }, [fetchAnalyses]);

  const loadResults = useCallback(async (id: string) => {
    try {
      const [sRes, cRes, dRes] = await Promise.all([
        fetch(`${API_BASE}/analyses/${id}/statistics`),
        fetch(`${API_BASE}/analyses/${id}/changes`),
        fetch(`${API_BASE}/analyses/${id}`),
      ]);
      if (sRes.ok) setStatistics(await sRes.json());
      if (cRes.ok) setChangesGeoJSON(await cRes.json());
      if (dRes.ok) {
        const detail = await dRes.json();
        setDate1ImageUrl(detail.date1_image_url ?? null);
        setDate2ImageUrl(detail.date2_image_url ?? null);
        setImageCoords(detail.image_coords ?? null);
      }
    } catch (e) { console.error(e); }
  }, []);

  const pollPending = useCallback((id: string) => {
    pollingRef.current = setTimeout(async () => {
      try {
        const res = await fetch(`${API_BASE}/analyses/${id}`);
        if (!res.ok) return;
        const a: Analysis = await res.json();
        setAnalyses((prev) =>
          prev.map((x) => (x.id === id ? { ...x, status: a.status, error: a.error ?? null } : x))
        );
        if (a.status === "pending") pollPending(id);
        else if (a.status === "complete") loadResults(id);
      } catch { pollPending(id); }
    }, 2500);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const handleSelect = useCallback(async (id: string) => {
    setSelectedId(id);
    setStatistics(null);
    setChangesGeoJSON(null);
    setSelectedRegion(null);
    setDate1ImageUrl(null);
    setDate2ImageUrl(null);
    setImageCoords(null);
    const a = analyses.find((x) => x.id === id);
    if (!a) return;
    if (a.status === "complete") loadResults(id);
    else if (a.status === "pending") pollPending(id);
  }, [analyses, loadResults, pollPending]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!date2Path.trim()) { setFormError("Date-2 .SAFE path is required."); return; }
    setFormError(null);
    setSubmitting(true);
    try {
      const res = await fetch(`${API_BASE}/analyses`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          date1_safe_path: date1Path.trim() || "synthetic",
          date2_safe_path: date2Path.trim(),
          aoi_geojson_path: aoiPath.trim() || "data/hisar.geojson",
        }),
      });
      if (!res.ok) { const err = await res.json(); setFormError(err.detail ?? "Failed to start analysis."); return; }
      const newAnalysis: Analysis = await res.json();
      setAnalyses((prev) => [newAnalysis, ...prev]);
      setSelectedId(newAnalysis.id);
      setStatistics(null);
      setChangesGeoJSON(null);
      setSelectedRegion(null);
      pollPending(newAnalysis.id);
    } catch { setFormError("Cannot reach backend — is FastAPI running on port 8000?"); }
    finally { setSubmitting(false); }
  };

  const toggleClass = (cls: ChangeClass) => {
    setActiveClasses((prev) => {
      const next = new Set(prev);
      if (next.has(cls)) { if (next.size > 1) next.delete(cls); }
      else next.add(cls);
      return next;
    });
  };

  const maxAreaHa = changesGeoJSON
    ? Math.max(0, ...changesGeoJSON.features.map((f) => f.properties.area_ha))
    : 100;

  useEffect(() => () => { if (pollingRef.current) clearTimeout(pollingRef.current); }, []);

  // ── Render ─────────────────────────────────────────────────────────────────
  return (
    <div className="app-shell">
      {/* Topbar */}
      <header className="topbar">
        <span className="topbar-logo">🛰️ Change Detection Explorer</span>
        <span className="topbar-subtitle">Sentinel-2 · Hisar, Haryana</span>
        <div className="topbar-spacer" />
        <span className="topbar-badge">2016 → 2026 · 10-year change</span>
      </header>

      <div className="workspace">
        {/* Sidebar */}
        <aside className="sidebar">
          <div className="sidebar-section">
            <div className="sidebar-section-title">New Analysis</div>
            <form className="analysis-form" onSubmit={handleSubmit}>
              <div className="form-group">
                <label className="form-label" htmlFor="date1-path">Date 1 — .SAFE path</label>
                <input id="date1-path" className="form-input" value={date1Path}
                  onChange={(e) => setDate1Path(e.target.value)} autoComplete="off" />
                <span className="form-hint">Use <code>synthetic</code> if you only have one scene</span>
              </div>

              <div className="form-group">
                <label className="form-label" htmlFor="date2-path">Date 2 — .SAFE path *</label>
                <input id="date2-path" className="form-input" value={date2Path}
                  onChange={(e) => setDate2Path(e.target.value)} autoComplete="off" required />
              </div>

              <div className="form-group">
                <label className="form-label" htmlFor="aoi-path">AOI GeoJSON path</label>
                <input id="aoi-path" className="form-input" value={aoiPath}
                  onChange={(e) => setAoiPath(e.target.value)} autoComplete="off" />
              </div>

              {formError && <div className="error-banner" role="alert">{formError}</div>}

              <button type="submit" className="btn-primary" id="run-analysis-btn" disabled={submitting}>
                {submitting ? (<><span className="spinner" />Starting…</>) : <>▶ Run Analysis</>}
              </button>
            </form>
          </div>

          <div className="sidebar-section" style={{ borderBottom: "none", paddingBottom: 0 }}>
            <div className="sidebar-section-title">Past Analyses ({analyses.length})</div>
          </div>
          <AnalysisList analyses={analyses} selectedId={selectedId} onSelect={handleSelect} />
        </aside>

        {/* Main */}
        <main className="main-panel">
          {/* Stats row */}
          <StatsCards stats={statistics} />

          {/* ── Map controls bar (slider lives here) ── */}
          {changesGeoJSON && (
            <div className="map-controls-bar">
              {/* Opacity slider */}
              <div className="slider-group">
                <span className="slider-label">Opacity</span>
                <input
                  id="opacity-slider"
                  type="range"
                  className="range-slider"
                  min={10} max={100} step={5}
                  value={opacity}
                  onChange={(e) => setOpacity(Number(e.target.value))}
                />
                <span className="slider-val">{opacity}%</span>
              </div>

              {/* Min area slider */}
              <div className="slider-group">
                <span className="slider-label">Min area</span>
                <input
                  id="area-slider"
                  type="range"
                  className="range-slider"
                  min={0}
                  max={Math.min(maxAreaHa, 50)}
                  step={0.5}
                  value={minAreaHa}
                  onChange={(e) => setMinAreaHa(Number(e.target.value))}
                />
                <span className="slider-val">≥ {minAreaHa.toFixed(1)} ha</span>
              </div>

              {/* Class toggles */}
              <div className="class-toggles">
                {ALL_CLASSES.map((cls) => (
                  <button
                    key={cls}
                    className={`class-toggle ${activeClasses.has(cls) ? "on" : "off"}`}
                    style={{ "--cls-color": CLASS_META[cls].color } as React.CSSProperties}
                    onClick={() => toggleClass(cls)}
                    title={`Toggle ${CLASS_META[cls].label}`}
                  >
                    {CLASS_META[cls].icon} {CLASS_META[cls].label}
                  </button>
                ))}
              </div>

              {/* Live count */}
              <span className="filter-count">
                {filteredGeoJSON?.features.length ?? 0} regions
              </span>
            </div>
          )}

          {/* Map — position:relative so child absolute divs fill correctly */}
          <div className="map-container" style={{ position: "relative" }}>
            <ChangeMap
              geojson={filteredGeoJSON}
              opacity={opacity / 100}
              date1ImageUrl={date1ImageUrl}
              date2ImageUrl={date2ImageUrl}
              imageCoords={imageCoords}
              onRegionClick={(props) => setSelectedRegion(props)}
            />

            <RegionDetail region={selectedRegion} onClose={() => setSelectedRegion(null)} />

            {!changesGeoJSON && (
              <div style={{ position: "absolute", inset: 0, display: "flex", alignItems: "center", justifyContent: "center", pointerEvents: "none" }}>
                {!selectedId && (
                  <div style={{ background: "var(--glass-bg)", backdropFilter: "var(--glass-blur)", border: "1px solid var(--border)", borderRadius: "var(--radius-lg)", padding: "24px 32px", textAlign: "center", pointerEvents: "auto" }}>
                    <div className="map-placeholder-icon">🛰️</div>
                    <div className="map-placeholder-text" style={{ marginTop: 12 }}>No analysis selected</div>
                    <div className="map-placeholder-sub" style={{ marginTop: 6 }}>
                      Run an analysis from the sidebar — change polygons appear on the map with area in hectares per region.
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>

        </main>
      </div>
    </div>
  );
}
