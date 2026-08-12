"use client";

interface RegionProps {
  region_id: number;
  change_class: string;
  change_direction: string;
  area_ha: number;
  area_km2: number;
  date1: string;
  date2: string;
}

interface RegionDetailProps {
  region: RegionProps | null;
  onClose: () => void;
}

const CLASS_LABELS: Record<string, string> = {
  built_up: "Built-up",
  vegetation: "Vegetation",
  water: "Water",
  soil: "Soil / Bare land",
};

function fmt(n: number, dp = 4): string {
  return n.toLocaleString("en-IN", { maximumFractionDigits: dp, minimumFractionDigits: dp });
}

export default function RegionDetail({ region, onClose }: RegionDetailProps) {
  if (!region) return null;

  const className = CLASS_LABELS[region.change_class] ?? region.change_class;
  const directionIcon = region.change_direction === "increase" ? "↑" : "↓";

  return (
    <div className="region-detail">
      <div className="region-detail-header">
        <div className="region-detail-title">Region #{region.region_id}</div>
        <button className="region-detail-close" onClick={onClose} aria-label="Close">
          ×
        </button>
      </div>

      <div className="region-detail-rows">
        <div className="region-detail-row">
          <span className="region-detail-key">Change type</span>
          <span className={`class-pill ${region.change_class}`}>
            {directionIcon} {className}
          </span>
        </div>

        <div className="region-detail-row">
          <span className="region-detail-key">Area</span>
          <span className="region-detail-val">{fmt(region.area_ha, 2)} ha</span>
        </div>

        <div className="region-detail-row">
          <span className="region-detail-key"></span>
          <span className="region-detail-val" style={{ fontSize: 11, color: "var(--text-muted)" }}>
            {fmt(region.area_km2, 6)} km²
          </span>
        </div>

        <hr style={{ border: "none", borderTop: "1px solid var(--border)", margin: "2px 0" }} />

        <div className="region-detail-row">
          <span className="region-detail-key">Before</span>
          <span className="region-detail-val" style={{ fontSize: 10 }}>{region.date1}</span>
        </div>

        <div className="region-detail-row">
          <span className="region-detail-key">After</span>
          <span className="region-detail-val" style={{ fontSize: 10 }}>{region.date2}</span>
        </div>
      </div>
    </div>
  );
}
