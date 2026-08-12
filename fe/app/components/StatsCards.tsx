"use client";

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

interface StatsCardProps {
  stats: Statistics | null;
}

const CLASS_META: Record<
  string,
  { label: string; cssClass: string; icon: string }
> = {
  built_up_increase: { label: "Built-up ↑", cssClass: "built_up", icon: "🏗️" },
  built_up_decrease: { label: "Built-up ↓", cssClass: "built_up", icon: "🏚️" },
  vegetation_increase: { label: "Vegetation ↑", cssClass: "vegetation", icon: "🌿" },
  vegetation_decrease: { label: "Vegetation ↓", cssClass: "vegetation", icon: "🍂" },
  water_increase: { label: "Water ↑", cssClass: "water", icon: "💧" },
  water_decrease: { label: "Water ↓", cssClass: "water", icon: "🏜️" },
  soil_increase: { label: "Soil ↑", cssClass: "soil", icon: "🪨" },
  soil_decrease: { label: "Soil ↓", cssClass: "soil", icon: "🪨" },
};

function fmt(n: number, dp = 2): string {
  return n.toLocaleString("en-IN", { maximumFractionDigits: dp, minimumFractionDigits: dp });
}

export default function StatsCards({ stats }: StatsCardProps) {
  if (!stats) {
    return (
      <div className="stats-row">
        <div className="stat-card" style={{ opacity: 0.4 }}>
          <div className="stat-label">Total changed area</div>
          <div className="stat-value" style={{ fontSize: 14, color: "var(--text-muted)" }}>
            Run an analysis to see statistics
          </div>
        </div>
      </div>
    );
  }

  const classEntries = Object.entries(stats.changes_by_class);

  return (
    <div className="stats-row">
      {/* Total */}
      <div className="stat-card total">
        <div className="stat-label">Total changed</div>
        <div className="stat-value">
          {fmt(stats.total_changed_area_ha)}{" "}
          <span className="stat-unit">ha</span>
        </div>
        <div className="stat-sub">{fmt(stats.total_changed_area_km2, 3)} km²</div>
      </div>

      {/* Per-class cards */}
      {classEntries.map(([key, cs]) => {
        const meta = CLASS_META[key];
        if (!meta || cs.area_ha === 0) return null;
        const baseClass = key.replace(/_increase|_decrease/, "");
        return (
          <div key={key} className={`stat-card ${baseClass}`}>
            <div className="stat-label">
              {meta.icon} {meta.label}
            </div>
            <div className="stat-value">
              {fmt(cs.area_ha)}{" "}
              <span className="stat-unit">ha</span>
            </div>
            <div className="stat-sub">{fmt(cs.area_km2, 4)} km²</div>
          </div>
        );
      })}

      {/* Date range */}
      <div className="stat-card" style={{ borderLeft: "3px solid var(--text-muted)" }}>
        <div className="stat-label">Period</div>
        <div style={{ fontSize: 11, color: "var(--text-secondary)", marginTop: 2, fontFamily: "JetBrains Mono, monospace", lineHeight: 1.7 }}>
          <div>{stats.date1}</div>
          <div style={{ color: "var(--text-muted)" }}>→</div>
          <div>{stats.date2}</div>
        </div>
      </div>
    </div>
  );
}
