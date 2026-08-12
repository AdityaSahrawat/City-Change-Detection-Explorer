"use client";

interface Analysis {
  id: string;
  created_at: string;
  date1_safe_path: string;
  date2_safe_path: string;
  status: "pending" | "complete" | "error";
  error?: string | null;
}

interface AnalysisListProps {
  analyses: Analysis[];
  selectedId: string | null;
  onSelect: (id: string) => void;
}

function shortPath(p: string): string {
  const parts = p.replace(/\\/g, "/").split("/");
  const name = parts[parts.length - 1] || p;
  return name.length > 28 ? name.slice(0, 25) + "…" : name;
}

function formatDate(iso: string): string {
  try {
    return new Date(iso).toLocaleString("en-IN", {
      day: "2-digit",
      month: "short",
      hour: "2-digit",
      minute: "2-digit",
    });
  } catch {
    return iso;
  }
}

export default function AnalysisList({ analyses, selectedId, onSelect }: AnalysisListProps) {
  if (analyses.length === 0) {
    return (
      <div className="empty-list">
        <div style={{ fontSize: 28, marginBottom: 8 }}>📡</div>
        No analyses yet.
        <br />
        Fill in the form above and click
        <br />
        <strong style={{ color: "var(--text-secondary)" }}>Run Analysis</strong>.
      </div>
    );
  }

  return (
    <div className="analyses-list">
      {analyses.map((a) => (
        <div
          key={a.id}
          className={`analysis-item ${selectedId === a.id ? "selected" : ""}`}
          onClick={() => onSelect(a.id)}
          role="button"
          tabIndex={0}
          onKeyDown={(e) => e.key === "Enter" && onSelect(a.id)}
        >
          <div className="analysis-item-id">{a.id.slice(0, 8)}…</div>
          <div className="analysis-item-dates" title={a.date1_safe_path}>
            {shortPath(a.date1_safe_path)} →{" "}
            {shortPath(a.date2_safe_path)}
          </div>
          <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
            <span className={`status-badge ${a.status}`}>
              <span className="status-dot" />
              {a.status}
            </span>
            <span style={{ fontSize: 10, color: "var(--text-muted)" }}>
              {formatDate(a.created_at)}
            </span>
          </div>
          {a.status === "error" && a.error && (
            <div style={{ fontSize: 10, color: "var(--accent-red)", marginTop: 4, whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>
              {a.error}
            </div>
          )}
        </div>
      ))}
    </div>
  );
}
