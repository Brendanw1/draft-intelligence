"use client";

import { useMeta } from "@/lib/hooks";
import { IndexPlayer } from "@/lib/types";
import { fmtPct } from "@/lib/format";

export function StatTiles({ filtered }: { filtered: IndexPlayer[] }) {
  const { data: meta } = useMeta();
  const eliteHigh = filtered.filter((r) => r.grade === "elite" || r.grade === "high").length;
  const hitters = filtered.filter((r) => r.type === "hitter").length;

  const tiles: { label: string; value: string; sub?: string; warn?: boolean }[] = [
    {
      label: "Players in view",
      value: filtered.length.toLocaleString(),
      sub: `${hitters.toLocaleString()} H · ${(filtered.length - hitters).toLocaleString()} P`,
    },
    {
      label: "Elite + High grade",
      value: eliteHigh.toLocaleString(),
      sub: "top 5% of qualified composites",
    },
    {
      label: "Pick error (backtest)",
      value: meta ? `±${Math.round(Math.max(...Object.values(meta.pick_band_mae)))}` : "…",
      sub: meta ? `picks, ${meta.backtest_year} holdout — read rounds, not slots` : undefined,
      warn: true,
    },
    {
      label: "Raw model bias",
      value: meta ? "2.3×" : "…",
      sub: "overconfident pre-calibration — site shows calibrated %",
      warn: true,
    },
  ];

  return (
    <div className="flex items-stretch gap-0 border-b border-rule">
      {tiles.map((t, i) => (
        <div
          key={t.label}
          className={`flex-1 px-4 py-2.5 ${i > 0 ? "border-l border-rule" : ""}`}
        >
          <div className="text-[10px] font-semibold uppercase tracking-[0.13em] text-ink-3">
            {t.label}
          </div>
          <div className="flex items-baseline gap-2">
            <span
              className={`text-[20px] font-semibold ${t.warn ? "text-flag" : "text-ink"}`}
            >
              {t.value}
            </span>
            {t.sub && <span className="truncate text-[11px] text-ink-3">{t.sub}</span>}
          </div>
        </div>
      ))}
    </div>
  );
}
