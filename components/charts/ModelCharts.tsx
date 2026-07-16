"use client";

import { useState } from "react";
import { BacktestRow, CalibrationBin } from "@/lib/types";
import { fmtPct } from "@/lib/format";

/** Reliability diagram: predicted (x) vs actual (y) per calibration bin. */
export function ReliabilityDiagram({ bins }: { bins: CalibrationBin[] }) {
  const [hover, setHover] = useState<CalibrationBin | null>(null);
  const W = 380;
  const H = 300;
  const M = { top: 16, right: 16, bottom: 40, left: 46 };
  const x = (v: number) => M.left + v * (W - M.left - M.right);
  const y = (v: number) => H - M.bottom - v * (H - M.top - M.bottom);
  const r = (count: number) => Math.max(3.5, Math.min(11, Math.sqrt(count) * 0.9));

  return (
    <div>
      <svg width={W} height={H} role="img" aria-label="Reliability diagram: predicted probability versus actual MLB rate per bin">
        {[0, 0.25, 0.5, 0.75, 1].map((t) => (
          <g key={t}>
            <line x1={x(t)} y1={y(0)} x2={x(t)} y2={y(1)} stroke="var(--rule)" />
            <line x1={x(0)} y1={y(t)} x2={x(1)} y2={y(t)} stroke="var(--rule)" />
            <text x={x(t)} y={H - M.bottom + 14} textAnchor="middle" fontSize={10} fill="var(--ink-3)">
              {fmtPct(t)}
            </text>
            <text x={M.left - 8} y={y(t) + 3} textAnchor="end" fontSize={10} fill="var(--ink-3)">
              {fmtPct(t)}
            </text>
          </g>
        ))}
        {/* perfect calibration reference */}
        <line x1={x(0)} y1={y(0)} x2={x(1)} y2={y(1)} stroke="var(--ref-line)" strokeDasharray="4 3" strokeWidth={1.5} />
        <text x={x(0.78)} y={y(0.84)} fontSize={10} fill="var(--ink-3)" transform={`rotate(-38 ${x(0.78)} ${y(0.84)})`}>
          perfectly calibrated
        </text>
        {/* observed curve */}
        <polyline
          points={bins.map((b) => `${x(b.pred_mean)},${y(b.actual_rate)}`).join(" ")}
          fill="none"
          stroke="var(--grade-elite)"
          strokeWidth={2}
        />
        {bins.map((b) => (
          <circle
            key={b.bin}
            cx={x(b.pred_mean)}
            cy={y(b.actual_rate)}
            r={r(b.count)}
            fill="var(--grade-elite)"
            fillOpacity={0.75}
            stroke="var(--paper)"
            strokeWidth={2}
            onMouseEnter={() => setHover(b)}
            onMouseLeave={() => setHover(null)}
            style={{ cursor: "help" }}
          />
        ))}
        <text x={(W + M.left - M.right) / 2} y={H - 6} textAnchor="middle" fontSize={10} fill="var(--ink-3)">
          model-predicted probability →
        </text>
        <text x={12} y={(H + M.top - M.bottom) / 2} textAnchor="middle" fontSize={10} fill="var(--ink-3)" transform={`rotate(-90 12 ${(H + M.top - M.bottom) / 2})`}>
          actual MLB rate →
        </text>
      </svg>
      <div className="h-6 text-[11px] text-ink-2">
        {hover ? (
          <>
            bin {hover.bin}: {hover.count} players · predicted {fmtPct(hover.pred_mean)} →
            actually reached {fmtPct(hover.actual_rate)}
          </>
        ) : (
          <span className="text-ink-3">
            Points below the line = overconfidence. Dot size = players in bin.
          </span>
        )}
      </div>
    </div>
  );
}

/** Backtest MAE vs baseline as paired bars per test year (lower is better). */
export function BacktestBars({ rows }: { rows: BacktestRow[] }) {
  const W = 380;
  const H = 220;
  const M = { top: 18, right: 12, bottom: 34, left: 40 };
  const max = Math.max(...rows.flatMap((r) => [r.mae, r.baseline_mae])) * 1.12;
  const y = (v: number) => H - M.bottom - (v / max) * (H - M.top - M.bottom);
  const groupW = (W - M.left - M.right) / rows.length;
  const barW = Math.min(34, groupW / 2 - 8);

  return (
    <div>
      <svg width={W} height={H} role="img" aria-label="Backtest pick error versus naive baseline per holdout year">
        {[50, 100, 150].map((t) =>
          t < max ? (
            <g key={t}>
              <line x1={M.left} y1={y(t)} x2={W - M.right} y2={y(t)} stroke="var(--rule)" />
              <text x={M.left - 6} y={y(t) + 3} textAnchor="end" fontSize={10} fill="var(--ink-3)">
                {t}
              </text>
            </g>
          ) : null,
        )}
        {rows.map((r, i) => {
          const cx = M.left + groupW * i + groupW / 2;
          return (
            <g key={r.test_year}>
              <rect
                x={cx - barW - 2}
                y={y(r.baseline_mae)}
                width={barW}
                height={H - M.bottom - y(r.baseline_mae)}
                rx={3}
                fill="var(--grade-low)"
              >
                <title>{`${r.test_year} naive baseline: ${r.baseline_mae} picks MAE`}</title>
              </rect>
              <rect
                x={cx + 2}
                y={y(r.mae)}
                width={barW}
                height={H - M.bottom - y(r.mae)}
                rx={3}
                fill="var(--grade-elite)"
              >
                <title>{`${r.test_year} model: ${r.mae} picks MAE · Spearman ρ ${r.spearman_rho} · n=${r.n_test}`}</title>
              </rect>
              <text x={cx} y={H - M.bottom + 14} textAnchor="middle" fontSize={10} fill="var(--ink-3)">
                {r.test_year}
              </text>
              <text x={cx} y={H - M.bottom + 26} textAnchor="middle" fontSize={9} fill="var(--ink-3)">
                ρ {r.spearman_rho.toFixed(2)}
              </text>
            </g>
          );
        })}
        <text x={M.left} y={11} fontSize={10} fill="var(--ink-3)">
          pick error (MAE, lower better) —{" "}
        </text>
        <rect x={M.left + 172} y={4} width={9} height={9} fill="var(--grade-low)" rx={2} />
        <text x={M.left + 185} y={11} fontSize={10} fill="var(--ink-3)">
          baseline
        </text>
        <rect x={M.left + 234} y={4} width={9} height={9} fill="var(--grade-elite)" rx={2} />
        <text x={M.left + 247} y={11} fontSize={10} fill="var(--ink-3)">
          model
        </text>
      </svg>
    </div>
  );
}

/** Horizontal feature-importance bars, sparse-signal features flagged. */
export function ImportanceBars({
  importance,
  flagged = [],
}: {
  importance: { feature: string; importance: number }[];
  flagged?: string[];
}) {
  const max = Math.max(...importance.map((d) => d.importance));
  return (
    <div className="space-y-[5px]">
      {importance.map((d) => {
        const isFlagged = flagged.includes(d.feature);
        return (
          <div key={d.feature} className="flex items-center gap-2">
            <span
              className="w-[110px] shrink-0 truncate text-right text-[11px] text-ink-2"
              title={d.feature}
            >
              {d.feature.replace(/^fg_/, "").replace(/_/g, " ")}
              {isFlagged && " ⚠"}
            </span>
            <div className="h-[9px] flex-1 overflow-hidden rounded-sm bg-paper-sunken">
              <div
                className="h-full rounded-sm"
                style={{
                  width: `${(d.importance / max) * 100}%`,
                  background: isFlagged ? "var(--flag)" : "var(--band-strong)",
                }}
                title={
                  isFlagged
                    ? `${d.feature}: flagged — sparse rare-event stat, may be a small-sample artifact`
                    : `${d.feature}: ${d.importance}`
                }
              />
            </div>
            <span className="w-[40px] shrink-0 text-[10px] text-ink-3">
              {(d.importance * 100).toFixed(1)}%
            </span>
          </div>
        );
      })}
    </div>
  );
}
