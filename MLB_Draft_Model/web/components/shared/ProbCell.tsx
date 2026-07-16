"use client";

import { fmtPct } from "@/lib/format";

/**
 * Calibrated MLB probability with a reference tick for the historical rate of
 * the player's raw-score calibration bin — predicted vs. what actually happened
 * to players scored like this.
 */
export function ProbCell({
  p,
  raw,
  hist,
  width = 64,
}: {
  p: number | null;
  raw: number | null;
  hist: number | null;
  width?: number;
}) {
  if (p == null) return <span className="text-[12px] text-ink-3">—</span>;
  const x = (v: number) => Math.min(1, Math.max(0, v)) * width;
  return (
    <span
      className="inline-flex items-center gap-2"
      title={`Calibrated MLB probability ${fmtPct(p)}${raw != null ? ` (raw model ${fmtPct(raw)})` : ""}${hist != null ? `. Players the model scored like this reached MLB ${fmtPct(hist)} of the time.` : ""}`}
    >
      <span className="w-[38px] text-right text-[13px] font-semibold">{fmtPct(p)}</span>
      <svg width={width} height={12} aria-hidden className="shrink-0">
        <rect x={0} y={4} width={width} height={4} rx={2} fill="var(--paper-sunken)" />
        <rect x={0} y={4} width={Math.max(2, x(p))} height={4} rx={2} fill="var(--band-strong)" />
        {hist != null && (
          <line x1={x(hist)} y1={1} x2={x(hist)} y2={11} stroke="var(--ink-2)" strokeWidth={1.5} strokeDasharray="2 1.5" />
        )}
      </svg>
    </span>
  );
}
