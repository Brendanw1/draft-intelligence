"use client";

import { pickToRound } from "@/lib/format";

// Empirical pick-to-round boundaries from actual MLB draft data (2015-2025).
const PICK_ROUND_BOUNDARIES = [
  40, 70, 107, 137, 167, 197, 227, 257, 287, 317,
  347, 377, 407, 437, 467, 497, 527, 557, 587, 617,
];

/**
 * Projected draft position rendered as what the model actually knows:
 * a round band (pick ± backtest MAE), with the point estimate as a tick.
 * Scale: picks 1–620 mapped onto a fixed track.
 */
export function BandBar({
  pick,
  band,
  width = 104,
  showLabel = true,
}: {
  pick: number | null;
  band: [number, number] | null;
  width?: number;
  showLabel?: boolean;
}) {
  if (pick == null || band == null)
    return <span className="text-[12px] text-ink-3">—</span>;
  const max = 620;
  const x = (v: number) => Math.min(1, Math.max(0, v / max)) * width;
  return (
    <span
      className="inline-flex items-center gap-2"
      title={`Point estimate: Pick ${Math.round(pick)} (R${pickToRound(pick)}). Band shows ±~110 pick backtest error margin.`}
    >
      <svg width={width} height={14} className="shrink-0" aria-hidden>
        <line x1={0} y1={7} x2={width} y2={7} stroke="var(--rule)" strokeWidth={1} />
        {/* round gridlines every 5 rounds using empirical boundaries */}
        {[5, 10, 15].map((r) => {
          const boundaryIdx = r - 1;
          const boundaryPick = boundaryIdx < PICK_ROUND_BOUNDARIES.length ? PICK_ROUND_BOUNDARIES[boundaryIdx] : r * 30;
          return (
            <line
              key={r}
              x1={x(boundaryPick)}
              y1={4}
              x2={x(boundaryPick)}
              y2={10}
              stroke="var(--rule)"
              strokeWidth={1}
            />
          );
        })}
        <rect
          x={x(band[0])}
          y={5}
          width={Math.max(2, x(band[1]) - x(band[0]))}
          height={4}
          rx={2}
          fill="var(--band)"
        />
        <line
          x1={x(pick)}
          y1={2}
          x2={x(pick)}
          y2={12}
          stroke="var(--band-strong)"
          strokeWidth={2}
        />
      </svg>
      {showLabel && (
        <span className="whitespace-nowrap text-[12px] font-medium">R{pickToRound(pick)} · pick {Math.round(pick)}</span>
      )}
    </span>
  );
}
