"use client";

import { useState } from "react";
import { Comp } from "@/lib/types";
import { fmtRound, NO_DATA } from "@/lib/format";

/**
 * Pick-axis dot plot showing where nearest-neighbor comps were drafted
 * and whether they reached MLB.
 * Picks 1–620 mapped onto the x-axis; each dot is one comp.
 */
function CompDotPlot({
  comps,
  projPick,
  hoveredIdx,
  onHover,
}: {
  comps: Comp[];
  projPick: number | null | undefined;
  hoveredIdx: number | null;
  onHover: (i: number | null) => void;
}) {
  const maxPick = 620;
  const w = 620;
  const h = 52;

  // Count comps with valid picks
  const valid = comps.filter((c) => c.pick != null && c.pick > 0);

  return (
    <svg
      viewBox={`0 0 ${w} ${h}`}
      className="mt-1 w-full max-w-full overflow-visible"
      aria-label="Comp draft-pick dot plot"
    >
      {/* Background axis line */}
      <line x1={0} y1={h / 2 + 4} x2={w} y2={h / 2 + 4} stroke="var(--rule)" strokeWidth={1} />

      {/* Round markers every 5 rounds */}
      {[1, 5, 10, 15, 20].map((r) => {
        const x = (r * 30.75 * w) / maxPick;
        return (
          <g key={r}>
            <line x1={x} y1={h / 2 - 1} x2={x} y2={h / 2 + 9} stroke="var(--rule-strong)" strokeWidth={1} />
            <text
              x={x}
              y={h - 2}
              textAnchor="middle"
              fill="var(--ink-3)"
              fontSize={8}
            >
              R{r}
            </text>
          </g>
        );
      })}

      {/* Dots for each comp */}
      {valid.map((c, i) => {
        const x = Math.min(w, Math.max(0, (c.pick! / maxPick) * w));
        const yOff = ((i % 5) - 2) * 3;
        const cy = h / 2 + 4 + yOff;
        const isHovered = hoveredIdx === i;
        const baseFill = c.reached_mlb ? "var(--grade-elite)" : "var(--nodata)";
        const accentFill = c.reached_mlb ? "var(--grade-elite)" : "var(--ink-3)";

        return (
          <g
            key={i}
            onMouseEnter={() => onHover(i)}
            onMouseLeave={() => onHover(null)}
            style={{ cursor: "pointer" }}
          >
            {/* Native browser tooltip with player details */}
            <title>{c.name}{c.pick != null ? ` · pick ${c.pick}${c.round ? ` (R${c.round})` : ""}` : ""}{c.reached_mlb ? " · reached MLB" : c.peak_level ? ` · peak ${c.peak_level}` : ""}</title>

            {/* Vertical guide line on hover — drops from dot to axis */}
            {isHovered && (
              <line
                x1={x}
                y1={h / 2 + 4}
                x2={x}
                y2={h / 2 + 4 + yOff}
                stroke={accentFill}
                strokeWidth={1}
                strokeDasharray="2 1.5"
                opacity={0.5}
              />
            )}

            {/* Glow ring on hover */}
            {isHovered && (
              <circle
                cx={x}
                cy={cy}
                r={7}
                fill="none"
                stroke={accentFill}
                strokeWidth={2}
                opacity={0.3}
              />
            )}

            {/* Main dot */}
            <circle
              cx={x}
              cy={cy}
              r={isHovered ? 5 : 2.5}
              fill={isHovered ? accentFill : baseFill}
              stroke={accentFill}
              strokeWidth={isHovered ? 1.5 : 0.5}
              style={{ transition: "r 0.12s ease, fill 0.12s ease" }}
            />

            {/* Player name label on hover — floats above the dot row */}
            {isHovered && (
              <text
                x={x}
                y={cy - 10}
                textAnchor="middle"
                fill="var(--ink)"
                fontSize={9}
                fontWeight={600}
                style={{ pointerEvents: "none" }}
              >
                {c.name ?? "—"}
              </text>
            )}
          </g>
        );
      })}

      {/* Reference line for projected pick */}
      {projPick != null && (
        <line
          x1={(projPick / maxPick) * w}
          y1={4}
          x2={(projPick / maxPick) * w}
          y2={h - 10}
          stroke="var(--band-strong)"
          strokeWidth={1.5}
          strokeDasharray="3 2"
        />
      )}
    </svg>
  );
}

export function CompsList({
  comps,
  projPick,
}: {
  comps: Comp[];
  projPick?: number | null;
}) {
  const [hoveredIdx, setHoveredIdx] = useState<number | null>(null);

  if (!comps.length)
    return (
      <div className="rounded bg-paper-sunken p-3 text-[12px] text-ink-3">
        No comps — needs a qualified 2026 stat line to match against drafted
        players.
      </div>
    );
  const reached = comps.filter((c) => c.reached_mlb).length;
  return (
    <div>
      <div className="mb-1 text-[13px]">
        <span className="font-semibold">
          {reached} of {comps.length}
        </span>{" "}
        <span className="text-ink-2">similar profiles reached MLB</span>
      </div>

      {/* Dot plot pick-axis visualization */}
      <CompDotPlot
        comps={comps}
        projPick={projPick}
        hoveredIdx={hoveredIdx}
        onHover={setHoveredIdx}
      />

      <div className="mt-2 space-y-1">
        {comps.map((c, i) => {
          const isHovered = hoveredIdx === i;
          return (
            <div
              key={i}
              onMouseEnter={() => setHoveredIdx(i)}
              onMouseLeave={() => setHoveredIdx(null)}
              className={`flex items-center gap-2 rounded border px-2.5 py-1.5 text-[12px] transition-all duration-100 ${
                isHovered
                  ? "border-maroon bg-maroon-soft shadow-sm"
                  : "border-rule bg-paper-raised"
              }`}
              title={`Similarity distance: ${c.dist != null ? c.dist.toFixed(3) : "?"}`}
              style={{ cursor: "pointer" }}
            >
              <span
                className="h-2 w-2 shrink-0 rounded-full"
                style={{
                  background: c.reached_mlb ? "var(--grade-elite)" : "var(--nodata)",
                }}
                title={c.reached_mlb ? "Reached MLB" : "Has not reached MLB"}
              />
              <span className="min-w-0 flex-1 truncate">
                <span className={`font-medium ${isHovered ? "text-maroon" : ""}`}>
                  {c.name ?? NO_DATA}
                </span>{" "}
                <span className="text-ink-3">
                  {c.school ?? ""} · {c.year ?? ""}
                </span>
              </span>
              <span className="shrink-0 text-ink-2">
                {c.pick != null
                  ? `pick ${c.pick}${c.round ? ` (${fmtRound(c.round)})` : ""}`
                  : NO_DATA}
              </span>
              <span className="w-[68px] shrink-0 text-right">
                {c.reached_mlb ? (
                  <span className="font-semibold text-[var(--grade-elite)]">MLB</span>
                ) : (
                  <span className="text-ink-2">{c.peak_level ? `peak ${c.peak_level}` : "—"}</span>
                )}
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
}
