"use client";

import { ordinal } from "@/lib/format";

export function PctlBars({
  pctl,
  panel,
}: {
  pctl: Record<string, number>;
  panel: { stat: string; label: string; why: string }[];
}) {
  const rows = panel.filter((d) => pctl[d.stat] != null);
  if (!rows.length)
    return <div className="text-[12px] text-ink-3">No percentile data.</div>;
  return (
    <div className="space-y-[7px]">
      {rows.map((d) => {
        const v = pctl[d.stat];
        const strong = v >= 70;
        const weak = v <= 30;
        return (
          <div key={d.stat} className="flex items-center gap-2" title={d.why}>
            <span className="w-[52px] shrink-0 text-[12px] font-medium">{d.label}</span>
            <span
              aria-hidden
              className={`w-3 text-[11px] ${strong ? "text-[var(--grade-elite)]" : weak ? "text-ink-3" : "text-transparent"}`}
            >
              {strong ? "▲" : weak ? "▼" : "·"}
            </span>
            <div className="relative h-[8px] flex-1 overflow-hidden rounded-sm bg-paper-sunken">
              <div
                className="absolute inset-y-0 left-0 rounded-sm"
                style={{
                  width: `${v}%`,
                  background: strong
                    ? "var(--grade-elite)"
                    : weak
                      ? "var(--grade-low)"
                      : "var(--grade-medium)",
                }}
              />
              {/* median reference */}
              <div className="absolute inset-y-0 left-1/2 w-px bg-[var(--ref-line)]" />
            </div>
            <span className="w-[46px] shrink-0 text-right text-[12px] text-ink-2">
              {ordinal(v)}
            </span>
          </div>
        );
      })}
      <div className="pt-1 text-[10px] leading-snug text-ink-3">
        Ordered by Tier 1 model feature importance. Direction-adjusted: higher is
        always better. Not per-player attribution — the model’s priorities, this
        player’s standing.
      </div>
    </div>
  );
}
