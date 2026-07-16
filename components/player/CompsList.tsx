"use client";

import { Comp } from "@/lib/types";
import { fmtRound, NO_DATA } from "@/lib/format";

export function CompsList({ comps }: { comps: Comp[] }) {
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
      <div className="mb-2 text-[13px]">
        <span className="font-semibold">
          {reached} of {comps.length}
        </span>{" "}
        <span className="text-ink-2">similar profiles reached MLB</span>
      </div>
      <div className="space-y-1">
        {comps.map((c, i) => (
          <div
            key={i}
            className="flex items-center gap-2 rounded border border-rule bg-paper-raised px-2.5 py-1.5 text-[12px]"
          >
            <span
              className="h-2 w-2 shrink-0 rounded-full"
              style={{
                background: c.reached_mlb ? "var(--grade-elite)" : "var(--nodata)",
              }}
              title={c.reached_mlb ? "Reached MLB" : "Has not reached MLB"}
            />
            <span className="min-w-0 flex-1 truncate">
              <span className="font-medium">{c.name ?? NO_DATA}</span>{" "}
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
        ))}
      </div>
    </div>
  );
}
