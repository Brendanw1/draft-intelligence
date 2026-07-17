"use client";

import { useVirtualizer } from "@tanstack/react-virtual";
import { useMemo, useRef } from "react";
import { fmtNum, fmtPct, fmtRate3, fmtInt, fmtHeight, NO_DATA } from "@/lib/format";
import {
  GRADE_ORDER,
  GRADE_TIER_NOTE,
  GRADE_LABEL,
  HITTER_BOARD_STATS,
  PITCHER_BOARD_STATS,
  StatDef,
} from "@/lib/stats";
import { useUI } from "@/lib/store";
import { Grade, IndexPlayer } from "@/lib/types";
import { BandBar } from "@/components/shared/BandBar";
import { ProbCell } from "@/components/shared/ProbCell";
import { ConfDot, FlagChips, GradeChip, TypeBadge } from "@/components/shared/Chips";

function ArrivalCell({ p, nn }: { p: number | null; nn: number | null }) {
  if (p == null) return <span className="text-[11px] text-nodata">—</span>;
  const pct = (p * 100).toFixed(0);
  const tier =
    p > 0.20 ? "sig-elite" :
    p > 0.10 ? "sig-high" :
    p > 0.05 ? "sig-medium" :
    "sig-low";
  return (
    <span
      className="inline-flex items-center gap-1.5"
      title={`Tier 3 arrival probability: ${pct}%.${nn != null ? ` Nearest-neighbor comp rate: ${(nn * 100).toFixed(1)}%.` : ""} Predicts P(MLB debut | drafted) with round-anchored prior.`}
    >
      <span className={`w-[32px] text-right text-[13px] font-semibold`} style={{ color: `var(--${tier})` }}>
        {(p * 100).toFixed(0)}%
      </span>
      <svg width={36} height={10} className="shrink-0" aria-hidden>
        <rect x={0} y={3.5} width={36} height={3} rx={1.5} fill="var(--surface-1)" />
        <rect
          x={0} y={3.5}
          width={Math.max(2, p * 36)}
          height={3} rx={1.5}
          fill={`var(--${tier})`}
        />
        {nn != null && (
          <line
            x1={Math.min(1, nn) * 36}
            y1={0.5} x2={Math.min(1, nn) * 36} y2={9.5}
            stroke="var(--ref-line)" strokeWidth={1.5} strokeDasharray="2 1"
          />
        )}
      </svg>
    </span>
  );
}

type Row =
  | { kind: "tier"; grade: Grade; count: number }
  | { kind: "player"; p: IndexPlayer };

function fmtStat(def: StatDef, v: number | null | undefined): string {
  if (v == null) return NO_DATA;
  switch (def.kind) {
    case "rate3":
      return fmtRate3(v);
    case "pct1":
      return fmtPct(v, 1);
    case "num1":
      return fmtNum(v, 1);
    case "num2":
      return fmtNum(v, 2);
    case "int":
      return fmtInt(v);
  }
}

export function BoardTable({
  rows,
  typeMode,
  grouped,
  sort,
  dir,
  onSort,
}: {
  rows: IndexPlayer[];
  typeMode: "all" | "hitter" | "pitcher";
  grouped: boolean;
  sort: string;
  dir: "asc" | "desc";
  onSort: (key: string) => void;
}) {
  const { openDrawer, toggleCompare, compareIds } = useUI();
  const statDefs =
    typeMode === "pitcher"
      ? PITCHER_BOARD_STATS
      : typeMode === "hitter"
        ? HITTER_BOARD_STATS
        : [];

  const display: Row[] = useMemo(() => {
    if (!grouped) return rows.map((p) => ({ kind: "player" as const, p }));
    const out: Row[] = [];
    for (const g of GRADE_ORDER) {
      const members = rows.filter((r) => r.grade === g);
      if (!members.length) continue;
      out.push({ kind: "tier", grade: g, count: members.length });
      members.forEach((p) => out.push({ kind: "player", p }));
    }
    return out;
  }, [rows, grouped]);

  const parentRef = useRef<HTMLDivElement>(null);
  const virtualizer = useVirtualizer({
    count: display.length,
    getScrollElement: () => parentRef.current,
    estimateSize: (i) => (display[i].kind === "tier" ? 34 : 40),
    overscan: 20,
  });

  const Th = ({
    label,
    sortKey,
    align = "right",
    width,
    title,
  }: {
    label: string;
    sortKey?: string;
    align?: "left" | "right";
    width?: number;
    title?: string;
  }) => (
    <button
      disabled={!sortKey}
      onClick={() => sortKey && onSort(sortKey)}
      title={title}
      style={width ? { width } : undefined}
      className={`flex items-center gap-1 px-2 text-[11px] font-semibold uppercase tracking-wide text-ink-3 ${
        align === "right" ? "justify-end text-right" : "justify-start text-left"
      } ${sortKey ? "cursor-pointer hover:text-ink" : "cursor-default"}`}
    >
      {label}
      {sortKey && sort === sortKey && <span aria-hidden>{dir === "desc" ? "▾" : "▴"}</span>}
    </button>
  );

  return (
    <div className="flex min-w-0 flex-1 flex-col">
      {/* header */}
      <div className="flex h-9 items-center border-b border-rule-strong bg-paper pr-3">
        <div className="w-[30px]" />
        <div className="flex w-[200px] shrink-0 items-center">
          <Th label="Player" sortKey="name" align="left" />
        </div>
        <div className="w-[52px] shrink-0"><Th label="Ht" sortKey="height" title="Height in inches — physical profile signal. Imputed from conference+position averages when missing." /></div>
        <div className="w-[36px] shrink-0 text-center text-[10px] text-ink-3" title="Body-mass index: green ≥27 (sturdy), yellow 24–27, gray <24. Computed from height + draft-record weight or imputed.">BMI</div>
        <div className="w-[62px] shrink-0"><Th label="Grade" align="left" title="Composite percentile tier within qualified same-type players: elite top 1%, high next 4%, medium next 15%" /></div>
        <div className="w-[64px] shrink-0"><Th label="Comp" sortKey="composite" title="Composite score 0–100" /></div>
        <div className="w-[168px] shrink-0"><Th label="Proj. Round" sortKey="pick" align="left" title="Projected round band: pick ± backtest error (~110 picks). The tick is the point estimate." /></div>
        <div className="w-[122px] shrink-0"><Th label="Top 10%" sortKey="mlb_p" title="Probability of being drafted in the top 10 rounds (pick ~315). Isotonic-calibrated — the dashed tick is what history supports." /></div>
        <div className="w-[96px] shrink-0"><Th label="Arrival" sortKey="mlb_arrival" title="Tier 3: P(MLB debut | drafted). Elastic Net with round-anchored prior + nearest-neighbor comp rate." /></div>
        <div className="w-[44px] shrink-0"><Th label="Conf" title="Tier 1 confidence from projected pick depth" /></div>
        {statDefs.map((d) => (
          <div key={d.key} className="w-[64px] shrink-0">
            <Th label={d.label} sortKey={d.key} title={d.help} />
          </div>
        ))}
        <div className="flex-1 pl-2"><Th label="Flags" align="left" /></div>
      </div>

      {/* virtualized body */}
      <div ref={parentRef} className="thin-scroll flex-1 overflow-y-auto">
        <div style={{ height: virtualizer.getTotalSize(), position: "relative" }}>
          {virtualizer.getVirtualItems().map((vi) => {
            const row = display[vi.index];
            if (row.kind === "tier") {
              return (
                <div
                  key={`tier-${row.grade}`}
                  className="absolute left-0 flex w-full items-center gap-2 border-b border-rule bg-paper-sunken px-3"
                  style={{ top: vi.start, height: vi.size }}
                >
                  <GradeChip grade={row.grade} small />
                  <span className="text-[11px] text-ink-3">
                    {row.count.toLocaleString()} players · {GRADE_TIER_NOTE[row.grade]}
                  </span>
                </div>
              );
            }
            const p = row.p;
            const inCompare = compareIds.includes(p.id);
            return (
              <div
                key={p.id}
                role="button"
                tabIndex={0}
                onClick={() => openDrawer(p.id)}
                onKeyDown={(e) => {
                  if (e.key === "Enter") openDrawer(p.id);
                  if (e.key === "c") toggleCompare(p.id);
                }}
                className="absolute left-0 flex w-full cursor-pointer items-center border-b border-rule pr-3 hover:bg-paper-raised"
                style={{ top: vi.start, height: vi.size }}
              >
                <div className="flex w-[30px] shrink-0 justify-center">
                  <input
                    type="checkbox"
                    checked={inCompare}
                    onClick={(e) => e.stopPropagation()}
                    onChange={() => toggleCompare(p.id)}
                    title="Add to compare (max 4)"
                    className="h-3.5 w-3.5 accent-[var(--maroon)]"
                  />
                </div>
                <div className="flex w-[200px] shrink-0 items-center gap-2 overflow-hidden">
                  <TypeBadge type={p.type} />
                  <span className="truncate">
                    <span className="font-medium">{p.name}</span>{" "}
                    <span className="text-[11px] text-ink-3">
                      {p.school_abb}
                      {p.conference ? ` · ${p.conference}` : ""}
                      {p.age != null ? ` · ${p.age}` : ""}
                    </span>
                  </span>
                </div>
                <div className="flex w-[52px] shrink-0 items-center justify-start px-2 text-[12px]">
                  {fmtHeight(p.height_inches)}
                </div>
                <div className="flex w-[36px] shrink-0 items-center justify-center">
                  {p.bmi != null ? (
                    <span
                      className={`inline-block h-2.5 w-2.5 rounded-full ${
                        p.bmi >= 27
                          ? "bg-[var(--grade-elite)]"
                          : p.bmi >= 24
                            ? "bg-[var(--grade-medium)]"
                            : "bg-paper-sunken"
                      }`}
                      title={`BMI ${p.bmi.toFixed(1)}`}
                    />
                  ) : (
                    <span className="text-[9px] text-ink-3">—</span>
                  )}
                </div>
                <div className="w-[62px] shrink-0">
                  <GradeChip grade={p.grade} small />
                </div>
                <div className="w-[64px] shrink-0 px-2 text-right text-[13px] font-semibold">
                  {fmtNum(p.composite, 1)}
                </div>
                <div className="w-[168px] shrink-0 px-2">
                  <BandBar pick={p.proj_pick} band={p.pick_band} width={92} />
                </div>
                <div className="w-[122px] shrink-0 px-2">
                  <ProbCell p={p.mlb_p} raw={p.mlb_p_raw} hist={p.hist_rate} width={52} />
                </div>
                <div className="flex w-[96px] shrink-0 items-center px-2">
                  <ArrivalCell p={p.mlb_arrival ?? null} nn={p.nn_mlb_rate ?? null} />
                </div>
                <div className="flex w-[44px] shrink-0 justify-end px-2">
                  <ConfDot level={p.t1_confidence} />
                </div>
                {statDefs.map((d) => (
                  <div key={d.key} className="w-[64px] shrink-0 px-2 text-right text-[13px]">
                    <span className={p.key_stats[d.key] == null ? "text-ink-3" : ""}>
                      {fmtStat(d, p.key_stats[d.key])}
                    </span>
                  </div>
                ))}
                <div className="min-w-0 flex-1 overflow-hidden pl-2">
                  <FlagChips flags={p.flags} max={1} nowrap />
                </div>
              </div>
            );
          })}
        </div>
        {!display.length && (
          <div className="p-10 text-center text-[13px] text-ink-3">
            No players match these filters.
          </div>
        )}
      </div>
    </div>
  );
}
