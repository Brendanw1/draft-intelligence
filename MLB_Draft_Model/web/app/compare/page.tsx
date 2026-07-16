"use client";

import { Suspense, useEffect, useState } from "react";
import { useSearchParams } from "next/navigation";
import { loadDetail } from "@/lib/data";
import { DetailPlayer } from "@/lib/types";
import { fmtNum, fmtPct, fmtRoundBand, NO_DATA } from "@/lib/format";
import {
  HITTER_PCTL_PANEL,
  PITCHER_PCTL_PANEL,
} from "@/lib/stats";
import { FlagChips, GradeChip, TypeBadge } from "@/components/shared/Chips";
import { ordinal } from "@/lib/format";

function CompareInner() {
  const params = useSearchParams();
  const ids = (params.get("ids") ?? "").split(",").filter(Boolean).slice(0, 4);
  const [players, setPlayers] = useState<(DetailPlayer | null)[]>([]);

  useEffect(() => {
    Promise.all(ids.map((id) => loadDetail(id).catch(() => null))).then(setPlayers);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [params]);

  const loaded = players.filter((p): p is DetailPlayer => p != null);
  if (!ids.length)
    return (
      <div className="p-10 text-[13px] text-ink-3">
        Add players to compare from the board (checkbox or press “c” on a row).
      </div>
    );
  if (!loaded.length)
    return <div className="p-10 text-[13px] text-ink-3">Loading players…</div>;

  const allSameType = loaded.every((p) => p.type === loaded[0].type);
  const panel = loaded[0].type === "hitter" ? HITTER_PCTL_PANEL : PITCHER_PCTL_PANEL;

  const Row = ({
    label,
    render,
    help,
  }: {
    label: string;
    render: (p: DetailPlayer) => React.ReactNode;
    help?: string;
  }) => (
    <tr className="border-b border-rule">
      <td
        className="w-[150px] whitespace-nowrap px-3 py-2 text-[11px] font-semibold uppercase tracking-wide text-ink-3"
        title={help}
      >
        {label}
      </td>
      {loaded.map((p) => (
        <td key={p.id} className="px-3 py-2 text-[13px]">
          {render(p)}
        </td>
      ))}
    </tr>
  );

  return (
    <div className="mx-auto max-w-[1100px] px-4 py-6">
      <h1
        className="mb-4 text-[24px] font-semibold"
        style={{ fontFamily: "var(--font-fraunces)" }}
      >
        Side-by-side
      </h1>
      <div className="overflow-x-auto rounded border border-rule">
        <table className="w-full">
          <thead>
            <tr className="border-b border-rule-strong bg-paper-sunken">
              <th className="px-3 py-2" />
              {loaded.map((p) => (
                <th key={p.id} className="min-w-[190px] px-3 py-2 text-left">
                  <div className="flex items-center gap-2">
                    <TypeBadge type={p.type} />
                    <span
                      className="text-[16px] font-semibold"
                      style={{ fontFamily: "var(--font-fraunces)" }}
                    >
                      {p.name}
                    </span>
                  </div>
                  <div className="mt-0.5 text-[11px] font-normal text-ink-3">
                    {p.school_abb}
                    {p.conference ? ` · ${p.conference}` : ""}
                    {p.age != null ? ` · age ${p.age}` : ""}
                  </div>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            <Row label="Grade" render={(p) => <GradeChip grade={p.grade} small />} />
            <Row
              label="Composite"
              help="40% projected draft slot · 60% calibrated MLB probability"
              render={(p) => <span className="font-semibold">{fmtNum(p.composite, 1)}</span>}
            />
            <Row
              label="Projected round"
              help="Pick ± backtest error, as a round range"
              render={(p) => fmtRoundBand(p.pick_band)}
            />
            <Row
              label="MLB% (calibrated)"
              render={(p) => <span className="font-semibold">{fmtPct(p.mlb_p)}</span>}
            />
            <Row
              label="Historical rate"
              help="How often players with this raw score actually reached MLB"
              render={(p) => (p.hist_rate != null ? `~${fmtPct(p.hist_rate)}` : NO_DATA)}
            />
            <Row
              label="Sample"
              render={(p) =>
                p.type === "hitter"
                  ? p.sample.pa != null
                    ? `${Math.round(p.sample.pa)} PA`
                    : NO_DATA
                  : p.sample.ip != null
                    ? `${p.sample.ip.toFixed(1)} IP`
                    : NO_DATA
              }
            />
            {allSameType &&
              panel.map((d) => (
                <Row
                  key={d.stat}
                  label={`${d.label} pctl`}
                  help={d.why}
                  render={(p) => {
                    const v = p.pctl[d.stat];
                    if (v == null) return NO_DATA;
                    return (
                      <span className="inline-flex items-center gap-2">
                        <span className="inline-block h-[7px] w-[72px] overflow-hidden rounded-sm bg-paper-sunken align-middle">
                          <span
                            className="block h-full rounded-sm"
                            style={{
                              width: `${v}%`,
                              background:
                                v >= 70
                                  ? "var(--grade-elite)"
                                  : v <= 30
                                    ? "var(--grade-low)"
                                    : "var(--grade-medium)",
                            }}
                          />
                        </span>
                        {ordinal(v)}
                      </span>
                    );
                  }}
                />
              ))}
            <Row
              label="Comps reached MLB"
              render={(p) =>
                p.comps.length
                  ? `${p.comps.filter((c) => c.reached_mlb).length} of ${p.comps.length}`
                  : NO_DATA
              }
            />
            <Row label="Flags" render={(p) => <FlagChips flags={p.flags} max={4} />} />
          </tbody>
        </table>
      </div>
      {!allSameType && (
        <p className="mt-3 text-[12px] text-ink-3">
          Mixed hitters and pitchers — percentile rows hidden because the
          populations aren’t comparable.
        </p>
      )}
    </div>
  );
}

export default function ComparePage() {
  return (
    <Suspense>
      <CompareInner />
    </Suspense>
  );
}
