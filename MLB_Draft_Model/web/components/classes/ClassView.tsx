"use client";

import Link from "next/link";
import { useMemo, useState } from "react";
import { useClassYear } from "@/lib/hooks";
import { ClassRow } from "@/lib/types";
import { fmtBonus, fmtPct, fmtRate3, fmtNum, NO_DATA } from "@/lib/format";
import { TypeBadge } from "@/components/shared/Chips";

const YEARS = [2021, 2022, 2023, 2024, 2025, 2026];

/** MLB rate by round band — bars with counts, from actual outcomes. */
function RateByRound({ rows }: { rows: ClassRow[] }) {
  const bands: { label: string; lo: number; hi: number }[] = [
    { label: "R1–2", lo: 1, hi: 2 },
    { label: "R3–5", lo: 3, hi: 5 },
    { label: "R6–10", lo: 6, hi: 10 },
    { label: "R11–20", lo: 11, hi: 20 },
  ];
  const data = bands.map((b) => {
    const members = rows.filter(
      (r) => r.round != null && r.round >= b.lo && r.round <= b.hi && r.reached_mlb != null,
    );
    const reached = members.filter((r) => r.reached_mlb).length;
    return { ...b, n: members.length, rate: members.length ? reached / members.length : null };
  });
  const max = Math.max(0.01, ...data.map((d) => d.rate ?? 0));
  return (
    <div className="space-y-1.5">
      {data.map((d) => (
        <div key={d.label} className="flex items-center gap-2 text-[12px]">
          <span className="w-[52px] shrink-0 text-ink-2">{d.label}</span>
          <div className="h-[10px] flex-1 overflow-hidden rounded-sm bg-paper-sunken">
            {d.rate != null && (
              <div
                className="h-full rounded-sm"
                style={{ width: `${(d.rate / max) * 100}%`, background: "var(--band-strong)" }}
                title={`${d.label}: ${fmtPct(d.rate)} of ${d.n} tracked players reached MLB`}
              />
            )}
          </div>
          <span className="w-[86px] shrink-0 text-right">
            {d.rate != null ? (
              <>
                <span className="font-semibold">{fmtPct(d.rate)}</span>{" "}
                <span className="text-ink-3">of {d.n}</span>
              </>
            ) : (
              <span className="text-ink-3">{NO_DATA}</span>
            )}
          </span>
        </div>
      ))}
    </div>
  );
}

export function ClassView({ year }: { year: number }) {
  const { data: rows, loading } = useClassYear(year);
  const [type, setType] = useState<"all" | "hitter" | "pitcher">("all");
  const [onlyMlb, setOnlyMlb] = useState(false);

  const filtered = useMemo(() => {
    if (!rows) return [];
    return rows.filter(
      (r) => (type === "all" || r.type === type) && (!onlyMlb || r.reached_mlb),
    );
  }, [rows, type, onlyMlb]);

  const tracked = rows?.filter((r) => r.reached_mlb != null) ?? [];
  const reached = tracked.filter((r) => r.reached_mlb).length;
  const recent = year >= 2025;

  return (
    <div className="mx-auto max-w-[1100px] px-4 py-6">
      <div className="flex flex-wrap items-center gap-2">
        {YEARS.map((y) => (
          <Link
            key={y}
            href={`/classes/${y}/`}
            className={`rounded px-2.5 py-1 text-[13px] ${
              y === year
                ? "bg-maroon-soft font-semibold text-maroon"
                : "text-ink-2 hover:text-ink"
            }`}
          >
            {y}
          </Link>
        ))}
      </div>
      <h1
        className="mt-3 text-[26px] font-semibold"
        style={{ fontFamily: "var(--font-fraunces)" }}
      >
        {year} draft class — college players with FanGraphs seasons
      </h1>

      {loading || !rows ? (
        <div className="p-10 text-[13px] text-ink-3">Loading class…</div>
      ) : (
        <>
          <div className="mt-4 flex flex-wrap items-start gap-8">
            <div className="min-w-[280px] flex-1">
              <div className="mb-2 text-[11px] font-semibold uppercase tracking-[0.14em] text-ink-3">
                Reached MLB by round {recent && "· too recent to judge"}
              </div>
              <RateByRound rows={rows} />
              <div className="mt-2 text-[11px] leading-snug text-ink-3">
                {tracked.length
                  ? `${reached} of ${tracked.length} tracked signees have debuted (${fmtPct(reached / Math.max(tracked.length, 1))}). `
                  : "No MiLB outcome tracking joined for this class. "}
                Draft position remains the strongest single predictor — which is
                exactly why the value map hunts for exceptions.
              </div>
            </div>
            <div className="flex items-center gap-3">
              <div className="flex overflow-hidden rounded border border-rule">
                {(["all", "hitter", "pitcher"] as const).map((t) => (
                  <button
                    key={t}
                    onClick={() => setType(t)}
                    className={`px-3 py-1 text-[12px] capitalize ${
                      type === t
                        ? "bg-maroon-soft font-semibold text-maroon"
                        : "text-ink-2 hover:bg-paper-sunken"
                    }`}
                  >
                    {t === "all" ? "All" : t + "s"}
                  </button>
                ))}
              </div>
              <label className="flex cursor-pointer items-center gap-1.5 text-[12px] text-ink-2">
                <input
                  type="checkbox"
                  checked={onlyMlb}
                  onChange={() => setOnlyMlb(!onlyMlb)}
                  className="h-3.5 w-3.5 accent-[var(--maroon)]"
                />
                reached MLB only
              </label>
            </div>
          </div>

          <div className="mt-4 overflow-x-auto rounded border border-rule">
            <table className="w-full text-[12px]">
              <thead>
                <tr className="border-b border-rule-strong bg-paper-sunken text-left">
                  {["", "Player", "School", "Pos", "Pick", "Rd", "Bonus", "Key stats", "Outcome"].map(
                    (h, i) => (
                      <th
                        key={i}
                        className="whitespace-nowrap px-2.5 py-1.5 text-[10px] font-semibold uppercase tracking-wide text-ink-3"
                      >
                        {h}
                      </th>
                    ),
                  )}
                </tr>
              </thead>
              <tbody>
                {filtered.map((r, i) => (
                  <tr key={i} className="border-b border-rule hover:bg-paper-raised">
                    <td className="px-2.5 py-1.5">
                      <TypeBadge type={r.type} />
                    </td>
                    <td className="whitespace-nowrap px-2.5 py-1.5 font-medium">
                      {r.name ?? NO_DATA}
                    </td>
                    <td className="max-w-[180px] truncate px-2.5 py-1.5 text-ink-2">
                      {r.school ?? NO_DATA}
                    </td>
                    <td className="px-2.5 py-1.5 text-ink-2">{r.position ?? NO_DATA}</td>
                    <td className="px-2.5 py-1.5 text-right">{r.pick ?? NO_DATA}</td>
                    <td className="px-2.5 py-1.5 text-right">
                      {r.round === 0 ? "comp" : (r.round ?? NO_DATA)}
                    </td>
                    <td className="px-2.5 py-1.5 text-right">{fmtBonus(r.bonus)}</td>
                    <td className="whitespace-nowrap px-2.5 py-1.5 text-ink-2">
                      {r.type === "hitter" ? (
                        <>
                          {fmtRate3(r.stats.wOBA)} wOBA · {fmtNum(r.stats.wRCplus, 0)} wRC+ ·{" "}
                          {fmtNum((r.stats.hr as number) ?? null, 0)} HR
                        </>
                      ) : (
                        <>
                          {fmtNum(r.stats.era, 2)} ERA · {fmtNum(r.stats.fip, 2)} FIP ·{" "}
                          {fmtNum(r.stats.ip, 0)} IP
                        </>
                      )}
                    </td>
                    <td className="whitespace-nowrap px-2.5 py-1.5">
                      {r.reached_mlb == null ? (
                        <span className="text-ink-3">not tracked</span>
                      ) : r.reached_mlb ? (
                        <span className="font-semibold text-[var(--grade-elite)]">MLB</span>
                      ) : (
                        <span className="text-ink-2">
                          {r.peak_level ? `peak ${r.peak_level}` : "signed"}
                        </span>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
            {!filtered.length && (
              <div className="p-8 text-center text-[13px] text-ink-3">
                No players match.
              </div>
            )}
          </div>
        </>
      )}
    </div>
  );
}
