"use client";

import { fmtInt, fmtNum, fmtPct, fmtRate3, NO_DATA } from "@/lib/format";
import { StatDef } from "@/lib/stats";

function fmt(def: StatDef, v: number | null | undefined): string {
  if (v == null) return NO_DATA;
  if (def.key === "Season") return String(Math.round(v));
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

export function SeasonTable({
  seasons,
  cols,
}: {
  seasons: Record<string, number | null>[];
  cols: StatDef[];
}) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-[12px]">
        <thead>
          <tr className="border-b border-rule-strong text-left">
            {cols.map((c) => (
              <th
                key={c.key}
                className="whitespace-nowrap px-1.5 py-1 text-right text-[10px] font-semibold uppercase tracking-wide text-ink-3 first:text-left"
                title={c.help}
              >
                {c.label}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {seasons.map((s, i) => {
            const isCurrent = s.Season === 2026;
            return (
              <tr
                key={i}
                className={`border-b border-rule ${isCurrent ? "font-semibold" : "text-ink-2"}`}
              >
                {cols.map((c) => (
                  <td
                    key={c.key}
                    className="whitespace-nowrap px-1.5 py-1 text-right first:text-left"
                  >
                    {fmt(c, s[c.key])}
                  </td>
                ))}
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
