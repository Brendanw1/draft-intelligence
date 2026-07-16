"use client";

import { Filters } from "@/lib/filters";
import { GRADE_LABEL, GRADE_ORDER } from "@/lib/stats";
import { Grade, IndexPlayer } from "@/lib/types";
import { useMemo } from "react";

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="border-b border-rule px-3 py-3">
      <div className="mb-2 text-[10px] font-semibold uppercase tracking-[0.14em] text-ink-3">
        {title}
      </div>
      {children}
    </div>
  );
}

function Check({
  checked,
  onChange,
  children,
  count,
}: {
  checked: boolean;
  onChange: () => void;
  children: React.ReactNode;
  count?: number;
}) {
  return (
    <label className="flex cursor-pointer items-center gap-2 py-[3px] text-[13px] text-ink-2 hover:text-ink">
      <input
        type="checkbox"
        checked={checked}
        onChange={onChange}
        className="h-3.5 w-3.5 accent-[var(--maroon)]"
      />
      <span className="flex-1">{children}</span>
      {count != null && <span className="text-[11px] text-ink-3">{count.toLocaleString()}</span>}
    </label>
  );
}

export function FilterRail({
  rows,
  filters,
  setFilters,
}: {
  rows: IndexPlayer[];
  filters: Filters;
  setFilters: (f: Filters) => void;
}) {
  const confs = useMemo(() => {
    const m = new Map<string, number>();
    rows.forEach((r) => {
      if (r.conference) m.set(r.conference, (m.get(r.conference) ?? 0) + 1);
    });
    return [...m.entries()].sort((a, b) => b[1] - a[1]);
  }, [rows]);

  const gradeCounts = useMemo(() => {
    const m: Record<string, number> = {};
    rows.forEach((r) => (m[r.grade] = (m[r.grade] ?? 0) + 1));
    return m;
  }, [rows]);

  const toggle = <K extends "conferences" | "grades" | "confidence">(key: K, v: string) => {
    const cur = filters[key] as string[];
    const next = cur.includes(v) ? cur.filter((x) => x !== v) : [...cur, v];
    setFilters({ ...filters, [key]: next });
  };

  return (
    <aside className="thin-scroll w-[212px] shrink-0 overflow-y-auto border-r border-rule bg-paper">
      <Section title="Search">
        <input
          value={filters.q}
          onChange={(e) => setFilters({ ...filters, q: e.target.value })}
          placeholder="Player or school…"
          className="w-full rounded border border-rule bg-paper-raised px-2 py-1.5 text-[13px] placeholder:text-ink-3 focus:border-rule-strong focus:outline-none"
        />
      </Section>

      <Section title="Population">
        <div className="mb-2 flex overflow-hidden rounded border border-rule">
          {(["all", "hitter", "pitcher"] as const).map((t) => (
            <button
              key={t}
              onClick={() => setFilters({ ...filters, type: t })}
              className={`flex-1 py-1 text-[12px] capitalize ${
                filters.type === t
                  ? "bg-maroon-soft font-semibold text-maroon"
                  : "text-ink-2 hover:bg-paper-sunken"
              }`}
            >
              {t === "all" ? "All" : t + "s"}
            </button>
          ))}
        </div>
        <Check
          checked={filters.minSample}
          onChange={() => setFilters({ ...filters, minSample: !filters.minSample })}
        >
          Qualified only{" "}
          <span className="text-[11px] text-ink-3">(50 PA / 20 IP)</span>
        </Check>
      </Section>

      <Section title="Value grade">
        {GRADE_ORDER.map((g: Grade) => (
          <Check
            key={g}
            checked={filters.grades.includes(g)}
            onChange={() => toggle("grades", g)}
            count={gradeCounts[g] ?? 0}
          >
            {GRADE_LABEL[g]}
          </Check>
        ))}
      </Section>

      <Section title="Pick confidence">
        {(["high", "medium", "low"] as const).map((c) => (
          <Check
            key={c}
            checked={filters.confidence.includes(c)}
            onChange={() => toggle("confidence", c)}
          >
            <span className="capitalize">{c}</span>
          </Check>
        ))}
      </Section>

      <Section title="Conference">
        <div className="thin-scroll max-h-[260px] overflow-y-auto pr-1">
          {confs.map(([c, n]) => (
            <Check
              key={c}
              checked={filters.conferences.includes(c)}
              onChange={() => toggle("conferences", c)}
              count={n}
            >
              {c}
            </Check>
          ))}
        </div>
        <div className="mt-1 text-[10px] leading-snug text-ink-3">
          2026 alignment, approximate. Conference is a label here — the models do
          not adjust for it.
        </div>
      </Section>

      <div className="px-3 py-3">
        <button
          onClick={() =>
            setFilters({
              ...filters,
              q: "",
              conferences: [],
              schools: [],
              grades: [],
              confidence: [],
              pickMin: null,
              pickMax: null,
              probMin: null,
              probMax: null,
            })
          }
          className="w-full rounded border border-rule py-1.5 text-[12px] text-ink-2 hover:border-rule-strong hover:text-ink"
        >
          Clear filters
        </button>
      </div>
    </aside>
  );
}
