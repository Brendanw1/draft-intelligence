"use client";

import { Suspense, useMemo, useState } from "react";
import Link from "next/link";
import { ValueScatter } from "@/components/charts/ValueScatter";
import { useIndex } from "@/lib/hooks";
import { applyFilters, DEFAULT_FILTERS } from "@/lib/filters";

type Brush = { pickMin: number; pickMax: number; probMin: number; probMax: number } | null;

function ValueInner() {
  const { data: rows, loading } = useIndex();
  const [type, setType] = useState<"all" | "hitter" | "pitcher">("all");
  const [brush, setBrush] = useState<Brush>(null);

  const filtered = useMemo(() => {
    if (!rows) return [];
    return applyFilters(rows, { ...DEFAULT_FILTERS, type, minSample: true });
  }, [rows, type]);

  const brushed = useMemo(() => {
    if (!brush) return [];
    return filtered.filter(
      (r) =>
        r.proj_pick != null &&
        r.mlb_p != null &&
        r.proj_pick >= brush.pickMin &&
        r.proj_pick <= brush.pickMax &&
        r.mlb_p >= brush.probMin &&
        r.mlb_p <= brush.probMax,
    );
  }, [filtered, brush]);

  const boardHref = brush
    ? `/board/?${new URLSearchParams({
        ...(type !== "all" ? { type } : {}),
        pick_min: String(brush.pickMin),
        pick_max: String(brush.pickMax),
        p_min: String(brush.probMin),
        p_max: String(brush.probMax),
      }).toString()}`
    : "/board/";

  return (
    <div className="mx-auto max-w-[1200px] px-4 py-5">
      <div className="mb-3 flex flex-wrap items-center gap-4">
        <div>
          <h1
            className="text-[24px] font-semibold leading-tight"
            style={{ fontFamily: "var(--font-fraunces)" }}
          >
            Value Map
          </h1>
          <p className="text-[12px] text-ink-2">
            Where the market model and the outcome model disagree. Qualified
            players only (50 PA / 20 IP). Drag to select a region; click a point
            for the dossier.
          </p>
        </div>
        <div className="ml-auto flex items-center gap-3">
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
          <div className="flex items-center gap-3 text-[11px] text-ink-2">
            <span className="inline-flex items-center gap-1">
              <span className="inline-block h-2.5 w-2.5 rounded-full" style={{ background: "var(--hitter)" }} />
              hitter ●
            </span>
            <span className="inline-flex items-center gap-1">
              <span className="inline-block h-2.5 w-2.5" style={{ background: "var(--pitcher)" }} />
              pitcher ■
            </span>
            <span className="inline-flex items-center gap-1">
              <span className="inline-block h-2.5 w-2.5 rounded-full" style={{ background: "var(--grade-elite)" }} />
              elite
            </span>
            <span className="inline-flex items-center gap-1">
              <span className="inline-block h-2.5 w-2.5 rounded-full" style={{ background: "var(--grade-high)" }} />
              high
            </span>
            <span className="inline-flex items-center gap-1">
              <span className="inline-block h-2.5 w-2.5 rounded-full opacity-50" style={{ background: "var(--grade-medium)" }} />
              med/low muted
            </span>
          </div>
        </div>
      </div>

      {loading ? (
        <div className="p-10 text-[13px] text-ink-3">Loading projections…</div>
      ) : (
        <div className="rounded border border-rule bg-paper-raised">
          <ValueScatter
            rows={filtered}
            showPitcherCaution={type !== "hitter"}
            onBrush={setBrush}
          />
        </div>
      )}

      {brush && (
        <div className="mt-3 flex items-center gap-3 rounded border border-rule-strong bg-paper-raised px-3 py-2">
          <span className="text-[13px]">
            <span className="font-semibold">{brushed.length.toLocaleString()}</span>{" "}
            players in selection — picks {brush.pickMin}–{brush.pickMax}, MLB%{" "}
            {(brush.probMin * 100).toFixed(0)}–{(brush.probMax * 100).toFixed(0)}%
          </span>
          <Link
            href={boardHref}
            className="rounded bg-maroon px-3 py-1 text-[12px] font-semibold text-white"
          >
            Open these on the board →
          </Link>
          <button
            onClick={() => setBrush(null)}
            className="text-[11px] text-ink-3 hover:text-ink"
          >
            clear selection
          </button>
        </div>
      )}
    </div>
  );
}

export default function ValuePage() {
  return (
    <Suspense>
      <ValueInner />
    </Suspense>
  );
}
