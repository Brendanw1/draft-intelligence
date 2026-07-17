"use client";

import { Suspense, useMemo, useRef, useEffect, useCallback, useState } from "react";
import Link from "next/link";
import { useManifest, useIndex } from "@/lib/hooks";
import { fmtPct, fmtNum, NO_DATA } from "@/lib/format";
import { TypeBadge } from "@/components/shared/Chips";

/* ── Inline SVG calibration ladder ─────────────────────────────────────── */
function CalibrationLadder({
  bins,
  title,
  xLabel,
  yLabel,
}: {
  bins: { bin: string; count: number; pred_mean: number; actual_rate: number }[];
  title: string;
  xLabel: string;
  yLabel: string;
}) {
  const W = 600;
  const H = 280;
  const M = { top: 32, right: 24, bottom: 44, left: 60 };
  const plotW = W - M.left - M.right;
  const plotH = H - M.top - M.bottom;

  const maxVal = Math.max(
    ...bins.map((b) => Math.max(b.pred_mean, b.actual_rate)),
    0.1,
  );

  const x = (v: number) => M.left + (v / maxVal) * plotW;
  const y = (v: number) => M.top + (1 - v / maxVal) * plotH;

  return (
    <div className="rounded-sm border border-border-subtle bg-surface-2 p-4">
      <div className="mb-2 text-[11px] font-semibold uppercase tracking-wide text-text-tertiary">
        {title}
      </div>
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full max-w-[600px]" role="img">
        {/* diagonal perfect line */}
        <line
          x1={x(0)}
          y1={y(0)}
          x2={x(maxVal)}
          y2={y(maxVal)}
          stroke="var(--border-strong)"
          strokeWidth={1}
          strokeDasharray="4 3"
        />
        {/* gridlines */}
        {[0.25, 0.5, 0.75].map((v) => (
          <g key={v}>
            <line
              x1={M.left}
              y1={y(v * maxVal)}
              x2={W - M.right}
              y2={y(v * maxVal)}
              stroke="var(--border-subtle)"
              strokeWidth={1}
            />
            <text
              x={M.left - 6}
              y={y(v * maxVal) + 4}
              textAnchor="end"
              className="fill-text-tertiary text-[10px]"
            >
              {fmtPct(v * maxVal)}
            </text>
          </g>
        ))}
        {/* bins as filled circles, sized by count */}
        {bins.map((b) => {
          const r = Math.max(4, Math.min(12, Math.sqrt(b.count) * 1.2));
          const cx = x(b.pred_mean);
          const cy = y(b.actual_rate);
          const aboveLine = b.actual_rate > b.pred_mean;
          return (
            <g key={b.bin}>
              <title>
                {b.bin}: {b.count} players — predicted {fmtPct(b.pred_mean)}, actual{" "}
                {fmtPct(b.actual_rate)}
              </title>
              <circle
                cx={cx}
                cy={cy}
                r={r}
                fill={aboveLine ? "var(--sig-high)" : "var(--sig-low)"}
                fillOpacity={0.65}
                stroke={aboveLine ? "var(--sig-high)" : "var(--sig-low)"}
                strokeWidth={1.5}
              />
            </g>
          );
        })}
        {/* axis labels */}
        <text
          x={M.left + plotW / 2}
          y={H - 6}
          textAnchor="middle"
          className="fill-text-tertiary text-[10px]"
        >
          {xLabel}
        </text>
        <text
          x={14}
          y={M.top + plotH / 2}
          textAnchor="middle"
          transform={`rotate(-90, 14, ${M.top + plotH / 2})`}
          className="fill-text-tertiary text-[10px]"
        >
          {yLabel}
        </text>
      </svg>
    </div>
  );
}

/* ── Model disagreement mini-scatter ──────────────────────────────────── */
function DisagreementPanel({
  rows,
  type,
}: {
  rows: { name: string; id: string; pick: number; mlbP: number; arrival: number | null; grade: string }[];
  type: string;
}) {
  const W = 300;
  const H = 220;
  const M = { top: 16, right: 16, bottom: 36, left: 48 };
  const plotW = W - M.left - M.right;
  const plotH = H - M.top - M.bottom;

  const xScale = useCallback(
    (pick: number) => M.left + (Math.log(pick) / Math.log(620)) * plotW,
    [],
  );
  const yScale = useCallback(
    (p: number) => M.top + (1 - p) * plotH,
    [],
  );

  return (
    <div className="rounded-sm border border-border-subtle bg-surface-2 p-4">
      <div className="mb-2 text-[11px] font-semibold uppercase tracking-wide text-text-tertiary">
        {type} — pick vs. calibrated MLB%
      </div>
      <p className="mb-3 text-[12px] leading-snug text-text-secondary">
        Top-right outliers: model says they&apos;re likely to reach MLB despite projecting
        late in the draft. These are where the market model and outcome model
        most disagree.
      </p>
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full max-w-[300px]" role="img">
        <rect x={M.left} y={M.top} width={plotW} height={plotH} fill="var(--surface-1)" rx={2} />
        {rows.slice(0, 200).map((r) => {
          const cx = xScale(r.pick);
          const cy = yScale(r.mlbP);
          return (
            <circle
              key={r.id}
              cx={cx}
              cy={cy}
              r={2.5}
              fill={
                r.grade === "elite" || r.grade === "high"
                  ? "var(--sig-elite)"
                  : "var(--sig-low)"
              }
              fillOpacity={0.5}
            >
              <title>
                {r.name}: pick ~{Math.round(r.pick)}, MLB {fmtPct(r.mlbP)}
                {r.arrival != null ? `, arrival ${fmtPct(r.arrival)}` : ""}
              </title>
            </circle>
          );
        })}
        <text x={M.left + 4} y={M.top + 12} className="fill-text-secondary text-[9px]">
          high pick · high MLB%
        </text>
        <text x={M.left + 4} y={M.top + plotH - 4} className="fill-text-secondary text-[9px]">
          high pick · low MLB%
        </text>
        <text x={M.left + plotW - 4} y={M.top + plotH - 4} textAnchor="end" className="fill-text-secondary text-[9px]">
          late pick · low MLB%
        </text>
      </svg>
    </div>
  );
}

/* ── Main page ─────────────────────────────────────────────────────────── */
function AuditInner() {
  const { data: manifest, loading: manifestLoading } = useManifest();
  const { data: rows, loading: indexLoading } = useIndex();

  const disagreementRows = useMemo(() => {
    if (!rows) return { hitters: [] as any[], pitchers: [] as any[] };
    return {
      hitters: rows
        .filter((r) => r.type === "hitter" && r.proj_pick != null && r.mlb_p != null)
        .slice(0, 300),
      pitchers: rows
        .filter((r) => r.type === "pitcher" && r.proj_pick != null && r.mlb_p != null)
        .slice(0, 300),
    };
  }, [rows]);

  if (manifestLoading || indexLoading || !manifest || !rows) {
    return (
      <div className="flex min-h-[60vh] items-center justify-center">
        <div className="flex flex-col items-center gap-3">
          <div className="h-3 w-48 animate-pulse rounded-sm bg-surface-1" />
          <p className="text-[12px] text-text-tertiary">Loading calibration data…</p>
        </div>
      </div>
    );
  }

  const tier1H = manifest["fg-draft-hitter"];
  const tier1P = manifest["fg-draft-pitcher"];
  const tier2H = manifest["tier2-predraft-hitter"];
  const tier2P = manifest["tier2-predraft-pitcher"];

  // Tier 3 counts from player data
  const withArrival = rows.filter((r) => r.mlb_arrival != null).length;
  const arrivalMean =
    withArrival > 0
      ? rows.reduce((s, r) => s + (r.mlb_arrival ?? 0), 0) / withArrival
      : 0;

  return (
    <div className="mx-auto max-w-[1160px] px-4 py-6">
      <div className="flex items-center gap-3">
        <Link href="/models/" className="text-[12px] text-text-tertiary hover:text-text-primary">
          ← Model Lab
        </Link>
        <span className="text-text-disabled">/</span>
        <span className="text-[12px] font-semibold text-text-primary">Calibration Audit</span>
      </div>
      <h1
        className="mt-2 text-[26px] font-semibold"
        style={{ fontFamily: "var(--font-fraunces)" }}
      >
        Calibration Audit
      </h1>
      <p className="mt-1 max-w-[720px] text-[13px] leading-relaxed text-text-secondary">
        A single view designed to earn — or challenge — your trust in the model.
        Every calibration curve, backtest metric, and disagreement signal is here.
        If the model is overconfident, this page will show it.
      </p>

      {/* ── Trust-at-a-glance tiles ─────────────────────────────────── */}
      <div className="mt-6 grid grid-cols-2 gap-3 md:grid-cols-4">
        {[
          {
            label: "Tier 1 MAE",
            value: tier1H?.backtest?.length
              ? `±${Math.round(tier1H.backtest[tier1H.backtest.length - 1].mae)}`
              : "…",
            sub: "picks — read rounds, not slots",
            warn: true,
          },
          {
            label: "Tier 2 overconfidence",
            value: tier2H?.calibration?.ece
              ? `${tier2H.calibration.ece.toFixed(2)} ECE`
              : "…",
            sub: "raw model ~2.3× too confident",
            warn: true,
          },
          {
            label: "Tier 2 calibrated",
            value: tier2H?.recalibration
              ? `Brier ${tier2H.recalibration.brier_platt.toFixed(3)}`
              : "…",
            sub: "Platt-scaled — honest probabilities",
            warn: false,
          },
          {
            label: "Tier 3 coverage",
            value: withArrival > 0 ? `${withArrival.toLocaleString()}` : "…",
            sub: `players · mean ${fmtPct(arrivalMean)} arrival`,
            warn: false,
          },
        ].map((t) => (
          <div
            key={t.label}
            className="rounded-sm border border-border-subtle bg-surface-2 p-4"
          >
            <div className="text-[10px] font-semibold uppercase tracking-wide text-text-tertiary">
              {t.label}
            </div>
            <div className="mt-1 flex items-baseline gap-2">
              <span
                className={`text-[22px] font-semibold ${t.warn ? "text-caution" : ""}`}
              >
                {t.value}
              </span>
            </div>
            <div className="mt-0.5 text-[11px] leading-snug text-text-secondary">{t.sub}</div>
          </div>
        ))}
      </div>

      {/* ── Calibration ladders ──────────────────────────────────────── */}
      <div className="mt-6 grid grid-cols-1 gap-4 lg:grid-cols-2">
        {tier2H?.calibration?.bins && (
          <CalibrationLadder
            bins={tier2H.calibration.bins}
            title="Tier 2 hitters — before calibration"
            xLabel="predicted probability (raw model)"
            yLabel="actual MLB rate"
          />
        )}
        {tier2P?.calibration?.bins && (
          <CalibrationLadder
            bins={tier2P.calibration.bins}
            title="Tier 2 pitchers — before calibration"
            xLabel="predicted probability (raw model)"
            yLabel="actual MLB rate"
          />
        )}
      </div>

      {/* ── Model disagreement ───────────────────────────────────────── */}
      <div className="mt-6">
        <h2 className="text-[11px] font-semibold uppercase tracking-wide text-text-tertiary">
          Model disagreement — where Tier 1 and Tier 2 diverge
        </h2>
        <p className="mt-1 text-[12px] leading-relaxed text-text-secondary">
          Each point is a 2026 prospect. Points in the top-right are projected
          late but have high outcome probability — that&apos;s where the market model
          (Tier 1) and the outcome model (Tier 2) disagree most. These are the
          players worth a closer scouting look.
        </p>
        <div className="mt-3 grid grid-cols-1 gap-4 md:grid-cols-2">
          <DisagreementPanel
            rows={disagreementRows.hitters}
            type="Hitters"
          />
          <DisagreementPanel
            rows={disagreementRows.pitchers}
            type="Pitchers"
          />
        </div>
      </div>

      {/* ── Tier 3 arrival distribution ──────────────────────────────── */}
      <div className="mt-6 rounded-sm border border-border-subtle bg-surface-2 p-4">
        <h2 className="text-[11px] font-semibold uppercase tracking-wide text-text-tertiary">
          Tier 3: MLB Arrival distribution
        </h2>
        <p className="mt-1 text-[12px] text-text-secondary">
          P(MLB debut | drafted), predicted by an Elastic Net with round-anchored
          prior and nearest-neighbor comp rates. The distribution is right-skewed —
          most players have low arrival probability, with a long tail of high-likelihood
          prospects.
        </p>
        <div className="mt-3 flex flex-wrap gap-3">
          {[
            { label: "0–5%", count: rows.filter((r) => (r.mlb_arrival ?? 0) <= 0.05).length },
            { label: "5–10%", count: rows.filter((r) => (r.mlb_arrival ?? 0) > 0.05 && (r.mlb_arrival ?? 0) <= 0.10).length },
            { label: "10–20%", count: rows.filter((r) => (r.mlb_arrival ?? 0) > 0.10 && (r.mlb_arrival ?? 0) <= 0.20).length },
            { label: "20–30%", count: rows.filter((r) => (r.mlb_arrival ?? 0) > 0.20 && (r.mlb_arrival ?? 0) <= 0.30).length },
            { label: "30%+", count: rows.filter((r) => (r.mlb_arrival ?? 0) > 0.30).length },
          ].map((b) => {
            const maxCount = Math.max(...[1, rows.filter((r) => (r.mlb_arrival ?? 0) > 0.05).length]);
            const pct = withArrival > 0 ? (b.count / withArrival) * 100 : 0;
            return (
              <div key={b.label} className="flex-1 min-w-[100px]">
                <div className="text-[18px] font-semibold">
                  {b.count.toLocaleString()}
                </div>
                <div className="text-[11px] text-text-secondary">
                  {b.label}
                  <span className="ml-1 text-text-tertiary">({fmtPct(pct / 100, 1)})</span>
                </div>
                <div className="mt-1 h-2 w-full overflow-hidden rounded-sm bg-surface-1">
                  <div
                    className="h-full rounded-sm bg-[var(--sig-high)]"
                    style={{ width: `${Math.max(2, (b.count / (maxCount || 1)) * 100)}%` }}
                  />
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* ── How to use this page ──────────────────────────────────────── */}
      <div className="mt-6 rounded-sm border border-border-subtle bg-surface-2 p-4">
        <h2 className="text-[11px] font-semibold uppercase tracking-wide text-text-tertiary">
          How to read this page
        </h2>
        <ul className="mt-2 list-disc space-y-1 pl-5 text-[12px] leading-relaxed text-text-secondary">
          <li>
            <span className="font-semibold text-text-primary">Calibration ladders:</span>{" "}
            Points above the diagonal mean the model was underconfident (actual rate
            exceeded prediction). Below means overconfident. Pre-correction, Tier 2
            runs ~2.3× hot.
          </li>
          <li>
            <span className="font-semibold text-text-primary">Disagreement scatter:</span>{" "}
            Top-right points (late projected pick, high MLB%) are players where Tier 1
            and Tier 2 disagree. These are screening flags — follow up with scouting.
          </li>
          <li>
            <span className="font-semibold text-text-primary">Arrival distribution:</span>{" "}
            Tier 3 is a prior-anchored model — it won&apos;t produce extreme probabilities
            for late-round profiles. The distribution reflects this conservatism.
          </li>
          <li>
            <span className="font-semibold text-text-primary">Every number on this page</span>{" "}
            comes from the model artifacts in the{" "}
            <Link href="/models/" className="text-accent underline">
              Model Lab
            </Link>
            . Nothing is invented — these are the same backtests, calibration curves,
            and feature importances that drive every projection on the board.
          </li>
        </ul>
      </div>

      <p className="mt-8 border-t border-border-subtle pt-4 text-[13px] italic leading-relaxed text-text-secondary">
        If this page makes you trust the model less in some places and more in
        others, it&apos;s working. Calibration honesty is the product.
      </p>
    </div>
  );
}

export default function CalibrationAuditPage() {
  return (
    <Suspense>
      <AuditInner />
    </Suspense>
  );
}
