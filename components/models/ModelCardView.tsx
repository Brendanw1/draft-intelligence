"use client";

import Link from "next/link";
import { useManifest } from "@/lib/hooks";
import { fmtPct } from "@/lib/format";
import { TypeBadge } from "@/components/shared/Chips";
import {
  BacktestBars,
  ImportanceBars,
  ReliabilityDiagram,
} from "@/components/charts/ModelCharts";

const USE_DONT: Record<
  string,
  { use: string[]; dont: string[] }
> = {
  "fg-draft-hitter": {
    use: [
      "Tiering large pools of D1 hitters into round ranges",
      "Screening transfers / opponents for draft-relevant profiles",
      "Flagging profiles the market will likely value (discipline + power volume)",
    ],
    dont: [
      "Ordering the top of a draft board — top-100 overlap in backtests is poor",
      "Reading the point estimate as a slot: error is ±~110 picks",
      "Comparing across conferences as if production were level-adjusted (it isn't)",
    ],
  },
  "fg-draft-pitcher": {
    use: [
      "Tiering D1 pitchers into round ranges — strikeout volume drives it",
      "Screening for profiles that fit early-round market preferences",
    ],
    dont: [
      "Ordering the top of the board (top-100 overlap ~0 in backtests)",
      "Trusting fg_SHO-driven bumps — shutouts are a sparse, flagged signal",
      "Using it on relievers with < 20 IP",
    ],
  },
  "tier2-predraft-hitter": {
    use: [
      "Ranking hitters by probability of reaching MLB — recall 0.80",
      "Finding late-round profiles whose outcome odds resemble early picks",
      "Reading the calibrated (Platt) number with its historical bin rate",
    ],
    dont: [
      "Reading raw model output — it runs ~2.3× overconfident",
      "Treating a high % as a guarantee: precision is 0.27 because the base rate is 12%",
      "Applying to HS or JUCO players — never in the training population",
    ],
  },
  "tier2-predraft-pitcher": {
    use: [
      "Screening — cross-validated AUC 0.77 supports coarse ranking",
      "The extremes: sub-10% and 80%+ scores were historically honest",
    ],
    dont: [
      "Trusting absolute probabilities between 20–60% — historically ~0–9% actually reached",
      "Evaluating on the single-positive test set — use the CV number",
      "Using on any pitcher with < 20 IP",
    ],
  },
};

export function ModelCardView({ artifactKey }: { artifactKey: string }) {
  const { data: manifest, loading } = useManifest();
  if (loading || !manifest)
    return <div className="p-10 text-[13px] text-ink-3">Loading model card…</div>;
  const m = manifest[artifactKey];
  if (!m)
    return (
      <div className="p-10 text-[13px] text-ink-3">Unknown artifact: {artifactKey}</div>
    );
  const ud = USE_DONT[artifactKey] ?? { use: [], dont: [] };

  return (
    <div className="mx-auto max-w-[900px] px-4 py-6">
      <Link href="/models/" className="text-[12px] text-ink-3 hover:text-ink">
        ← Model Lab
      </Link>
      <div className="mt-2 flex items-center gap-2">
        <TypeBadge type={m.type} />
        <span className="rounded-sm bg-paper-sunken px-1.5 py-px text-[10px] font-semibold uppercase tracking-wide text-ink-3">
          Tier {m.tier}
        </span>
        <span className="text-[11px] text-ink-3">{m.artifact}</span>
      </div>
      <h1
        className="mt-1 text-[26px] font-semibold"
        style={{ fontFamily: "var(--font-fraunces)" }}
      >
        {m.display_name}
      </h1>

      {/* 1 — what it predicts */}
      <section className="mt-5 border-t border-rule pt-4">
        <h2 className="text-[11px] font-semibold uppercase tracking-[0.14em] text-ink-3">
          What it predicts, and for whom
        </h2>
        <div className="mt-2 grid grid-cols-1 gap-x-8 gap-y-2 text-[13px] md:grid-cols-2">
          <div><span className="text-ink-3">Target:</span> {m.target}</div>
          <div><span className="text-ink-3">Algorithm:</span> {m.algorithm}</div>
          <div className="md:col-span-2">
            <span className="text-ink-3">Population:</span> {m.training_population}
          </div>
          <div>
            <span className="text-ink-3">Training n:</span>{" "}
            <span className="font-semibold">{m.n_train.toLocaleString()}</span>
            {m.n_positive != null && (
              <>
                {" "}
                <span className="text-ink-3">
                  ({m.n_positive} reached MLB — {fmtPct(m.base_rate ?? null)} base rate)
                </span>
              </>
            )}
          </div>
          <div>
            <span className="text-ink-3">Features:</span> {m.n_features}
          </div>
        </div>
      </section>

      {/* 2 — how well it works */}
      <section className="mt-5 border-t border-rule pt-4">
        <h2 className="text-[11px] font-semibold uppercase tracking-[0.14em] text-ink-3">
          How well it works
        </h2>
        {m.backtest && m.backtest.length > 0 && (
          <div className="mt-3 flex flex-wrap items-start gap-8">
            <BacktestBars rows={m.backtest} />
            <div className="max-w-[360px] text-[13px] leading-relaxed text-ink-2">
              <p>
                Leave-future-out backtest: train on years before the holdout, score
                the holdout class. The model beats the naive baseline by{" "}
                <span className="font-semibold text-ink">
                  {Math.round(
                    m.backtest.reduce((a, b) => a + (b.baseline_mae - b.mae), 0) /
                      m.backtest.length,
                  )}{" "}
                  picks
                </span>{" "}
                on average, and ordering correlation improved every year as training
                data grew.
              </p>
              <p className="mt-2">
                It is <span className="font-semibold text-ink">poor at naming the exact
                top 100</span> (overlap{" "}
                {m.backtest.map((b) => `${Math.round(b.top100_overlap)}%`).join(", ")}{" "}
                by year). Use it for tiers, not slots.
              </p>
            </div>
          </div>
        )}
        {m.calibration && (
          <div className="mt-3 flex flex-wrap items-start gap-8">
            <ReliabilityDiagram bins={m.calibration.bins} />
            <div className="max-w-[360px] text-[13px] leading-relaxed text-ink-2">
              <p>
                Raw output ran hot: mean prediction{" "}
                <span className="font-semibold text-ink">
                  {fmtPct(m.calibration.mean_pred)}
                </span>{" "}
                vs. an actual rate of{" "}
                <span className="font-semibold text-ink">
                  {fmtPct(m.calibration.mean_actual)}
                </span>{" "}
                (ECE {m.calibration.ece.toFixed(2)}). Every probability on this site
                is Platt-calibrated
                {m.recalibration &&
                  ` — post-calibration the mean drops to ${fmtPct(m.recalibration.mean_platt_val)}, next to the actual ${fmtPct(m.recalibration.mean_actual_val)}`}
                .
              </p>
              {m.notes && <p className="mt-2 font-medium text-flag">{m.notes}</p>}
            </div>
          </div>
        )}
        {!m.backtest?.length && !m.calibration && (
          <p className="mt-2 text-[13px] text-ink-3">No validation artifacts recorded.</p>
        )}
      </section>

      {/* 3 — what it looks at */}
      <section className="mt-5 border-t border-rule pt-4">
        <h2 className="text-[11px] font-semibold uppercase tracking-[0.14em] text-ink-3">
          What it looks at — top {m.importance.length} of {m.n_features} features
        </h2>
        <div className="mt-3 max-w-[560px]">
          <ImportanceBars importance={m.importance} flagged={m.flagged_features ?? []} />
          {(m.flagged_features?.length ?? 0) > 0 && (
            <p className="mt-2 text-[12px] text-flag">
              ⚠ {m.flagged_features!.join(", ")}: rare-event stat flagged for ablation —
              the model may be latching onto a sparse signal.
            </p>
          )}
        </div>
      </section>

      {/* 4 — use / don't use */}
      <section className="mt-5 border-t border-rule pt-4">
        <h2 className="text-[11px] font-semibold uppercase tracking-[0.14em] text-ink-3">
          Use it for / don’t use it for
        </h2>
        <div className="mt-3 grid grid-cols-1 gap-4 md:grid-cols-2">
          <div className="rounded border border-rule bg-paper-raised p-3">
            <div className="mb-1 text-[12px] font-semibold">Use for</div>
            <ul className="list-disc space-y-1 pl-4 text-[13px] text-ink-2">
              {ud.use.map((u) => (
                <li key={u}>{u}</li>
              ))}
            </ul>
          </div>
          <div className="rounded border border-rule bg-paper-raised p-3">
            <div className="mb-1 text-[12px] font-semibold text-flag">Don’t use for</div>
            <ul className="list-disc space-y-1 pl-4 text-[13px] text-ink-2">
              {ud.dont.map((u) => (
                <li key={u}>{u}</li>
              ))}
            </ul>
          </div>
        </div>
      </section>

      <p className="mt-6 border-t border-rule pt-4 text-[13px] italic leading-relaxed text-ink-2">
        This model summarizes public statistical performance. It has never seen a
        player throw. It is a screening lens to point scouting hours, not a
        replacement for them.
      </p>
    </div>
  );
}
