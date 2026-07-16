"use client";

import Link from "next/link";
import { useManifest } from "@/lib/hooks";
import { fmtPct } from "@/lib/format";
import { TypeBadge } from "@/components/shared/Chips";

const ORDER = [
  "fg-draft-hitter",
  "fg-draft-pitcher",
  "tier2-predraft-hitter",
  "tier2-predraft-pitcher",
];

export default function ModelsPage() {
  const { data: manifest, loading } = useManifest();

  return (
    <div className="mx-auto max-w-[1000px] px-4 py-6">
      <h1
        className="text-[26px] font-semibold"
        style={{ fontFamily: "var(--font-fraunces)" }}
      >
        Model Lab
      </h1>
      <p className="mt-1 max-w-[640px] text-[13px] leading-relaxed text-ink-2">
        Four production artifacts. Tier 1 predicts where the industry drafts a
        stat profile; Tier 2 predicts whether the profile actually reaches MLB.
        Every number on this site traces back to one of these cards.
      </p>

      {/* pipeline diagram */}
      <div className="mt-5 overflow-x-auto rounded border border-rule bg-paper-raised p-4">
        <div className="flex min-w-[720px] items-center gap-2 text-[12px]">
          {[
            ["FanGraphs D1 stats", "2021–2026, joined via xMLBAMID"],
            ["Tier 1 — draft position", "XGBoost regression → projected pick"],
            ["Tier 2 — MLB outcome", "pre-draft classifier, fed the projected pick"],
            ["Calibration", "Platt / isotonic — fixes 2.3× overconfidence"],
            ["2026 projections", "10,734 players scored"],
          ].map(([title, sub], i, arr) => (
            <div key={title} className="flex items-center gap-2">
              <div className="rounded border border-rule bg-paper px-3 py-2">
                <div className="font-semibold">{title}</div>
                <div className="text-[10px] text-ink-3">{sub}</div>
              </div>
              {i < arr.length - 1 && <span className="text-ink-3">→</span>}
            </div>
          ))}
        </div>
      </div>

      {loading || !manifest ? (
        <div className="p-10 text-[13px] text-ink-3">Loading manifest…</div>
      ) : (
        <div className="mt-5 grid grid-cols-1 gap-4 md:grid-cols-2">
          {ORDER.map((key) => {
            const m = manifest[key];
            if (!m) return null;
            const latestBt = m.backtest?.length
              ? m.backtest[m.backtest.length - 1]
              : null;
            return (
              <Link
                key={key}
                href={`/models/${key}/`}
                className="group rounded border border-rule bg-paper-raised p-4 hover:border-rule-strong"
              >
                <div className="flex items-center gap-2">
                  <TypeBadge type={m.type} />
                  <span className="rounded-sm bg-paper-sunken px-1.5 py-px text-[10px] font-semibold uppercase tracking-wide text-ink-3">
                    Tier {m.tier}
                  </span>
                </div>
                <div
                  className="mt-2 text-[17px] font-semibold group-hover:text-maroon"
                  style={{ fontFamily: "var(--font-fraunces)" }}
                >
                  {m.display_name}
                </div>
                <div className="mt-1 text-[12px] leading-snug text-ink-2">{m.target}</div>
                <div className="mt-3 flex flex-wrap gap-x-5 gap-y-1 text-[12px]">
                  <span>
                    <span className="text-ink-3">n =</span>{" "}
                    <span className="font-semibold">{m.n_train.toLocaleString()}</span>
                  </span>
                  {latestBt && (
                    <span>
                      <span className="text-ink-3">MAE</span>{" "}
                      <span className="font-semibold">±{Math.round(latestBt.mae)}</span>{" "}
                      <span className="text-ink-3">vs {Math.round(latestBt.baseline_mae)} naive</span>
                    </span>
                  )}
                  {m.calibration && (
                    <span>
                      <span className="text-ink-3">ECE</span>{" "}
                      <span className="font-semibold text-flag">
                        {m.calibration.ece.toFixed(2)}
                      </span>{" "}
                      <span className="text-ink-3">pre-calibration</span>
                    </span>
                  )}
                  {m.base_rate != null && (
                    <span>
                      <span className="text-ink-3">base rate</span>{" "}
                      <span className="font-semibold">{fmtPct(m.base_rate)}</span>
                    </span>
                  )}
                </div>
              </Link>
            );
          })}
        </div>
      )}
    </div>
  );
}
