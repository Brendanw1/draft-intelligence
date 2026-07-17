"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { fmtPct, fmtNum, fmtRoundBand, pickToRound, fmtHeight, NO_DATA } from "@/lib/format";
import {
  HITTER_PCTL_PANEL,
  HITTER_SEASON_COLS,
  PITCHER_PCTL_PANEL,
  PITCHER_SEASON_COLS,
} from "@/lib/stats";
import { loadNote, saveNote, useUI } from "@/lib/store";
import { DetailPlayer } from "@/lib/types";
import { FlagChips, GradeChip, TypeBadge } from "@/components/shared/Chips";
import { SeasonTable } from "./SeasonTable";
import { PctlBars } from "./PctlBars";
import { CompsList } from "./CompsList";

function VerdictRow({ p }: { p: DetailPlayer }) {
  const roundBand = fmtRoundBand(p.pick_band);
  return (
    <div className="grid grid-cols-3 divide-x divide-rule border-y border-rule">
      <div className="px-4 py-3">
        <div className="text-[10px] font-semibold uppercase tracking-[0.13em] text-ink-3">
          Projected
        </div>
        <div className="text-[24px] font-semibold leading-tight">
          {p.proj_pick != null
            ? `R${pickToRound(p.proj_pick)} · pick ${Math.round(p.proj_pick)}`
            : NO_DATA}
        </div>
        <div className="text-[11px] text-ink-2">
          {roundBand != null ? `${roundBand} range ±~110 picks` : "no round estimate"}
        </div>
        {/* band on a R1–R20 axis */}
        {p.pick_band && p.proj_pick != null && (
          <svg width="100%" height={18} viewBox="0 0 200 18" className="mt-1">
            <line x1={0} y1={9} x2={200} y2={9} stroke="var(--rule)" />
            {[1, 5, 10, 15, 20].map((r) => (
              <g key={r}>
                <line x1={(r * 30.75 * 200) / 620} y1={6} x2={(r * 30.75 * 200) / 620} y2={12} stroke="var(--rule-strong)" />
              </g>
            ))}
            <rect
              x={(p.pick_band[0] / 620) * 200}
              y={6.5}
              width={Math.max(3, ((p.pick_band[1] - p.pick_band[0]) / 620) * 200)}
              height={5}
              rx={2.5}
              fill="var(--band)"
            />
            <line
              x1={(p.proj_pick / 620) * 200}
              y1={2}
              x2={(p.proj_pick / 620) * 200}
              y2={16}
              stroke="var(--band-strong)"
              strokeWidth={2.5}
            />
          </svg>
        )}
      </div>
      <div className="px-4 py-3">
        <div className="text-[10px] font-semibold uppercase tracking-[0.13em] text-ink-3">
          Top-10-round probability
        </div>
        <div className="text-[24px] font-semibold leading-tight">{fmtPct(p.mlb_p)}</div>
        <div className="text-[11px] leading-snug text-ink-2">
          raw model: {fmtPct(p.mlb_p_raw)}
          {p.hist_rate != null && (
            <>
              {" "}
              · players scored like this historically: ~{fmtPct(p.hist_rate)}
            </>
          )}
        </div>
        {p.flags.includes("wide_spread") && (
          <div className="mt-1 inline-block rounded-sm bg-flag-bg px-1.5 py-px text-[10px] font-medium text-flag">
            wide model spread — trust the calibrated number
          </div>
        )}
      </div>
      <div className="px-4 py-3">
        <div className="text-[10px] font-semibold uppercase tracking-[0.13em] text-ink-3">
          Grade
        </div>
        <div className="mt-1">
          <GradeChip grade={p.grade} />
        </div>
        <div className="mt-1 text-[11px] text-ink-2">
          composite {fmtNum(p.composite, 1)}{" "}
          <span className="text-ink-3">(30% slot · 40% top-10 · 30% arrival)</span>
        </div>
      </div>
    </div>
  );
}

function TrustPanel({ p }: { p: DetailPlayer }) {
  const [open, setOpen] = useState(false);
  const t = p.type;
  return (
    <div className="border-t border-rule">
      <button
        onClick={() => setOpen(!open)}
        className="flex w-full items-center justify-between px-4 py-2.5 text-left text-[11px] font-semibold uppercase tracking-[0.13em] text-ink-3 hover:text-ink"
      >
        Model provenance & caveats
        <span aria-hidden>{open ? "−" : "+"}</span>
      </button>
      {open && (
        <div className="space-y-2 px-4 pb-4 text-[12px] leading-relaxed text-ink-2">
          <p>
            Scored by{" "}
            <Link href={`/models/fg-draft-${t}/`} className="text-maroon underline">
              fg_draft_{t}
            </Link>{" "}
            (draft position),{" "}
            <Link href={`/models/tier2-predraft-${t}/`} className="text-maroon underline">
              tier2_predraft_{t}
            </Link>{" "}
            (top-10-round probability, Platt-calibrated), and Tier 3 Elastic Net (MLB arrival, round-anchored).
            Inputs are FanGraphs D1 stats normalized by conference strength — no
            scouting, defense, or medical signal.
          </p>
          <ul className="list-disc space-y-1 pl-4">
            <li>Pick estimates carry ±~110 picks of backtest error — read rounds, not slots.</li>
            <li>Conference-adjusted stats (wOBA_adj, ERA_adj) and conf_strength are first-class features — small-conference production is discounted proportionally.</li>
            {t === "pitcher" && (
              <li>
                Pitcher probabilities between 20–60% were historically overconfident —
                lean on the “players scored like this” rate.
              </li>
            )}
            {p.flags.includes("low_pa") || p.flags.includes("low_ip") ? (
              <li>Below the qualification threshold — every rate stat here is unstable.</li>
            ) : null}
          </ul>
        </div>
      )}
    </div>
  );
}

function Notes({ id }: { id: string }) {
  const [text, setText] = useState("");
  useEffect(() => setText(loadNote(id)), [id]);
  return (
    <div className="border-t border-rule px-4 py-3">
      <div className="mb-1 flex items-baseline justify-between">
        <span className="text-[10px] font-semibold uppercase tracking-[0.13em] text-ink-3">
          Your notes
        </span>
        <span className="text-[10px] text-ink-3">saved locally — not model input</span>
      </div>
      <textarea
        value={text}
        onChange={(e) => {
          setText(e.target.value);
          saveNote(id, e.target.value);
        }}
        rows={3}
        placeholder="Looks, makeup, follow status…"
        className="w-full rounded border border-rule bg-paper-raised p-2 text-[13px] placeholder:text-ink-3 focus:border-rule-strong focus:outline-none"
      />
    </div>
  );
}

export function PlayerDossier({
  p,
  inDrawer,
}: {
  p: DetailPlayer;
  inDrawer?: boolean;
}) {
  const { toggleCompare, compareIds } = useUI();
  const panel = p.type === "hitter" ? HITTER_PCTL_PANEL : PITCHER_PCTL_PANEL;
  const seasonCols = p.type === "hitter" ? HITTER_SEASON_COLS : PITCHER_SEASON_COLS;
  const qualified = !p.flags.includes("low_pa") && !p.flags.includes("low_ip");

  return (
    <div className="flex h-full flex-col">
      {/* header */}
      <div className="px-4 pb-3 pt-4">
        <div className="flex items-start justify-between gap-3">
          <div>
            <h2
              className="text-[22px] font-semibold leading-tight"
              style={{ fontFamily: "var(--font-fraunces)" }}
            >
              {p.name}
            </h2>
            <div className="mt-0.5 flex flex-wrap items-center gap-2 text-[12px] text-ink-2">
              <TypeBadge type={p.type} />
              <span>{p.school}</span>
              {p.conference && (
                <span className="rounded-sm bg-paper-sunken px-1.5 py-px text-[11px]">
                  {p.conference}
                </span>
              )}
              <span>{p.age != null ? `age ${p.age}` : NO_DATA}</span>
              <span className="text-ink-3">
                {p.type === "hitter"
                  ? `${p.sample.pa != null ? Math.round(p.sample.pa) + " PA" : "PA " + NO_DATA}`
                  : `${p.sample.ip != null ? p.sample.ip.toFixed(1) + " IP" : "IP " + NO_DATA}`}{" "}
                in 2026
              </span>
            </div>
          </div>
          <div className={`flex shrink-0 items-center gap-2 ${inDrawer ? "pr-16" : ""}`}>
            <button
              onClick={() => toggleCompare(p.id)}
              className={`rounded border px-2 py-1 text-[11px] ${
                compareIds.includes(p.id)
                  ? "border-maroon bg-maroon-soft text-maroon"
                  : "border-rule text-ink-2 hover:border-rule-strong"
              }`}
            >
              {compareIds.includes(p.id) ? "✓ comparing" : "+ compare"}
            </button>
            {inDrawer && (
              <Link
                href={`/player/?id=${encodeURIComponent(p.id)}`}
                className="rounded border border-rule px-2 py-1 text-[11px] text-ink-2 hover:border-rule-strong"
              >
                expand ↗
              </Link>
            )}
          </div>
        </div>
        {p.flags.length > 0 && (
          <div className="mt-2">
            <FlagChips flags={p.flags} max={5} />
          </div>
        )}
      </div>

      <VerdictRow p={p} />

      {/* MLB Outlook — Tier 3: arrival probability */}
      {p.mlb_arrival != null && (
        <div className="border-t border-rule px-4 py-3">
          <div className="mb-2 text-[10px] font-semibold uppercase tracking-[0.13em] text-ink-3">
            MLB Outlook (if drafted) — Tier 3
          </div>
          <div className="grid grid-cols-3 gap-3">
            <div>
              <div className="text-[10px] text-ink-3">Arrival probability</div>
              <div className="text-[18px] font-semibold leading-tight">
                {fmtPct(p.mlb_arrival)}
              </div>
              <div className="text-[11px] leading-snug text-ink-2">
                Elastic Net with round-anchored prior
              </div>
            </div>
            <div>
              <div className="text-[10px] text-ink-3">NN comp rate</div>
              <div className="text-[18px] font-semibold leading-tight">
                {p.nn_mlb_rate != null ? fmtPct(p.nn_mlb_rate) : NO_DATA}
              </div>
              <div className="text-[11px] leading-snug text-ink-2">
                % of 20 nearest comps who reached MLB
              </div>
            </div>
            <div>
              <div className="text-[10px] text-ink-3">Round baseline</div>
              <div className="text-[18px] font-semibold leading-tight">
                {p.proj_round != null
                  ? `R${p.proj_round} → ${fmtPct(
                      p.proj_round <= 5 ? 0.27 :
                      p.proj_round <= 10 ? 0.12 :
                      p.proj_round <= 15 ? 0.06 :
                      0.03
                    )}`
                  : NO_DATA}
              </div>
              <div className="text-[11px] leading-snug text-ink-2">
                historical MLB rate for this round
              </div>
            </div>
          </div>
          {/* model disagreement indicator */}
          {p.mlb_p != null && p.mlb_arrival != null && (
            <div className="mt-2 rounded-sm bg-caution-bg px-2.5 py-1.5">
              <div className="flex items-center gap-2 text-[11px]">
                <span className="font-semibold text-caution">Model check:</span>
                {p.mlb_p > 0.50 && p.mlb_arrival < 0.10 ? (
                  <span className="text-text-secondary">
                    Tier 2 is bullish ({fmtPct(p.mlb_p)} top-10%) but Tier 3 is cautious
                    ({fmtPct(p.mlb_arrival)} arrival). The Tier 2 historical rate is{" "}
                    {p.hist_rate != null ? fmtPct(p.hist_rate) : "unknown"} — check the model card.
                  </span>
                ) : p.mlb_p < 0.15 && p.mlb_arrival > 0.10 ? (
                  <span className="text-text-secondary">
                    Tier 2 sees low top-10 odds ({fmtPct(p.mlb_p)}) but Tier 3 suggests
                    above-average arrival potential ({fmtPct(p.mlb_arrival)}) if drafted.
                    Worth a closer scouting look.
                  </span>
                ) : p.mlb_p > 0.50 && p.mlb_arrival > 0.15 ? (
                  <span className="text-text-secondary">
                    Both models agree: strong top-10 probability ({fmtPct(p.mlb_p)}) and
                    above-average arrival outlook ({fmtPct(p.mlb_arrival)}).
                  </span>
                ) : (
                  <span className="text-text-secondary">
                    Tier 2 and Tier 3 are broadly aligned —{" "}
                    {p.mlb_arrival > 0.05
                      ? "moderate-to-strong arrival outlook"
                      : "arrival probability is in the typical range"}
                    .
                  </span>
                )}
              </div>
            </div>
          )}
        </div>
      )}

      {/* Physical profile — height/BMI are #1 and #2 model features */}
      <div className="border-t border-rule px-4 py-3">
        <div className="mb-2 text-[10px] font-semibold uppercase tracking-[0.13em] text-ink-3">
          Physical profile
        </div>
        <div className="grid grid-cols-3 gap-3">
          <div>
            <div className="text-[10px] text-ink-3">Height</div>
            <div className="text-[18px] font-semibold leading-tight">
              {p.height_display ?? (p.height_inches ? fmtHeight(p.height_inches) : NO_DATA)}
            </div>
            <div className="text-[11px] text-ink-2">
              {p.height_inches != null
                ? `${p.height_inches} in`
                : NO_DATA}
            </div>
          </div>
          <div>
            <div className="text-[10px] text-ink-3">BMI</div>
            <div className="text-[18px] font-semibold leading-tight">
              {p.bmi != null ? p.bmi.toFixed(1) : NO_DATA}
            </div>
            <div className="text-[11px] text-ink-2">
              {p.bmi != null
                ? p.bmi >= 27
                  ? "sturdy build"
                  : p.bmi >= 24
                    ? "athletic"
                    : "lean"
                : "not recorded"}
            </div>
          </div>
          <div>
            <div className="text-[10px] text-ink-3">Draftability</div>
            <div className="text-[18px] font-semibold leading-tight">
              {p.draftability_score != null ? `${(p.draftability_score * 100).toFixed(1)}%` : NO_DATA}
            </div>
            <div className="text-[11px] text-ink-2">
              {p.conference_tier != null
                ? `conf tier ${p.conference_tier} (legacy — model uses continuous conf_strength)`
                : NO_DATA}
            </div>
          </div>
        </div>
      </div>

      <div className="thin-scroll flex-1 overflow-y-auto">
        {/* why the model thinks this */}
        <div className="px-4 py-3">
          <div className="mb-2 flex items-baseline justify-between">
            <span className="text-[10px] font-semibold uppercase tracking-[0.13em] text-ink-3">
              Model priority stats — this player’s standing
            </span>
            <span className="text-[10px] text-ink-3">
              vs. qualified 2026 {p.type}s
            </span>
          </div>
          {qualified && Object.keys(p.pctl).length ? (
            <PctlBars pctl={p.pctl} panel={panel} />
          ) : (
            <div className="rounded bg-paper-sunken p-3 text-[12px] text-ink-3">
              Below the sample threshold — percentiles suppressed rather than shown
              on noise.
            </div>
          )}
        </div>

        {/* seasons */}
        <div className="border-t border-rule px-4 py-3">
          <div className="mb-2 text-[10px] font-semibold uppercase tracking-[0.13em] text-ink-3">
            FanGraphs seasons
          </div>
          {p.seasons.length ? (
            <SeasonTable seasons={p.seasons} cols={seasonCols} />
          ) : (
            <div className="text-[12px] text-ink-3">No stat line joined.</div>
          )}
        </div>

        {/* comps */}
        <div className="border-t border-rule px-4 py-3">
          <div className="mb-2 flex items-baseline justify-between">
            <span className="text-[10px] font-semibold uppercase tracking-[0.13em] text-ink-3">
              Similar drafted players, 2021–24
            </span>
            <span className="text-[10px] text-ink-3">nearest stat profiles</span>
          </div>
          <CompsList comps={p.comps} projPick={p.proj_pick} />
        </div>

        <TrustPanel p={p} />
        <Notes id={p.id} />
      </div>
    </div>
  );
}
