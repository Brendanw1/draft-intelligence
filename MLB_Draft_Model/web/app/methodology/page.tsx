import Link from "next/link";

const GLOSSARY: [string, string][] = [
  ["wOBA", "Weighted on-base average — every offensive event valued by how many runs it's actually worth, per plate appearance. The best single rate stat for a hitter."],
  ["wRC+", "Runs created, indexed so 100 = D1 average. 150 means 50% better than average. Not conference-adjusted here — a 150 in the SEC is worth more than a 150 in the SWAC, though conf_strength and adjusted stats partially compensate."],
  ["ISO", "Isolated power: slugging minus average. Pure extra-base damage."],
  ["BB% / K%", "Walks and strikeouts per plate appearance. The Tier 1 hitter model leans hard on plate discipline — this matches how teams actually draft."],
  ["FIP", "Fielding-independent pitching — what a pitcher's ERA 'should' be from strikeouts, walks, and homers alone. Our FIP uses an MLB constant, so compare pitchers to each other, not to a magic number."],
  ["K-BB%", "Strikeout rate minus walk rate — the fastest single read on a pitcher's dominance."],
  ["Platt calibration", "A learned correction that maps the model's raw score to an honest probability, fit on out-of-fold scores (no player in both the fitting and the scored rows). It is the number shown everywhere on this site. On the 2024 class, with the model trained only on seasons whose labels were complete before the 2024 draft: AUC 0.857 hitters / 0.843 pitchers, ECE 1.4% / 0.6%. The 2024 class had been read on earlier versions of the model, so this is a re-read; the next clean read is the 2025 class after the 2027 draft. The board's probabilities add up to about 714 top-315 picks over three drafts, about 238 a year, against roughly 250 four-year college players taken in the top 315 each year."],
  ["Tier 2 / Top-315 probability", "P(drafted at pick 315 or better, about 10 rounds, in this season's draft or the next two). Every D1 season with a FanGraphs line is a training row, drafted or not, freshmen included, for the seasons 2021–2024 whose three drafts have all happened. Age, height and BMI are not inputs: the sources record them for drafted players far more completely than for undrafted ones, so they told the model the answer (fixed 2026-10-29)."],
  ["Isotonic calibration", "A second, non-linear correction fit on the same out-of-fold scores, kept as a cross-check. When Platt and isotonic agree, trust the number more."],
  ["ECE", "Expected calibration error — the average gap between what the model predicted and what actually happened, before correction."],
  ["Composite score", "Three-component blend: 30% projected draft slot value (Tier 1) + 40% calibrated top-315 probability (Tier 2) + 30% Tier 3 MLB arrival probability. Scaled 0–100. It is an opinion, not an output of any single model."],
  ["Tier 5 / WAR Value", "Withdrawn: failed its backtest. A career-WAR model (P(reach MLB) × expected WAR if reached) was scored out of fold on the 2021 class, the only class with five pro seasons complete. It did not beat predicting the median for pitchers (R² −0.03) and barely did for hitters (R² +0.03). No WAR numbers appear on this site."],
  ["Value grade", "Composite percentile within qualified players of the same type: elite = top 1%, high = 95th–99th, medium = 80th–95th, low = the rest. Percentile tiers, because calibrated probabilities compress absolute scores."],
  ["Tier 1 / Projected pick", "Where a drafted player with this stat line would go: the pick if drafted. It is trained on drafted players only, so it says nothing about whether he is drafted (that is Tier 2). Backtested forward by draft year (fit on earlier classes): mean absolute error 117 picks hitters / 115 pitchers on the 2024 and 2025 classes, against about 140 for predicting the average pick."],
  ["Spearman ρ", "Rank correlation between projected and actual draft order in backtests. ~0.5 means the model orders players meaningfully but far from perfectly."],
  ["Tier 3 / MLB Arrival", "Predicts P(MLB debut within 3 seasons | drafted): an L2 logistic regression on standardised stats, with the draft round's historical debut rate as a fixed logit offset and a nearest-neighbor comp rate. Trained on the 2021–2022 classes, the ones with three complete MLB seasons in the data (debut dates were pulled in July 2026, so 2026 counts as incomplete). Draft round carries most of it: in cross-validation the round rate alone ranks players better (AUC 0.72 hitters, 0.77 pitchers) than the full model (0.70, 0.75)."],
  ["Conf_strength", "Continuous conference strength score: the conference's rate of top-315 picks within three drafts divided by the D1 rate, from the training seasons only (2021–2024 on the site; each backtest refits it on its own training seasons). SEC = 3.26×, SWAC = 0.06×."],
  ["Conference-adjusted stats", "Raw stats (wOBA, ERA, etc.) are adjusted by subtracting the conference-average stat (e.g. wOBA_adj = wOBA − SEC_avg_wOBA). Separate strength×stat interaction features let the model learn non-linear conference effects. Together this prevents overrating small-conference production and underrating elite-conference production."],
];

const LIMITS: [string, string][] = [
  ["Conference adjustment is approximate", "Conference strength is based on empirical draft rates (seasons 2021–2024), which is a reasonable proxy but not a direct measure of conference quality. A few small conferences (DI Independent) have high draft rates due to individual programs, not overall strength."],
  ["Tier 3 is mostly draft round", "Tier 3 (MLB arrival) is trained on 2021–2023 draftees, each with three full seasons to debut. Its college stats add no ranking power beyond the draft round: read it as the round's historical rate, nudged. Small 2026 samples can push it around."],
  ["No class year or age", "A sophomore hitting .350 and a senior hitting .350 look identical to Tier 2. Age was removed because FanGraphs has it for every drafted player but for under half of the undrafted. A freshman's probability covers his next three drafts, of which he is eligible for at most one."],
];

export default function MethodologyPage() {
  return (
    <div className="mx-auto max-w-[820px] px-4 py-6">
      <h1
        className="text-[26px] font-semibold"
        style={{ fontFamily: "var(--font-fraunces)" }}
      >
        Methodology
      </h1>
      <p className="mt-1 text-[13px] leading-relaxed text-ink-2">
        What the numbers are, where they come from, and where they break. Written
        for the staff meeting, not the seminar room.
      </p>

      <section className="mt-6">
        <h2 className="text-[11px] font-semibold uppercase tracking-[0.14em] text-ink-3">
          The pipeline in one paragraph
        </h2>
        <p className="mt-2 text-[13px] leading-relaxed text-ink-2">
          FanGraphs D1 stats (2021–2026) are joined to MLB draft records via
          MLBAM IDs (99.9% match rate). Conference-adjusted stats (wOBA_adj,
          ERA_adj) and a continuous conference strength score (conf_strength)
          normalize for competitive level — small-conference production is
          now discounted proportionally. A regression model learns where drafted
          stat profiles went (the projected pick, if drafted); a second classification
          model, trained on every D1 season, predicts a pick of 315 or better within three drafts. Raw
          probabilities are corrected with Platt scaling fit on out-of-fold scores, the calibration scored on the held-out 2024 class. A third tier, a
          logistic regression offset by the draft round's historical rate, predicts P(MLB debut within 3 seasons | drafted).
          A career-WAR tier (Tier 5) was built and withdrawn after it failed its backtest; it is not on the site.
          Every 2026 D1 player with a FanGraphs line — all 10,734 — gets scored.
          Details on each artifact live in the{" "}
          <Link href="/models/" className="text-maroon underline">Model Lab</Link>.
        </p>
      </section>

      <section className="mt-6">
        <h2 className="text-[11px] font-semibold uppercase tracking-[0.14em] text-ink-3">
          How to read this site honestly
        </h2>
        <ul className="mt-2 list-disc space-y-1.5 pl-5 text-[13px] leading-relaxed text-ink-2">
          <li>
            <span className="font-semibold text-ink">Round bands, not slots.</span>{" "}
            The tick is a point estimate; the band is what the backtest supports.
          </li>
          <li>
            <span className="font-semibold text-ink">Calibrated percentages, always.</span>{" "}
            When you see “raw model: 84%” next to “21%”, the 84% is what the model
            wanted to say and the 21% is what history supports.
          </li>
          <li>
            <span className="font-semibold text-ink">Grades are tiers, not ranks.</span>{" "}
            There is deliberately no “#1 overall” anywhere on the board.
          </li>
          <li>
            <span className="font-semibold text-ink">Gray means no data, never zero.</span>{" "}
            A dash is an honest absence.
          </li>
          <li>
            <span className="font-semibold text-ink">Tier 3 arrival probability</span>{" "}
            predicts P(MLB debut within 3 seasons, if drafted). It is round-anchored — a 5th-rounder
            with 15% arrival is above their round baseline; a 1st-rounder with 15%
            is below theirs. Always compare to the round's historical rate.
          </li>
        </ul>
      </section>

      <section className="mt-6">
        <h2 className="text-[11px] font-semibold uppercase tracking-[0.14em] text-ink-3">
          Known limitations — the register
        </h2>
        <div className="mt-2 divide-y divide-rule rounded border border-rule">
          {LIMITS.map(([title, body]) => (
            <div key={title} className="px-3 py-2.5">
              <div className="text-[13px] font-semibold">{title}</div>
              <div className="text-[12px] leading-relaxed text-ink-2">{body}</div>
            </div>
          ))}
        </div>
      </section>

      <section className="mt-6">
        <h2 className="text-[11px] font-semibold uppercase tracking-[0.14em] text-ink-3">
          Glossary
        </h2>
        <div className="mt-2 divide-y divide-rule rounded border border-rule">
          {GLOSSARY.map(([term, body]) => (
            <div key={term} className="flex gap-4 px-3 py-2.5">
              <div className="w-[120px] shrink-0 text-[13px] font-semibold">{term}</div>
              <div className="text-[12px] leading-relaxed text-ink-2">{body}</div>
            </div>
          ))}
        </div>
      </section>

      <section className="mt-6">
        <h2 className="text-[11px] font-semibold uppercase tracking-[0.14em] text-ink-3">
          Data freshness & provenance
        </h2>
        <ul className="mt-2 list-disc space-y-1 pl-5 text-[13px] text-ink-2">
          <li>College stats: FanGraphs D1 leaderboards, 2021–2026.</li>
          <li>Draft records: MLB Stats API, 2015–2026 (9,300+ picks).</li>
          <li>Pro outcomes: MiLB game feeds 2021–2025 + MLB debut records.</li>
          <li>Conference labels: static 2026 map, approximate — a display label, not a model input.</li>
          <li>Tier 3 arrival model: trained on the 2021–2023 classes (debut by the end of the third season after the draft). 2024+ classes are not yet observed for three seasons and are left out.</li>
          <li>Model vintage is stamped in the top bar; regenerate via <code className="rounded bg-paper-sunken px-1">scripts/export_frontend_data.py</code>.</li>
        </ul>
      </section>

      <p className="mt-8 border-t border-rule pt-4 text-[13px] italic leading-relaxed text-ink-2">
        These models summarize public statistical performance. They have never
        seen a player throw. They are screening lenses to point scouting hours,
        not replacements for them.
      </p>
    </div>
  );
}
