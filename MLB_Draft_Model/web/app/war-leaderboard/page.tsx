import Link from "next/link";

// Tier 5 (career WAR) was withdrawn on 2026-10-29 after failing its backtest.
// The route stays so old links land on an explanation instead of a 404.
export default function WARLeaderboardPage() {
  return (
    <div className="mx-auto max-w-[820px] px-4 py-6">
      <h1
        className="text-[26px] font-semibold"
        style={{ fontFamily: "var(--font-fraunces)" }}
      >
        WAR Value Leaderboard
      </h1>
      <p className="mt-2 text-[13px] font-semibold text-flag">Withdrawn: failed its backtest.</p>
      <p className="mt-2 text-[13px] leading-relaxed text-ink-2">
        This page ranked 2026 players by expected career WAR from the Tier 5 model. Scored out of
        fold on the 2021 class, the only class with five pro seasons complete, it did not beat
        predicting the median WAR for pitchers (R² −0.03) and barely did for hitters (R² +0.03).
        A number that weak should not rank anyone, so the model and its WAR figures are off the
        site. See the{" "}
        <Link href="/methodology/" className="text-maroon underline">
          Methodology
        </Link>{" "}
        page.
      </p>
    </div>
  );
}
