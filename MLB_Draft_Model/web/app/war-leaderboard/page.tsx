"use client";

import { Suspense, useMemo, useState } from "react";
import { useIndex } from "@/lib/hooks";
import { applyFilters, DEFAULT_FILTERS } from "@/lib/filters";
import { fmtWAR } from "@/lib/format";

export default function WARLeaderboardPage() {
  const { data: rows, loading } = useIndex();
  const [type, setType] = useState<"all" | "hitter" | "pitcher">("all");
  const [showHurdleGate, setShowHurdleGate] = useState(true);

  const filtered = useMemo(() => {
    if (!rows) return [];
    const base = applyFilters(rows, { ...DEFAULT_FILTERS, type, minSample: true });
    // Filter to players with Tier 5 predictions
    return base.filter((r) => r.tier5_hurdle_prob != null);
  }, [rows, type]);

  // Sort by expected WAR (descending)
  const sorted = useMemo(() => {
    return [...filtered].sort((a, b) => {
      const warA = a.tier5_expected_war ?? -999;
      const warB = b.tier5_expected_war ?? -999;
      return warB - warA;
    });
  }, [filtered]);

  // Apply hurdle gate filter (only show players with >50% hurdle probability)
  const gated = useMemo(() => {
    if (!showHurdleGate) return sorted;
    return sorted.filter((r) => (r.tier5_hurdle_prob ?? 0) > 0.5);
  }, [sorted, showHurdleGate]);

  return (
    <Suspense>
      <div className="mx-auto max-w-[1400px] px-4 py-5">
        <div className="mb-4">
          <h1
            className="text-[28px] font-semibold leading-tight"
            style={{ fontFamily: "var(--font-fraunces)" }}
          >
            WAR Value Leaderboard
          </h1>
          <p className="mt-1 text-[13px] text-ink-2">
            Expected career WAR from Tier 5 value regression model. Hurdle gate shows only players with
            &gt;50% probability of positive WAR.
          </p>
        </div>

        {/* Controls */}
        <div className="mb-4 flex items-center gap-4">
          <div className="flex overflow-hidden rounded border border-rule">
            {(["all", "hitter", "pitcher"] as const).map((t) => (
              <button
                key={t}
                onClick={() => setType(t)}
                className={`px-3 py-1.5 text-[12px] capitalize ${
                  type === t
                    ? "bg-maroon-soft font-semibold text-maroon"
                    : "text-ink-2 hover:bg-paper-sunken"
                }`}
              >
                {t === "all" ? "All" : t + "s"}
              </button>
            ))}
          </div>

          <label className="flex items-center gap-2 text-[12px] text-ink-2">
            <input
              type="checkbox"
              checked={showHurdleGate}
              onChange={(e) => setShowHurdleGate(e.target.checked)}
              className="h-3.5 w-3.5"
            />
            Hurdle gate (&gt;50% positive WAR)
          </label>

          <div className="ml-auto text-[12px] text-ink-2">
            {gated.length} players
          </div>
        </div>

        {/* Table */}
        {loading ? (
          <div className="p-10 text-center text-[13px] text-ink-3">Loading...</div>
        ) : gated.length === 0 ? (
          <div className="rounded border border-rule bg-paper-raised p-10 text-center text-[13px] text-ink-3">
            No players match the current filters.
          </div>
        ) : (
          <div className="overflow-hidden rounded border border-rule">
            <table className="w-full text-[13px]">
              <thead className="bg-paper-sunken">
                <tr>
                  <th className="px-3 py-2 text-left font-semibold text-ink">#</th>
                  <th className="px-3 py-2 text-left font-semibold text-ink">Player</th>
                  <th className="px-3 py-2 text-left font-semibold text-ink">School</th>
                  <th className="px-3 py-2 text-center font-semibold text-ink">Type</th>
                  <th className="px-3 py-2 text-right font-semibold text-ink">Proj Pick</th>
                  <th className="px-3 py-2 text-right font-semibold text-ink">MLB%</th>
                  <th className="px-3 py-2 text-right font-semibold text-ink">Hurdle%</th>
                  <th className="px-3 py-2 text-right font-semibold text-ink">Exp WAR</th>
                  <th className="px-3 py-2 text-center font-semibold text-ink">Confidence</th>
                </tr>
              </thead>
              <tbody>
                {gated.map((player, idx) => {
                  const hurdleProb = player.tier5_hurdle_prob ?? 0;
                  const expectedWAR = player.tier5_expected_war ?? 0;
                  const confidence = player.tier5_confidence ?? "low";

                  return (
                    <tr
                      key={player.id}
                      className="border-t border-rule hover:bg-paper-sunken"
                    >
                      <td className="px-3 py-2 text-ink-2">{idx + 1}</td>
                      <td className="px-3 py-2 font-medium text-ink">{player.name}</td>
                      <td className="px-3 py-2 text-ink-2">{player.school}</td>
                      <td className="px-3 py-2 text-center">
                        <span
                          className={`inline-block rounded px-2 py-0.5 text-[11px] font-semibold ${
                            player.type === "hitter"
                              ? "bg-hitter/10 text-hitter"
                              : "bg-pitcher/10 text-pitcher"
                          }`}
                        >
                          {player.type === "hitter" ? "H" : "P"}
                        </span>
                      </td>
                      <td className="px-3 py-2 text-right text-ink-2">
                        {player.proj_pick != null ? Math.round(player.proj_pick) : "—"}
                      </td>
                      <td className="px-3 py-2 text-right">
                        {player.mlb_p != null ? (
                          <span className="font-semibold text-ink">
                            {(player.mlb_p * 100).toFixed(0)}%
                          </span>
                        ) : (
                          <span className="text-ink-3">—</span>
                        )}
                      </td>
                      <td className="px-3 py-2 text-right">
                        <span
                          className={`font-semibold ${
                            hurdleProb > 0.7
                              ? "text-green-600"
                              : hurdleProb > 0.5
                              ? "text-yellow-600"
                              : "text-red-600"
                          }`}
                        >
                          {(hurdleProb * 100).toFixed(0)}%
                        </span>
                      </td>
                      <td className="px-3 py-2 text-right">
                        <span
                          className={`font-semibold ${
                            expectedWAR > 2
                              ? "text-green-600"
                              : expectedWAR > 0.5
                              ? "text-ink"
                              : expectedWAR > 0
                              ? "text-yellow-600"
                              : "text-red-600"
                          }`}
                        >
                          {fmtWAR(expectedWAR)}
                        </span>
                      </td>
                      <td className="px-3 py-2 text-center">
                        <span
                          className={`inline-block rounded px-2 py-0.5 text-[11px] font-semibold ${
                            confidence === "high"
                              ? "bg-green-100 text-green-700"
                              : confidence === "medium"
                              ? "bg-yellow-100 text-yellow-700"
                              : "bg-red-100 text-red-700"
                          }`}
                        >
                          {confidence.toUpperCase()}
                        </span>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}

        {/* Legend */}
        <div className="mt-4 rounded border border-rule bg-paper-raised p-3 text-[11px] text-ink-2">
          <div className="mb-1 font-semibold text-ink">Notes:</div>
          <ul className="list-disc space-y-0.5 pl-4">
            <li>
              <strong>Exp WAR</strong>: Expected career WAR from Tier 5 value regression (LightGBM)
            </li>
            <li>
              <strong>Hurdle%</strong>: Probability of positive WAR (&gt;0.0) from Tier 5 hurdle model
            </li>
            <li>
              <strong>Confidence</strong>: Based on model gate status (high = R² ≥ 0.0, low = R² &lt; 0.0)
            </li>
            <li>
              Hurdle gate filters to players with &gt;50% probability of positive WAR
            </li>
          </ul>
        </div>
      </div>
    </Suspense>
  );
}
