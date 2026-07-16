// Formatting rules: probabilities to whole percents, picks to whole numbers,
// rate stats to 3 places, never render null as 0.
export const NO_DATA = "—";

export function fmtPct(v: number | null | undefined, digits = 0): string {
  if (v == null) return NO_DATA;
  return `${(v * 100).toFixed(digits)}%`;
}

export function fmtRate3(v: number | null | undefined): string {
  if (v == null) return NO_DATA;
  return v.toFixed(3).replace(/^0\./, ".");
}

export function fmtNum(v: number | null | undefined, digits = 0): string {
  if (v == null) return NO_DATA;
  return v.toFixed(digits);
}

export function fmtInt(v: number | null | undefined): string {
  if (v == null) return NO_DATA;
  return Math.round(v).toLocaleString();
}

export function fmtRound(v: number | null | undefined): string {
  if (v == null) return NO_DATA;
  return `R${Math.round(v)}`;
}

export function pickToRound(pick: number): number {
  // ~30 picks per round after round 1 comp picks; good enough for display bands
  return Math.max(1, Math.min(20, Math.ceil(pick / 30.75)));
}

export function fmtRoundBand(band: [number, number] | null): string {
  if (!band) return NO_DATA;
  const lo = pickToRound(band[0]);
  const hi = pickToRound(band[1]);
  return lo === hi ? `R${lo}` : `R${lo}–R${hi}`;
}

export function fmtBonus(v: number | null | undefined): string {
  if (v == null) return NO_DATA;
  if (v >= 1_000_000) return `$${(v / 1_000_000).toFixed(2)}M`;
  if (v >= 1_000) return `$${Math.round(v / 1_000)}K`;
  return `$${v}`;
}

export function ordinal(n: number): string {
  const s = ["th", "st", "nd", "rd"];
  const v = n % 100;
  return n + (s[(v - 20) % 10] || s[v] || s[0]);
}

export function fmtHeight(inches: number | null | undefined): string {
  if (inches == null) return NO_DATA;
  const ft = Math.floor(inches / 12);
  const inc = Math.round(inches % 12);
  return `${ft}'${inc}"`;
}

export const FLAG_LABELS: Record<string, string> = {
  no_mlbam_id: "no MLBAM ID",
  low_pa: "low PA",
  low_ip: "low IP",
  no_stat_join: "no stat join",
  wide_spread: "wide model spread",
};

export const FLAG_HELP: Record<string, string> = {
  no_mlbam_id:
    "No MLB Advanced Media ID on record — draft/pro history joins are unavailable for this player.",
  low_pa: "Under 50 plate appearances in 2026 — treat every rate stat as unstable.",
  low_ip: "Under 20 innings in 2026 — treat every rate stat as unstable.",
  no_stat_join: "Projection exists but the FanGraphs stat line failed to join.",
  wide_spread:
    "Raw model probability and calibrated probability disagree by more than 35 points — the model is extrapolating. Trust the calibrated number.",
};
