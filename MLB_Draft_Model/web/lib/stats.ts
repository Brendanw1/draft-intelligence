import { Grade, PlayerType } from "./types";

export const GRADE_ORDER: Grade[] = ["elite", "high", "medium", "low"];

export const GRADE_LABEL: Record<Grade, string> = {
  elite: "Elite",
  high: "High",
  medium: "Medium",
  low: "Low",
};

export const GRADE_TIER_NOTE: Record<Grade, string> = {
  elite: "top 1% composite among qualified — day-one conversation",
  high: "95th–99th percentile — priority follow",
  medium: "80th–95th percentile — situational",
  low: "below 80th percentile — organizational depth",
};

// CSS variables so dark mode re-derives automatically
export const gradeColor = (g: Grade) => `var(--grade-${g})`;
export const gradeBg = (g: Grade) => `var(--grade-${g}-bg)`;
export const typeColor = (t: PlayerType) => `var(--${t})`;
export const typeBg = (t: PlayerType) => `var(--${t}-bg)`;

export interface StatDef {
  key: string; // key in key_stats (index) or season row (detail)
  seasonKey?: string; // key in season rows if different
  label: string;
  kind: "rate3" | "pct1" | "num1" | "num2" | "int";
  lowerBetter?: boolean;
  help?: string;
}

export const HITTER_BOARD_STATS: StatDef[] = [
  { key: "wOBA", label: "wOBA", kind: "rate3", help: "Weighted on-base average — total offense per PA." },
  { key: "wRCplus", label: "wRC+", kind: "int", help: "Runs created, indexed to D1 average = 100. No conference adjustment." },
  { key: "ops", label: "OPS", kind: "rate3" },
  { key: "hr", label: "HR", kind: "int" },
  { key: "bb_pct", label: "BB%", kind: "pct1" },
  { key: "k_pct", label: "K%", kind: "pct1", lowerBetter: true },
];

export const PITCHER_BOARD_STATS: StatDef[] = [
  { key: "era", label: "ERA", kind: "num2", lowerBetter: true },
  { key: "fip", label: "FIP", kind: "num2", lowerBetter: true, help: "Fielding-independent pitching. Uses an MLB constant — compare relatively, not absolutely." },
  { key: "k9", label: "K/9", kind: "num1" },
  { key: "bb9", label: "BB/9", kind: "num1", lowerBetter: true },
  { key: "k_pct", label: "K%", kind: "pct1" },
  { key: "bb_pct", label: "BB%", kind: "pct1", lowerBetter: true },
];

// Percentile panel: the stats the Tier 1 models weight most, in importance order
export const HITTER_PCTL_PANEL: { stat: string; label: string; why: string }[] = [
  { stat: "wRCplus", label: "wRC+", why: "top production signal" },
  { stat: "bb_pct", label: "BB%", why: "plate discipline — heavily weighted" },
  { stat: "hr", label: "HR", why: "power volume" },
  { stat: "wOBA", label: "wOBA", why: "rate production" },
  { stat: "iso", label: "ISO", why: "raw power" },
  { stat: "k_pct", label: "K%", why: "contact risk (lower is better)" },
  { stat: "pa", label: "PA", why: "sample / durability" },
];

export const PITCHER_PCTL_PANEL: { stat: string; label: string; why: string }[] = [
  { stat: "so", label: "SO", why: "strikeout volume — #1 model feature" },
  { stat: "k_minus_bb_pct", label: "K-BB%", why: "stuff minus command" },
  { stat: "k_pct", label: "K%", why: "swing-and-miss" },
  { stat: "fip", label: "FIP", why: "defense-independent run prevention" },
  { stat: "whip", label: "WHIP", why: "baserunner suppression" },
  { stat: "gs", label: "GS", why: "starter workload" },
  { stat: "ip", label: "IP", why: "sample / durability" },
  { stat: "bb9", label: "BB/9", why: "walk rate (lower is better)" },
];

export const HITTER_SEASON_COLS: StatDef[] = [
  { key: "Season", label: "Yr", kind: "int" },
  { key: "PA", label: "PA", kind: "int" },
  { key: "AVG", label: "AVG", kind: "rate3" },
  { key: "OBP", label: "OBP", kind: "rate3" },
  { key: "SLG", label: "SLG", kind: "rate3" },
  { key: "HR", label: "HR", kind: "int" },
  { key: "SB", label: "SB", kind: "int" },
  { key: "BB_pct", label: "BB%", kind: "pct1" },
  { key: "K_pct", label: "K%", kind: "pct1" },
  { key: "wOBA", label: "wOBA", kind: "rate3" },
  { key: "wRC_plus", label: "wRC+", kind: "int" },
];

export const PITCHER_SEASON_COLS: StatDef[] = [
  { key: "Season", label: "Yr", kind: "int" },
  { key: "G", label: "G", kind: "int" },
  { key: "GS", label: "GS", kind: "int" },
  { key: "IP", label: "IP", kind: "num1" },
  { key: "SO", label: "SO", kind: "int" },
  { key: "ERA", label: "ERA", kind: "num2" },
  { key: "FIP", label: "FIP", kind: "num2" },
  { key: "WHIP", label: "WHIP", kind: "num2" },
  { key: "K_pct", label: "K%", kind: "pct1" },
  { key: "BB_pct", label: "BB%", kind: "pct1" },
  { key: "HR_per_nine", label: "HR/9", kind: "num2" },
];
