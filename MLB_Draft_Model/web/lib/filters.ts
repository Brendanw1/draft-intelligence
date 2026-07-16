import { Grade, IndexPlayer, PlayerType } from "./types";

export interface Filters {
  type: PlayerType | "all";
  q: string; // name search
  conferences: string[];
  schools: string[];
  grades: Grade[];
  confidence: ("high" | "medium" | "low")[];
  minSample: boolean; // PA>=50 / IP>=20
  hideNoJoin: boolean;
  pickMin: number | null;
  pickMax: number | null;
  probMin: number | null; // calibrated MLB%
  probMax: number | null;
  sort: string;
  dir: "asc" | "desc";
}

export const DEFAULT_FILTERS: Filters = {
  type: "all",
  q: "",
  conferences: [],
  schools: [],
  grades: [],
  confidence: [],
  minSample: true,
  hideNoJoin: false,
  pickMin: null,
  pickMax: null,
  probMin: null,
  probMax: null,
  sort: "composite",
  dir: "desc",
};

export function filtersToParams(f: Filters): URLSearchParams {
  const p = new URLSearchParams();
  if (f.type !== "all") p.set("type", f.type);
  if (f.q) p.set("q", f.q);
  if (f.conferences.length) p.set("conf", f.conferences.join("~"));
  if (f.schools.length) p.set("school", f.schools.join("~"));
  if (f.grades.length) p.set("grade", f.grades.join("~"));
  if (f.confidence.length) p.set("conf1", f.confidence.join("~"));
  if (!f.minSample) p.set("all_samples", "1");
  if (f.pickMin != null) p.set("pick_min", String(f.pickMin));
  if (f.pickMax != null) p.set("pick_max", String(f.pickMax));
  if (f.probMin != null) p.set("p_min", String(f.probMin));
  if (f.probMax != null) p.set("p_max", String(f.probMax));
  if (f.sort !== "composite") p.set("sort", f.sort);
  if (f.dir !== "desc") p.set("dir", f.dir);
  return p;
}

export function paramsToFilters(p: URLSearchParams): Filters {
  const num = (k: string) => {
    const v = p.get(k);
    return v == null || v === "" || isNaN(Number(v)) ? null : Number(v);
  };
  const list = (k: string) => {
    const v = p.get(k);
    return v ? v.split("~").filter(Boolean) : [];
  };
  const type = p.get("type");
  return {
    type: type === "hitter" || type === "pitcher" ? type : "all",
    q: p.get("q") ?? "",
    conferences: list("conf"),
    schools: list("school"),
    grades: list("grade") as Grade[],
    confidence: list("conf1") as Filters["confidence"],
    minSample: p.get("all_samples") !== "1",
    hideNoJoin: false,
    pickMin: num("pick_min"),
    pickMax: num("pick_max"),
    probMin: num("p_min"),
    probMax: num("p_max"),
    sort: p.get("sort") ?? "composite",
    dir: p.get("dir") === "asc" ? "asc" : "desc",
  };
}

export function applyFilters(rows: IndexPlayer[], f: Filters): IndexPlayer[] {
  const q = f.q.trim().toLowerCase();
  return rows.filter((r) => {
    if (f.type !== "all" && r.type !== f.type) return false;
    if (q && !r.name.toLowerCase().includes(q) && !r.school.toLowerCase().includes(q)) return false;
    if (f.conferences.length && !f.conferences.includes(r.conference ?? "")) return false;
    if (f.schools.length && !f.schools.includes(r.school_abb)) return false;
    if (f.grades.length && !f.grades.includes(r.grade)) return false;
    if (f.confidence.length && !f.confidence.includes(r.t1_confidence ?? "low")) return false;
    if (f.minSample) {
      const ok =
        r.type === "hitter"
          ? r.sample.pa != null && r.sample.pa >= 50
          : r.sample.ip != null && r.sample.ip >= 20;
      if (!ok) return false;
    }
    if (f.pickMin != null && (r.proj_pick == null || r.proj_pick < f.pickMin)) return false;
    if (f.pickMax != null && (r.proj_pick == null || r.proj_pick > f.pickMax)) return false;
    if (f.probMin != null && (r.mlb_p == null || r.mlb_p < f.probMin)) return false;
    if (f.probMax != null && (r.mlb_p == null || r.mlb_p > f.probMax)) return false;
    return true;
  });
}

type SortAccessor = (r: IndexPlayer) => number | string | null;
const SORTS: Record<string, SortAccessor> = {
  composite: (r) => r.composite,
  name: (r) => r.name,
  school: (r) => r.school_abb,
  pick: (r) => r.proj_pick,
  mlb_p: (r) => r.mlb_p,
  age: (r) => r.age,
  height: (r) => r.height_inches ?? null,
  bmi: (r) => r.bmi ?? null,
  draftability: (r) => r.draftability_score ?? null,
  conference_tier: (r) => r.conference_tier ?? null,
};

export function sortRows(rows: IndexPlayer[], sort: string, dir: "asc" | "desc"): IndexPlayer[] {
  const acc =
    SORTS[sort] ??
    ((r: IndexPlayer) => (r.key_stats[sort] != null ? r.key_stats[sort] : null));
  const mult = dir === "asc" ? 1 : -1;
  return [...rows].sort((a, b) => {
    const va = acc(a);
    const vb = acc(b);
    if (va == null && vb == null) return 0;
    if (va == null) return 1; // nulls last regardless of direction
    if (vb == null) return -1;
    if (typeof va === "string" || typeof vb === "string")
      return mult * String(va).localeCompare(String(vb));
    return mult * (va - vb);
  });
}
