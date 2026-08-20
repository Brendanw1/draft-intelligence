#!/usr/bin/env python3
"""Pull Baseball-Reference bulk WAR data and join onto draft person_ids.

Produces ``data/war/war_ground_truth.json``: one record per draftee (2015-2023)
carrying every ``WAR_GROUND_TRUTH_FIELDS`` key. WAR metrics are null (not 0)
for players who never reached MLB.

Pipeline:
  1. Load ``draft_all_picks.json``, extract unique person_ids for 2015..2023.
  2. Crosswalk person_id (MLBAM) -> key_bbref via pybaseball's Chadwick
     register, batched in chunks of 500. Network failures degrade to a partial
     crosswalk, never a crash.
  3. Download ``war_daily_bat.txt`` / ``war_daily_pitch.txt`` bulk files
     (tab-separated, keyed by ``mlb_ID`` numeric MLBAM id with a ``WAR``
     column per year/stint). Retry with backoff; on failure emit
     ``download_status`` and empty/skeleton output.
  4. Parse bulk files, group rows by player, sum stints into season WAR.
  5. Join back to person_id and emit ground-truth records.

``--dry-run`` skips all network I/O and drives the full pipeline on tiny inline
mock data. ``--verify`` runs the V1-V4 quality gates and prints a report.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

BASE = Path(__file__).resolve().parents[1]
DRAFT_PATH = BASE / "data" / "draft" / "draft_all_picks.json"
WAR_DIR = BASE / "data" / "war"
RAW_DIR = WAR_DIR / "raw"
CROSSWALK_PATH = WAR_DIR / "crosswalk_bbref.json"
OUTPUT_PATH = WAR_DIR / "war_ground_truth.json"

WAR_BAT_URL = "https://www.baseball-reference.com/data/war_daily_bat.txt"
WAR_PITCH_URL = "https://www.baseball-reference.com/data/war_daily_pitch.txt"

MIN_DRAFT_YEAR = 2015
MAX_DRAFT_YEAR = 2023
CROSSWALK_CHUNK_SIZE = 500
BBREF_MATCH_MIN = 0.85
DOWNLOAD_RETRIES = 2

try:
    from mlb_draft_dashboard.tier5_contracts import WAR_GROUND_TRUTH_FIELDS
except Exception:  # pragma: no cover - fallback when PYTHONPATH=src is absent
    WAR_GROUND_TRUTH_FIELDS = [
        "person_id", "key_bbref", "career_war_to_date",
        "war_years_1_through_5", "war_year_1_through_3",
        "peak_single_season_war", "seasons_with_positive_war",
        "total_mlb_seasons", "mlb_debut_season", "draft_year",
    ]


# ---------------------------------------------------------------------------
# Pure helpers (network-free, unit-tested)
# ---------------------------------------------------------------------------

def chunk_ids(ids: List, size: int = CROSSWALK_CHUNK_SIZE) -> List[List]:
    return [list(ids[i:i + size]) for i in range(0, len(ids), size)]


def sniff_delimiter(line: str) -> str:
    return "," if line.count(",") > line.count("\t") else "\t"


def split_line(line: str, delimiter: str = "\t") -> List[str]:
    return line.rstrip("\n").split(delimiter)


def parse_tsv_line(line: str) -> List[str]:
    return split_line(line, "\t")


def detect_header(lines: List[str]) -> List[str]:
    for line in lines:
        if not line.strip():
            continue
        return split_line(line, sniff_delimiter(line))
    return []


def filter_rows_to_keys(rows: List[dict], key_set, key_col: str) -> List[dict]:
    return [r for r in rows if r.get(key_col) in key_set]


def season_wars_by_key(rows: List[dict], key_col: str,
                       war_col: str = "WAR", year_col: str = "year_ID") -> Dict:
    out: Dict = {}
    for r in rows:
        k = r.get(key_col)
        if k is None or k == "":
            continue
        try:
            year = int(r.get(year_col))
        except (TypeError, ValueError):
            continue
        try:
            war = float(r.get(war_col))
        except (TypeError, ValueError):
            war = 0.0
        out.setdefault(k, {})
        out[k][year] = out[k].get(year, 0.0) + war
    return out


def career_war_through_year(season_wars: Dict, draft_year: int, n: int) -> float:
    """Sum WAR in seasons (draft_year, draft_year+n]."""
    return sum(w for y, w in season_wars.items() if draft_year < y <= draft_year + n)


def war_year_1_through_3(season_wars: Dict, draft_year: int) -> float:
    return career_war_through_year(season_wars, draft_year, 3)


def war_years_1_through_5(season_wars: Dict, draft_year: int) -> float:
    return career_war_through_year(season_wars, draft_year, 5)


def peak_single_season_war(season_wars: Dict) -> Optional[float]:
    return max(season_wars.values()) if season_wars else None


def seasons_with_positive_war(season_wars: Dict) -> int:
    return sum(1 for w in season_wars.values() if w > 0)


def total_mlb_seasons(season_wars: Dict) -> int:
    return len(season_wars)


def compute_war_metrics(season_wars: Optional[Dict], draft_year: int) -> Dict:
    season_wars = season_wars or {}
    if not season_wars:
        return {
            "career_war_to_date": None,
            "war_years_1_through_5": None,
            "war_year_1_through_3": None,
            "peak_single_season_war": None,
            "seasons_with_positive_war": 0,
            "total_mlb_seasons": 0,
            "mlb_debut_season": None,
        }
    return {
        "career_war_to_date": sum(season_wars.values()),
        "war_years_1_through_5": war_years_1_through_5(season_wars, draft_year),
        "war_year_1_through_3": war_year_1_through_3(season_wars, draft_year),
        "peak_single_season_war": peak_single_season_war(season_wars),
        "seasons_with_positive_war": seasons_with_positive_war(season_wars),
        "total_mlb_seasons": total_mlb_seasons(season_wars),
        "mlb_debut_season": min(season_wars.keys()),
    }


def extract_draftees(records: List[dict], min_year: int, max_year: int) -> List[Tuple]:
    seen = set()
    out = []
    for r in records:
        year = r.get("year")
        pid = r.get("person_id")
        if year is None or pid is None or pid in seen:
            continue
        if min_year <= year <= max_year:
            seen.add(pid)
            out.append((pid, year))
    return out


def debut_season_from_date(date_str: Optional[str]) -> Optional[int]:
    if not date_str or date_str in ("None", ""):
        return None
    try:
        return int(str(date_str)[:4])
    except (TypeError, ValueError):
        return None


def build_ground_truth(draftees: List[Tuple], crosswalk: Dict,
                       season_wars_by_mlbam: Dict,
                       season_wars_by_bbref: Optional[Dict] = None,
                       debut_seasons: Optional[Dict] = None) -> List[dict]:
    season_wars_by_bbref = season_wars_by_bbref or {}
    debut_seasons = debut_seasons or {}
    records = []
    for pid, draft_year in draftees:
        sw = season_wars_by_mlbam.get(pid) or season_wars_by_mlbam.get(str(pid))
        key_bbref = crosswalk.get(pid) or crosswalk.get(str(pid))
        if sw is None and key_bbref:
            sw = season_wars_by_bbref.get(key_bbref)
        rec = {
            "person_id": pid,
            "key_bbref": key_bbref,
            "draft_year": draft_year,
        }
        rec.update(compute_war_metrics(sw, draft_year))
        debut = debut_seasons.get(pid) or debut_seasons.get(str(pid))
        if debut is not None:
            rec["mlb_debut_season"] = debut
        records.append(rec)
    return records


# ---------------------------------------------------------------------------
# Verification gates V1-V4
# ---------------------------------------------------------------------------

def verify_v1(records: List[dict]) -> Dict:
    debuted = [r for r in records if r.get("mlb_debut_season") is not None]
    matched = sum(1 for r in debuted if r.get("key_bbref"))
    total = len(debuted)
    rate = matched / total if total else 0.0
    return {"pass": rate >= BBREF_MATCH_MIN, "debuted_match_rate": rate,
            "matched": matched, "debuted_total": total,
            "all_records": len(records)}


def verify_v2(records: List[dict]) -> Dict:
    missing = sum(1 for r in records
                  if not set(WAR_GROUND_TRUTH_FIELDS).issubset(r.keys()))
    return {"pass": missing == 0, "n": len(records), "missing": missing}


def verify_v3(records: List[dict], draft_debut_seasons: Optional[Dict] = None) -> Dict:
    draft_debut_seasons = draft_debut_seasons or {}
    impossible = 0
    stale = 0
    for r in records:
        if r.get("career_war_to_date") is None:
            continue
        debut = r.get("mlb_debut_season")
        if debut is not None and debut < r.get("draft_year", 0):
            impossible += 1
        pid = r.get("person_id")
        if not (draft_debut_seasons.get(pid) or draft_debut_seasons.get(str(pid))):
            stale += 1
    return {
        "pass": impossible == 0,
        "impossible_war": impossible,
        "stale_snapshot_count": stale,
        "note": "phantom WAR = debut season < draft year (bad join); "
                "stale = WAR rows but draft mlb_debut_date is null",
    }


def verify_v4(records: List[dict]) -> Dict:
    values = [r["war_years_1_through_5"] for r in records
              if r.get("war_years_1_through_5") is not None]
    if not values:
        return {"pass": True, "median": None, "count": 0}
    values = sorted(values)
    n = len(values)
    mid = n // 2
    median = float(values[mid]) if n % 2 == 1 else (values[mid - 1] + values[mid]) / 2.0
    return {"pass": median <= 0, "median": median, "count": n}


# ---------------------------------------------------------------------------
# Network wrappers (not exercised by tests)
# ---------------------------------------------------------------------------

def build_crosswalk(ids: List, chunk_size: int = CROSSWALK_CHUNK_SIZE) -> Dict:
    try:
        import pandas as pd
        import pybaseball
    except Exception as e:
        print(f"WARN: pybaseball unavailable, crosswalk empty: {e}")
        return {}
    crosswalk: Dict = {}
    for chunk in chunk_ids(ids, chunk_size):
        try:
            df = pybaseball.playerid_reverse_lookup(
                [int(i) for i in chunk], key_type="mlbam")
            for _, row in df.iterrows():
                mlbam = row.get("key_mlbam")
                bbref = row.get("key_bbref")
                if mlbam is None or bbref is None:
                    continue
                if pd.isna(bbref) or str(bbref).strip() == "":
                    continue
                crosswalk[int(mlbam)] = str(bbref)
        except Exception as e:
            print(f"WARN: crosswalk chunk failed: {e}")
            time.sleep(1)
    return crosswalk


def download_bulk_file(url: str, dest: Path,
                       retries: int = DOWNLOAD_RETRIES) -> bool:
    import urllib.request
    headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = resp.read()
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(data)
            return True
        except Exception as e:
            print(f"WARN: download attempt {attempt + 1} failed: {e}")
            if attempt < retries:
                time.sleep(2.0 * (attempt + 1))
    return False


def _parse_delimited_lines(lines: List[str]):
    header: List[str] = []
    delimiter = "\t"
    for line in lines:
        if not line.strip():
            continue
        delimiter = sniff_delimiter(line)
        header = split_line(line, delimiter)
        break
    rows: List[dict] = []
    for line in lines[1:]:
        cols = split_line(line, delimiter)
        if not cols or all(c.strip() == "" for c in cols):
            continue
        row = {}
        for i, h in enumerate(header):
            if i < len(cols):
                row[h] = cols[i].strip()
        rows.append(row)
    return header, delimiter, rows


def parse_bulk_file(path: Path) -> List[dict]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    if not lines:
        return []
    return _parse_delimited_lines(lines)[2]


def bbref_key_by_mlbam(rows: List[dict]) -> Dict:
    out: Dict = {}
    for r in rows:
        mid = r.get("mlb_ID")
        pid = r.get("player_ID")
        if mid and pid:
            out[mid] = pid
    return out


def merge_season_war_indices(*indices: Dict) -> Dict:
    merged: Dict = {}
    for idx in indices:
        for k, sw in idx.items():
            if k not in merged:
                merged[k] = dict(sw)
            else:
                for y, w in sw.items():
                    merged[k][y] = merged[k].get(y, 0.0) + w
    return merged


def load_json(path: Path):
    with open(path) as f:
        return json.load(f)


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


# ---------------------------------------------------------------------------
# Mock data for --dry-run
# ---------------------------------------------------------------------------

MOCK_BULK_BAT = (
    "name_common\tmlb_ID\tplayer_ID\tyear_ID\tteam_ID\tstint_ID\tlg_ID\tG\tPA\tWAR\n"
    "Mock Swanson\t621020\tmock621020\t2016\tARI\t1\tNL\t150\t600\t3.2\n"
    "Mock Swanson\t621020\tmock621020\t2017\tARI\t1\tNL\t145\t580\t2.1\n"
    "Mock Swanson\t621020\tmock621020\t2018\tARI\t1\tNL\t140\t560\t1.5\n"
    "Mock Bregman\t608324\tmock608324\t2016\tHOU\t1\tAL\t49\t201\t1.0\n"
    "Mock Bregman\t608324\tmock608324\t2016\tHOU\t2\tAL\t100\t450\t1.0\n"
    "Mock Bregman\t608324\tmock608324\t2017\tHOU\t1\tAL\t155\t620\t4.5\n"
    "Mock Tucker\t663656\tmock663656\t2018\tHOU\t1\tAL\t28\t72\t0.0\n"
    "Mock Tucker\t663656\tmock663656\t2019\tHOU\t1\tAL\t22\t67\t0.0\n"
    "Mock Tate\t622253\tmock622253\t2019\tBAL\t1\tAL\t16\t3\t0.0\n"
    "Mock Rodgers\t663898\tmock663898\t2019\tCOL\t1\tNL\t25\t81\t-0.2\n"
    "Mock Rodgers\t663898\tmock663898\t2020\tCOL\t1\tNL\t7\t21\t-0.1\n"
    "Mock Jay\t664079\tmock664079\t2019\tMIN\t1\tAL\t10\t2\t-0.5\n"
)


def mock_bulk_rows() -> List[dict]:
    return parse_bulk_file_lines(MOCK_BULK_BAT)


def parse_bulk_file_lines(text: str) -> List[dict]:
    return _parse_delimited_lines(text.splitlines())[2]


# ---------------------------------------------------------------------------
# Verification report
# ---------------------------------------------------------------------------

def run_verify(records: List[dict], draft_debut_seasons: Optional[Dict] = None) -> Dict:
    gates = {
        "V1_bbref_match": verify_v1(records),
        "V2_structure": verify_v2(records),
        "V3_no_phantom_war": verify_v3(records, draft_debut_seasons),
        "V4_median_lte_zero": verify_v4(records),
    }
    return gates


def print_verify_report(gates: Dict) -> None:
    print("\n" + "=" * 60)
    print("VERIFICATION REPORT (V1-V4)")
    print("=" * 60)
    for name, g in gates.items():
        status = "PASS" if g["pass"] else "WARN"
        print(f"  {name:<22s} {status}  {g}")
    all_pass = all(g["pass"] for g in gates.values())
    print("-" * 60)
    print(f"  OVERALL: {'PASS' if all_pass else 'WARN (review gates above)'}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Pull BRef bulk WAR ground truth.")
    parser.add_argument("--draft-path", type=Path, default=DRAFT_PATH)
    parser.add_argument("--war-dir", type=Path, default=WAR_DIR)
    parser.add_argument("--crosswalk-out", type=Path, default=CROSSWALK_PATH)
    parser.add_argument("--dry-run", action="store_true",
                        help="Skip network; drive pipeline on inline mock data.")
    parser.add_argument("--verify", action="store_true",
                        help="Run V1-V4 gates and print a report.")
    args = parser.parse_args(argv)

    raw_dir = args.war_dir / "raw"
    output_path = args.war_dir / "war_ground_truth.json"
    if args.dry_run:
        output_path = args.war_dir / "war_ground_truth.dryrun.json"
    crosswalk_out = args.crosswalk_out

    draft = load_json(args.draft_path)
    draftees = extract_draftees(draft, MIN_DRAFT_YEAR, MAX_DRAFT_YEAR)
    print(f"Draftees (2015-2023): {len(draftees)} unique person_ids")

    debut_seasons = {}
    for r in draft:
        pid = r.get("person_id")
        if pid is not None:
            debut_seasons[pid] = debut_season_from_date(r.get("mlb_debut_date"))

    if args.dry_run:
        crosswalk = {pid: f"bbref{pid}" for pid, _ in draftees}
        rows = mock_bulk_rows()
        mlbam_idx = season_wars_by_key(rows, key_col="mlb_ID")
        bbref_idx = season_wars_by_key(rows, key_col="player_ID")
        download_status = "SKIPPED_DRY_RUN"
        header_cols = detect_header(MOCK_BULK_BAT.splitlines())
    else:
        ids = [pid for pid, _ in draftees]
        print(f"Building crosswalk for {len(ids)} ids (chunks of {CROSSWALK_CHUNK_SIZE})...")
        crosswalk = build_crosswalk(ids)
        print(f"Chadwick crosswalk: {len(crosswalk)} mappings")

        bat_ok = download_bulk_file(WAR_BAT_URL, raw_dir / "war_daily_bat.txt")
        pitch_ok = download_bulk_file(WAR_PITCH_URL, raw_dir / "war_daily_pitch.txt")
        header_cols = []
        bulk_bbref: Dict = {}
        if bat_ok or pitch_ok:
            download_status = "OK"
            mlbam_idx, bbref_idx = {}, {}
            for name in ("war_daily_bat.txt", "war_daily_pitch.txt"):
                p = raw_dir / name
                if not p.exists():
                    continue
                lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
                if not header_cols:
                    header_cols = detect_header(lines)
                rows = _parse_delimited_lines(lines)[2]
                mlbam_idx = merge_season_war_indices(
                    mlbam_idx, season_wars_by_key(rows, key_col="mlb_ID"))
                bbref_idx = merge_season_war_indices(
                    bbref_idx, season_wars_by_key(rows, key_col="player_ID"))
                bulk_bbref = {**bulk_bbref, **bbref_key_by_mlbam(rows)}
        else:
            download_status = "SKIPPED_NETWORK_UNAVAILABLE"
            mlbam_idx, bbref_idx = {}, {}

        # Enrich crosswalk with player_ID from the bulk files (bulk fills gaps,
        # Chadwick wins on conflict), then cache.
        crosswalk = {**bulk_bbref, **crosswalk}
        write_json(crosswalk_out, crosswalk)
        print(f"Crosswalk cached: {crosswalk_out} ({len(crosswalk)} mappings "
              f"= {len(bulk_bbref)} bulk + Chadwick)")

    records = build_ground_truth(
        draftees, crosswalk, mlbam_idx, bbref_idx, debut_seasons)

    payload = {
        "download_status": download_status,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "n_draftees": len(draftees),
        "n_records": len(records),
        "n_with_war": sum(1 for r in records if r["career_war_to_date"] is not None),
        "n_debuted": sum(1 for r in records if r["mlb_debut_season"] is not None),
        "crosswalk_size": len(crosswalk),
        "detected_bulk_columns": header_cols,
        "records": records,
    }
    write_json(output_path, payload)
    print(f"Wrote {output_path} ({len(records)} records, "
          f"status={download_status})")

    if args.verify:
        gates = run_verify(records, debut_seasons)
        print_verify_report(gates)
    return 0


if __name__ == "__main__":
    sys.exit(main())
