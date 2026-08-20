"""Tests for scripts/pull_war_ground_truth.py (Tier 5 WAR ground-truth pull).

TDD RED→GREEN: these tests exercise the pure, network-free functions of the
WAR-pull pipeline so the whole thing is verifiable offline. Network I/O lives
behind thin wrappers that these tests never call.
"""

from __future__ import annotations

from mlb_draft_dashboard.tier5_contracts import WAR_GROUND_TRUTH_FIELDS
from scripts.pull_war_ground_truth import (
    build_ground_truth,
    career_war_through_year,
    chunk_ids,
    compute_war_metrics,
    detect_header,
    extract_draftees,
    filter_rows_to_keys,
    parse_bulk_file_lines,
    parse_tsv_line,
    peak_single_season_war,
    season_wars_by_key,
    seasons_with_positive_war,
    sniff_delimiter,
    total_mlb_seasons,
    verify_v1,
    verify_v2,
    verify_v3,
    verify_v4,
    war_year_1_through_3,
    war_years_1_through_5,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _row(mlb_id, year, war, bbref=None):
    """Build a bulk-file row dict keyed by mlb_ID."""
    return {
        "name_common": f"Player{mlb_id}",
        "mlb_ID": str(mlb_id),
        "player_ID": bbref or f"bbref{mlb_id}",
        "year_ID": str(year),
        "team_ID": "ARI",
        "stint_ID": "1",
        "lg_ID": "NL",
        "G": "150",
        "PA": "600",
        "WAR": str(war),
    }


def _draftees():
    """(person_id, draft_year) pairs for a 2015 draftee and a 2019 draftee."""
    return [(621020, 2015), (688130, 2019), (999999, 2022)]


# ---------------------------------------------------------------------------
# Crosswalk chunking
# ---------------------------------------------------------------------------

def test_chunk_ids_batches_at_500() -> None:
    ids = list(range(1200))
    chunks = chunk_ids(ids, size=500)
    assert [len(c) for c in chunks] == [500, 500, 200]
    assert chunks[0][0] == 0
    assert chunks[2][-1] == 1199


def test_chunk_ids_exact_multiple() -> None:
    chunks = chunk_ids(list(range(1000)), size=500)
    assert [len(c) for c in chunks] == [500, 500]


# ---------------------------------------------------------------------------
# TSV parsing / header detection
# ---------------------------------------------------------------------------

def test_detect_header_and_parse_tsv() -> None:
    line = "name_common\tmlb_ID\tplayer_ID\tyear_ID\tteam_ID\tWAR\n"
    header = detect_header([line])
    assert "mlb_ID" in header and "player_ID" in header and "WAR" in header
    cols = parse_tsv_line("a\tb\tc")
    assert cols == ["a", "b", "c"]


def test_filter_rows_to_keys() -> None:
    rows = [_row(1, 2020, 1.0), _row(2, 2020, 2.0), _row(3, 2020, 3.0)]
    kept = filter_rows_to_keys(rows, {"1", "3"}, key_col="mlb_ID")
    assert {r["mlb_ID"] for r in kept} == {"1", "3"}


def test_sniff_delimiter_prefers_comma() -> None:
    assert sniff_delimiter("a,b,c") == ","
    assert sniff_delimiter("a\tb\tc") == "\t"


def test_parse_comma_delimited_bulk_real_columns() -> None:
    # BRef war_daily_*.txt is comma-separated with mlb_ID/player_ID/year_ID/WAR
    text = (
        "name_common,age,mlb_ID,player_ID,year_ID,team_ID,stint_ID,lg_ID,WAR\n"
        "Dansby Swanson,22,621020,swansda01,2016,ATL,1,NL,1.16\n"
    )
    rows = parse_bulk_file_lines(text)
    assert rows[0]["mlb_ID"] == "621020"
    assert rows[0]["player_ID"] == "swansda01"
    assert rows[0]["year_ID"] == "2016"
    assert rows[0]["WAR"] == "1.16"


def test_season_wars_sums_stints() -> None:
    # Same player, same year, two stints → summed
    rows = [_row(621020, 2016, 1.0), _row(621020, 2016, 0.5), _row(621020, 2017, 2.0)]
    idx = season_wars_by_key(rows, key_col="mlb_ID")
    assert idx["621020"][2016] == 1.5
    assert idx["621020"][2017] == 2.0


# ---------------------------------------------------------------------------
# WAR metric computation
# ---------------------------------------------------------------------------

def test_war_metric_windows() -> None:
    season_wars = {
        2016: 1.0, 2017: 2.0, 2018: 3.0, 2019: 4.0,
        2020: 5.0, 2021: 6.0, 2022: 7.0, 2023: 8.0,
    }
    draft_year = 2015
    # career WAR through year N (years draft_year+1 .. draft_year+N)
    assert career_war_through_year(season_wars, draft_year, 3) == 6.0   # 2016-2018
    assert career_war_through_year(season_wars, draft_year, 5) == 15.0  # 2016-2020
    assert career_war_through_year(season_wars, draft_year, 8) == 36.0  # 2016-2023
    assert war_year_1_through_3(season_wars, draft_year) == 6.0
    assert war_years_1_through_5(season_wars, draft_year) == 15.0


def test_peak_and_season_counts() -> None:
    season_wars = {2016: -0.5, 2017: 0.0, 2018: 2.5, 2019: 1.0}
    assert peak_single_season_war(season_wars) == 2.5
    assert seasons_with_positive_war(season_wars) == 2  # 2.5 and 1.0 (>0)
    assert total_mlb_seasons(season_wars) == 4


def test_compute_war_metrics_never_debuted() -> None:
    metrics = compute_war_metrics({}, draft_year=2019)
    assert metrics["career_war_to_date"] is None
    assert metrics["war_years_1_through_5"] is None
    assert metrics["war_year_1_through_3"] is None
    assert metrics["peak_single_season_war"] is None
    assert metrics["seasons_with_positive_war"] == 0
    assert metrics["total_mlb_seasons"] == 0
    assert metrics["mlb_debut_season"] is None


def test_compute_war_metrics_debuted() -> None:
    season_wars = {2016: 1.0, 2017: 2.0, 2018: -0.5}
    metrics = compute_war_metrics(season_wars, draft_year=2015)
    assert metrics["mlb_debut_season"] == 2016
    assert metrics["total_mlb_seasons"] == 3
    assert metrics["career_war_to_date"] == 2.5
    assert metrics["war_year_1_through_3"] == 2.5
    assert metrics["war_years_1_through_5"] == 2.5


# ---------------------------------------------------------------------------
# Ground-truth build + joins
# ---------------------------------------------------------------------------

def test_build_ground_truth_join_and_fields() -> None:
    crosswalk = {621020: "swansda01", 688130: "mock688130", 999999: "mock999999"}
    mlbam_idx = {"621020": {2016: 3.0, 2017: 2.0}, "688130": {2020: -0.2}}
    records = build_ground_truth(_draftees(), crosswalk, mlbam_idx)
    by_pid = {r["person_id"]: r for r in records}
    assert len(records) == 3

    # Debuted player gets WAR + bbref key
    r = by_pid[621020]
    assert r["key_bbref"] == "swansda01"
    assert r["career_war_to_date"] == 5.0
    assert r["draft_year"] == 2015
    assert r["mlb_debut_season"] == 2016

    # Never-debuted player (no rows) gets null WAR, not zero
    r_never = by_pid[999999]
    assert r_never["career_war_to_date"] is None
    assert r_never["war_years_1_through_5"] is None
    assert r_never["total_mlb_seasons"] == 0

    # Every record carries all WAR_GROUND_TRUTH_FIELDS
    for rec in records:
        assert set(WAR_GROUND_TRUTH_FIELDS).issubset(rec.keys())


def test_extract_draftees_uses_year_not_draft_year() -> None:
    raw = [
        {"person_id": 1, "year": 2015, "mlb_debut_date": "2016-01-01"},
        {"person_id": 2, "year": 2024, "mlb_debut_date": None},   # excluded
        {"person_id": 3, "year": 2023, "mlb_debut_date": None},
        {"person_id": 4, "year": 2014, "mlb_debut_date": None},   # excluded
    ]
    out = extract_draftees(raw, min_year=2015, max_year=2023)
    assert out == [(1, 2015), (3, 2023)]


# ---------------------------------------------------------------------------
# V1-V4 verification gates
# ---------------------------------------------------------------------------

def test_verify_v1_85pct_bbref_match() -> None:
    draftees = [(i, 2019) for i in range(10)]
    debut_seasons = {i: 2020 for i in range(6)}  # 6 of 10 debuted
    # 5 of 6 debuted matched -> 83.3% < 85% -> fail
    crosswalk = {i: f"b{i}" for i in range(5)}
    records = build_ground_truth(draftees, crosswalk, {}, debut_seasons=debut_seasons)
    good = verify_v1(records)
    assert good["debuted_total"] == 6
    assert good["matched"] == 5
    assert good["pass"] is False
    # 6 of 6 debuted matched -> 100% -> pass
    crosswalk2 = {i: f"b{i}" for i in range(6)}
    records2 = build_ground_truth(draftees, crosswalk2, {}, debut_seasons=debut_seasons)
    assert verify_v1(records2)["pass"] is True


def test_verify_v2_structure() -> None:
    records = build_ground_truth(_draftees(), {621020: "x"}, {"621020": {2016: 1.0}})
    assert verify_v2(records)["pass"] is True
    # Corrupt one record → fail
    records[0].pop("war_years_1_through_5")
    assert verify_v2(records)["pass"] is False


def test_verify_v3_no_phantom_war() -> None:
    # WAR debut season before draft year -> impossible (bad join)
    bad = build_ground_truth([(1, 2019)], {1: "x"}, {"1": {2015: 2.0}},
                             debut_seasons={})
    result = verify_v3(bad, draft_debut_seasons={})
    assert result["pass"] is False
    assert result["impossible_war"] == 1
    # WAR debut season after draft year -> clean
    good = build_ground_truth([(1, 2019)], {1: "x"}, {"1": {2020: 2.0}},
                              debut_seasons={})
    assert verify_v3(good, draft_debut_seasons={})["pass"] is True
    # WAR without draft mlb_debut_date is stale-snapshot, not a bad join
    assert verify_v3(good, draft_debut_seasons={})["stale_snapshot_count"] == 1


def test_verify_v4_median_lte_zero() -> None:
    # Zero-inflated / negative-heavy WAR → median ≤ 0
    records = build_ground_truth(
        [(i, 2019) for i in range(6)],
        {i: f"b{i}" for i in range(6)},
        {
            0: {2020: 0.0}, 1: {2020: -0.5}, 2: {2020: 0.3},
            3: {2020: 0.0}, 4: {2020: -1.0}, 5: {},  # 5 never-debuted → null
        },
    )
    result = verify_v4(records)
    assert result["pass"] is True
    assert result["count"] == 5  # 5 non-null
    assert result["median"] == 0.0
