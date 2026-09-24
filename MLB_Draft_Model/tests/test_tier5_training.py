"""Tests for scripts/build_tier5_training.py (Tier 5 WAR training-set build).

TDD RED→GREEN: these tests exercise the pure, filesystem-free functions of the
tier5 training pipeline. File I/O lives in ``main()`` which these tests never
call. Field names match the ACTUAL data shapes (``player_type`` not ``role``,
``Age`` capital A, ``draft_pick`` not ``draft_pick_number``), per the verified
T3/T4 output-shape notes.
"""

from __future__ import annotations

import math

from mlb_draft_dashboard.tier5_contracts import (
    HURDLE_THRESHOLD,
    signed_log_war,
    validate_temporal_no_leakage,
)
from scripts.build_tier5_training import (
    add_features,
    apply_min_sample_filter,
    assign_round_logit_prior,
    assign_split,
    build_round_rates,
    build_training_set,
    compute_label,
    compute_nn_mlb_rates,
    is_two_way,
    load_war_map,
    safe_float,
    verify_v1,
    verify_v2,
    verify_v3,
    verify_v4,
)
from tests.helpers_tier5 import make_mock_war_record


# ---------------------------------------------------------------------------
# Fixtures (ACTUAL expanded_training_set shape)
# ---------------------------------------------------------------------------

def _expanded_record(person_id, player_type, draft_year, draft_round=5,
                     draft_pick=100, pa=None, ip=None, conference="SEC",
                     season=2021, age=21.0, debut=False):
    """Build one expanded-training-set record with ACTUAL field names."""
    rec = {
        "person_id": person_id,
        "player_name": f"Player{person_id}",
        "player_type": player_type,
        "season": season,
        "draft_year": draft_year,
        "draft_pick": draft_pick,
        "draft_round": draft_round,
        "conference": conference,
        "height_inches": 72.0,
        "bmi": 24.0,
        "Age": age,
    }
    if player_type == "hitter":
        rec["PA"] = pa if pa is not None else 200.0
        rec["wOBA"] = 0.400
        rec["OPS"] = 1.000
        rec["AVG"] = 0.300
        rec["SLG"] = 0.500
        rec["BB_pct"] = 0.100
        rec["K_pct"] = 0.180
        rec["ISO"] = 0.200
        rec["wRC_plus"] = 130.0
    else:
        rec["IP"] = ip if ip is not None else 80.0
        rec["ERA"] = 3.50
        rec["FIP"] = 3.80
        rec["WHIP"] = 1.10
        rec["K_per_nine"] = 10.0
        rec["BB_per_nine"] = 3.0
        rec["K_pct"] = 0.25
        rec["BB_pct"] = 0.08
    return rec


def _draft_record(person_id, draft_year, mlb_debut_date=None,
                  position_type="Hitter"):
    return {
        "person_id": person_id,
        "year": draft_year,
        "mlb_debut_date": mlb_debut_date,
        "position_type": position_type,
    }


def _conf_stats():
    return {
        "per_season": {},
        "conference_overall": {},
        "tier_fallback": {"3": {"hitter": {}, "pitcher": {}}},
    }


def _conf_strength():
    return {"SEC": {"strength": 2.98}}


def _war_data(records):
    return {"records": records}


# ---------------------------------------------------------------------------
# load_war_map
# ---------------------------------------------------------------------------

def test_load_war_map_accepts_dict_with_records() -> None:
    recs = [make_mock_war_record(person_id=1, war=3.0, draft_year=2019),
            make_mock_war_record(person_id=2, war=None, draft_year=2019)]
    mapping = load_war_map(_war_data(recs))
    assert mapping[1]["war_years_1_through_5"] == 3.0
    assert mapping[2]["war_years_1_through_5"] is None


def test_load_war_map_accepts_bare_list() -> None:
    recs = [make_mock_war_record(person_id=7, war=1.5, draft_year=2019)]
    mapping = load_war_map(recs)
    assert mapping[7]["war_years_1_through_5"] == 1.5


# ---------------------------------------------------------------------------
# assign_split (honest 2021-2023 overlap)
# ---------------------------------------------------------------------------

def test_assign_split_windowing() -> None:
    assert assign_split(2021) == "train"
    assert assign_split(2022) == "heldout"
    assert assign_split(2023) == "heldout"
    assert assign_split(2024) is None       # inference-only, excluded
    assert assign_split(2025) is None
    assert assign_split(2019) is None       # no college stats pre-2021


# ---------------------------------------------------------------------------
# compute_label (hurdle + signed-log)
# ---------------------------------------------------------------------------

def test_compute_label_hurdle_and_signed_log() -> None:
    assert compute_label(3.0)["meaningfully_productive"] is True
    assert compute_label(2.0)["meaningfully_productive"] is False  # strictly >
    assert compute_label(1.0)["meaningfully_productive"] is False
    assert compute_label(-1.5)["meaningfully_productive"] is False
    assert compute_label(3.0)["signed_log_war"] == signed_log_war(3.0)
    assert compute_label(-2.0)["signed_log_war"] < 0


def test_compute_label_null_war_no_zero_impute() -> None:
    label = compute_label(None)
    assert label["meaningfully_productive"] is False
    assert label["signed_log_war"] is None  # null, NOT 0.0


# ---------------------------------------------------------------------------
# min-sample filters
# ---------------------------------------------------------------------------

def test_min_sample_filter_pa_and_ip() -> None:
    players = [
        _expanded_record(1, "hitter", 2021, pa=200.0),
        _expanded_record(2, "hitter", 2021, pa=49.0),   # below 50 PA
        _expanded_record(3, "pitcher", 2021, ip=80.0),
        _expanded_record(4, "pitcher", 2021, ip=19.0),  # below 20 IP
    ]
    kept, excluded, skipped_ip = apply_min_sample_filter(players, min_pa=50, min_ip=20)
    kept_ids = {p["person_id"] for p in kept}
    assert kept_ids == {1, 3}
    assert excluded == 2
    assert skipped_ip == 0


# ---------------------------------------------------------------------------
# two-way detection
# ---------------------------------------------------------------------------

def test_two_way_detection() -> None:
    # person 1 appears as both hitter and pitcher
    player_types_by_person = {1: {"hitter", "pitcher"}, 2: {"hitter"}}
    two_way_person_ids = set()
    assert is_two_way(1, player_types_by_person, two_way_person_ids) is True
    assert is_two_way(2, player_types_by_person, two_way_person_ids) is False
    # person 3 flagged via draft position_type
    two_way_person_ids = {3}
    assert is_two_way(3, {}, two_way_person_ids) is True


# ---------------------------------------------------------------------------
# add_features (copied Tier 3 derivation)
# ---------------------------------------------------------------------------

def test_add_features_derives_conf_strength_and_adj() -> None:
    records = [
        _expanded_record(1, "hitter", 2021, conference="SEC"),
        _expanded_record(2, "pitcher", 2021, conference="SEC"),
    ]
    out = add_features(records, _conf_stats(), _conf_strength())
    for r in out:
        assert r["conf_strength"] == 2.98
    h = out[0]
    p = out[1]
    assert "wOBA_adj" in h and "OPS_adj" in h
    assert "ERA_adj" in p and "FIP_adj" in p
    assert "K_per_nine_adj" in p and "BB_per_nine_adj" in p


def test_safe_float_nulls() -> None:
    assert safe_float(None) is None
    assert safe_float("3.5") == 3.5
    assert safe_float("garbage") is None


# ---------------------------------------------------------------------------
# round rates + logit + nn rate
# ---------------------------------------------------------------------------

def test_build_round_rates_and_logit() -> None:
    players = [
        _expanded_record(1, "hitter", 2021, draft_round=1),
        _expanded_record(2, "hitter", 2021, draft_round=1),
        _expanded_record(3, "hitter", 2021, draft_round=10),
    ]
    players[0]["has_mlb_debut"] = 1
    players[1]["has_mlb_debut"] = 1
    players[2]["has_mlb_debut"] = 0
    rates = build_round_rates(players)
    assert rates[1]["debut"] == 2 and rates[1]["total"] == 2
    assert rates[1]["rate"] == 1.0
    assert rates[10]["rate"] == 0.0
    # round 1 (better) has higher logit than round 10
    assert rates[1]["logit"] > rates[10]["logit"]


def test_assign_round_logit_prior() -> None:
    players = [_expanded_record(1, "hitter", 2021, draft_round=3)]
    rates = {3: {"logit": 0.7}}
    assign_round_logit_prior(players, rates)
    assert players[0]["round_logit_prior"] == 0.7


def test_compute_nn_mlb_rates_in_unit_interval() -> None:
    players = [
        _expanded_record(i, "hitter", 2021) for i in range(5)
    ]
    for i, p in enumerate(players):
        p["has_mlb_debut"] = 1 if i % 2 == 0 else 0
    rates, _, _ = compute_nn_mlb_rates(players, ["wOBA_adj", "OPS_adj", "K_pct_adj"])
    assert len(rates) == 5
    assert all(0.0 <= r <= 1.0 for r in rates)


# ---------------------------------------------------------------------------
# End-to-end build + verification gates
# ---------------------------------------------------------------------------

def _mini_inputs():
    """Small synthetic pipeline inputs spanning train/heldout/excluded years."""
    expanded = [
        # 2021 train: two hitters (one productive, one not), one pitcher
        _expanded_record(1, "hitter", 2021, draft_round=1, draft_pick=1, pa=200),
        _expanded_record(2, "hitter", 2021, draft_round=10, draft_pick=300, pa=150),
        _expanded_record(3, "pitcher", 2021, draft_round=2, draft_pick=40, ip=90),
        # 2022 heldout
        _expanded_record(4, "hitter", 2022, draft_round=1, draft_pick=5, pa=180),
        # 2024 inference-only → excluded
        _expanded_record(5, "hitter", 2024, draft_round=1, draft_pick=10, pa=220),
    ]
    draft = [
        _draft_record(1, 2021, mlb_debut_date="2022-04-01"),
        _draft_record(2, 2021, mlb_debut_date=None),
        _draft_record(3, 2021, mlb_debut_date="2023-06-01"),
        _draft_record(4, 2022, mlb_debut_date="2024-05-01"),
        _draft_record(5, 2024, mlb_debut_date=None),
    ]
    war = [
        make_mock_war_record(person_id=1, war=5.0, draft_year=2021),
        make_mock_war_record(person_id=2, war=0.5, draft_year=2021),
        make_mock_war_record(person_id=3, war=2.5, draft_year=2021),
        make_mock_war_record(person_id=4, war=None, draft_year=2022),
        make_mock_war_record(person_id=5, war=None, draft_year=2024),
    ]
    return expanded, draft, _war_data(war), _conf_stats(), _conf_strength()


def test_build_training_set_splits_and_fields() -> None:
    expanded, draft, war, cs, cst = _mini_inputs()
    records, meta = build_training_set(war, expanded, draft, cs, cst)

    by_pid = {r["person_id"]: r for r in records}
    # 2024 excluded
    assert 5 not in by_pid
    # splits
    assert by_pid[1]["split"] == "train"
    assert by_pid[4]["split"] == "heldout"
    # hurdle + signed log
    assert by_pid[1]["meaningfully_productive"] is True
    assert by_pid[1]["signed_log_war"] == signed_log_war(5.0)
    assert by_pid[2]["meaningfully_productive"] is False
    # war_years_1_through_5 carried through
    assert by_pid[3]["war_years_1_through_5"] == 2.5
    # actual field names present (player_type / Age / draft_pick)
    assert "player_type" in by_pid[1] and "role" not in by_pid[1]
    assert "Age" in by_pid[1]
    assert "draft_pick" in by_pid[1]
    # Tier 3 features present
    assert "round_logit_prior" in by_pid[1] and "nn_mlb_rate" in by_pid[1]
    assert "conf_strength" in by_pid[1]
    # meta cohort counts
    assert meta["n_train"] == 3
    assert meta["n_heldout"] == 1
    assert meta["n_excluded_inference"] >= 1


def test_verify_v3_no_heldout_in_train() -> None:
    expanded, draft, war, cs, cst = _mini_inputs()
    records, _ = build_training_set(war, expanded, draft, cs, cst)
    # no train record has draft_year > 2021
    assert validate_temporal_no_leakage(records, train_max_year=2021) is True
    result = verify_v3(records, train_max_year=2021)
    assert result["pass"] is True
    assert result["leakage_count"] == 0


def test_verify_v1_hurdle_rate_fraction() -> None:
    expanded, draft, war, cs, cst = _mini_inputs()
    records, _ = build_training_set(war, expanded, draft, cs, cst)
    result = verify_v1(records)
    # meaningful: pids 1 (5.0) and 3 (2.5) → 2 of 4
    assert result["pass"] is True
    assert abs(result["hurdle_rate"] - 0.5) < 1e-9


def test_verify_v2_round_correlation_negative() -> None:
    expanded, draft, war, cs, cst = _mini_inputs()
    records, _ = build_training_set(war, expanded, draft, cs, cst)
    result = verify_v2(records)
    assert result["pass"] is True
    assert result["correlation"] < 0  # better round → higher WAR → negative


def test_verify_v4_distribution_summary() -> None:
    expanded, draft, war, cs, cst = _mini_inputs()
    records, _ = build_training_set(war, expanded, draft, cs, cst)
    result = verify_v4(records)
    assert result["count"] == 3  # war non-null: pids 1,2,3
    assert "median" in result
    assert "positive_count" in result
    assert result["positive_count"] == 3  # 5.0, 0.5, 2.5 all > 0


def test_hurdle_threshold_is_2_0() -> None:
    assert HURDLE_THRESHOLD == 2.0
