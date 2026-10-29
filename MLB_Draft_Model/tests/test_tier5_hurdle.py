"""Tests for scripts/train_tier5_hurdle.py (Tier 5 Stage A hurdle classifier).

TDD RED→GREEN: exercise the pure, filesystem-free functions of the hurdle
trainer. File I/O and artifact persistence live in ``main()`` which these
tests never call.

Covers the Stage A contract:
  * Elastic Net logistic (saga, max_iter=2000) — never XGBoost.
  * Hitter/pitcher feature sets copied verbatim from Tier 3 (no milb_year1_wOBA
    for hitters).
  * Temporal split: train = draft_year 2021, heldout = 2022-2023 (honest
    overlap window). No 2022+ leakage into train.
  * GroupKFold by person_id, Brier, bootstrap AUC CI (n_iter), calibration
    deciles, AUC gate >0.65 WARNs (never crashes).
"""

from __future__ import annotations

import numpy as np
import pytest

from scripts.train_tier5_hurdle import (
    HITTER_T5_FEATURES,
    HURDLE_AUC_GATE,
    PITCHER_T5_FEATURES,
    bootstrap_auc_ci,
    build_matrix,
    calibration_deciles,
    check_auc_gate,
    cv_auc_brier,
    evaluate,
    fit_hurdle_model,
    split_by_role,
    temporal_split,
    train_hurdle,
    tune_l1_ratio,
)


# ---------------------------------------------------------------------------
# Feature-set contract (copied verbatim from Tier 3)
# ---------------------------------------------------------------------------

def test_hitter_features_match_tier3_and_exclude_milb() -> None:
    assert HITTER_T5_FEATURES == [
        "Age", "conf_strength",
        "wOBA_adj", "OPS_adj", "BB_pct_adj", "K_pct_adj",
        "height_inches", "bmi", "round_logit_prior", "nn_mlb_rate",
    ]
    assert "milb_year1_wOBA" not in HITTER_T5_FEATURES
    assert len(HITTER_T5_FEATURES) == 10


def test_pitcher_features_match_tier3() -> None:
    assert PITCHER_T5_FEATURES == [
        "Age", "conf_strength",
        "ERA_adj", "FIP_adj", "K_per_nine_adj", "BB_per_nine_adj",
        "height_inches", "bmi", "round_logit_prior", "nn_mlb_rate",
    ]
    assert len(PITCHER_T5_FEATURES) == 10


def test_hurdle_gate_is_065() -> None:
    assert HURDLE_AUC_GATE == 0.65


# ---------------------------------------------------------------------------
# Fixture factory (ACTUAL expanded_training_set field names)
# ---------------------------------------------------------------------------

def _mock_record(person_id: int, player_type: str, split: str,
                 productive: bool = False) -> dict:
    """One Tier 5 training record with ACTUAL field names (player_type, Age)."""
    rec = {
        "person_id": person_id,
        "player_name": f"Player{person_id}",
        "player_type": player_type,
        "split": split,
        "meaningfully_productive": productive,
        "Age": 21.0,
        "conf_strength": 1.0,
        "height_inches": 72.0,
        "bmi": 24.0,
        "round_logit_prior": -1.0,
        "nn_mlb_rate": 0.2,
        "draft_year": 2021 if split == "train" else 2022,
        "draft_round": 5,
        "draft_pick": 150,
    }
    if player_type == "hitter":
        rec.update({
            "wOBA_adj": 0.15, "OPS_adj": 0.30,
            "BB_pct_adj": 0.04, "K_pct_adj": -0.06,
        })
    else:
        rec.update({
            "ERA_adj": -0.2, "FIP_adj": -0.1,
            "K_per_nine_adj": 1.0, "BB_per_nine_adj": -0.5,
        })
    return rec


def make_mock_records(n: int = 100, seed: int = 7) -> list:
    """Deterministic 100-row fixture: half hitters, half pitchers, ~19% pos."""
    records = []
    for i in range(n):
        player_type = "hitter" if i % 2 == 0 else "pitcher"
        split = "train" if i % 3 != 0 else "heldout"
        productive = (i % 10 == 0) or (i % 10 == 5)
        records.append(_mock_record(i, player_type, split, productive))
    return records


@pytest.fixture(scope="module")
def mock_records() -> list:
    return make_mock_records(100)


# ---------------------------------------------------------------------------
# Temporal split (honest 2021 train / 2022-2023 heldout)
# ---------------------------------------------------------------------------

def test_temporal_split_partitions_by_split_field(mock_records) -> None:
    train, heldout = temporal_split(mock_records)
    assert len(train) + len(heldout) == len(mock_records)
    assert all(r["split"] == "train" for r in train)
    assert all(r["split"] == "heldout" for r in heldout)
    # no 2022+ leakage into train (honest window: train is draft_year 2021 only)
    assert all(r.get("draft_year", 0) == 2021 for r in train)
    assert all(r.get("draft_year", 0) in (2022, 2023) for r in heldout)


def test_temporal_split_no_overlap(mock_records) -> None:
    train, heldout = temporal_split(mock_records)
    train_ids = {r["person_id"] for r in train}
    heldout_ids = {r["person_id"] for r in heldout}
    assert train_ids.isdisjoint(heldout_ids)


# ---------------------------------------------------------------------------
# Hitter/pitcher split (never pooled)
# ---------------------------------------------------------------------------

def test_split_by_role_is_disjoint_and_never_pooled(mock_records) -> None:
    pools = split_by_role(mock_records)
    assert set(pools.keys()) == {"hitter", "pitcher"}
    assert all(r["player_type"] == "hitter" for r in pools["hitter"])
    assert all(r["player_type"] == "pitcher" for r in pools["pitcher"])
    hitter_ids = {r["person_id"] for r in pools["hitter"]}
    pitcher_ids = {r["person_id"] for r in pools["pitcher"]}
    assert hitter_ids.isdisjoint(pitcher_ids)
    assert len(pools["hitter"]) + len(pools["pitcher"]) == len(mock_records)


# ---------------------------------------------------------------------------
# build_matrix (feature selection + label + group)
# ---------------------------------------------------------------------------

def test_build_matrix_shapes_and_label(mock_records) -> None:
    hitters = [r for r in mock_records if r["player_type"] == "hitter"]
    X, y, groups = build_matrix(hitters, HITTER_T5_FEATURES)
    assert X.shape == (len(hitters), len(HITTER_T5_FEATURES))
    assert len(y) == len(hitters)
    assert len(groups) == len(hitters)
    # label is meaningfully_productive (bool → int)
    assert set(np.unique(y)).issubset({0, 1})
    # group is person_id (GroupKFold grouping key)
    assert list(groups) == [r["person_id"] for r in hitters]


def test_build_matrix_uses_only_declared_features() -> None:
    # A record carrying forbidden extras must not bleed into the matrix.
    rec = _mock_record(1, "hitter", "train", productive=False)
    rec["milb_year1_wOBA"] = 0.420       # must be ignored for hitters
    rec["avg_velo"] = 92.5               # TrackMan field — not a hurdle feature
    X, _, _ = build_matrix([rec], HITTER_T5_FEATURES)
    assert X.shape[1] == len(HITTER_T5_FEATURES)
    # column 0 = Age (23-style capital A field maps to float)
    assert X[0][0] == 21.0


# ---------------------------------------------------------------------------
# Model contract: Elastic Net logistic, saga, max_iter 2000
# ---------------------------------------------------------------------------

def test_fit_model_is_elastic_net_saga() -> None:
    X, y, _ = build_matrix(make_mock_records(80), HITTER_T5_FEATURES)
    model = fit_hurdle_model(X, y, l1_ratio=0.3)
    # Never XGBoost: must be a scikit-learn LogisticRegression with elasticnet.
    assert model.__class__.__name__ == "LogisticRegression"
    assert model.penalty == "elasticnet"
    assert model.solver == "saga"
    assert model.max_iter == 2000


def test_cv_auc_brier_returns_sane_values() -> None:
    X, y, groups = build_matrix(make_mock_records(80), HITTER_T5_FEATURES)
    result = cv_auc_brier(X, y, groups, l1_ratio=0.3, n_splits=4)
    assert "auc_mean" in result and "brier_mean" in result
    assert 0.0 <= result["brier_mean"] <= 1.0
    # auc_mean may be nan when a fold has a single class; allow but don't crash
    assert result["brier_mean"] == result["brier_mean"]


def test_tune_l1_ratio_picks_from_grid() -> None:
    X, y, groups = build_matrix(make_mock_records(80), HITTER_T5_FEATURES)
    grid = (0.1, 0.5, 0.9)
    best = tune_l1_ratio(X, y, groups, grid=grid)
    assert best in grid


# ---------------------------------------------------------------------------
# Brier + evaluate
# ---------------------------------------------------------------------------

def test_evaluate_brier_in_unit_interval() -> None:
    X, y, _ = build_matrix(make_mock_records(80), HITTER_T5_FEATURES)
    model = fit_hurdle_model(X, y, l1_ratio=0.3)
    result = evaluate(model, X, y)
    assert 0.0 <= result["brier"] <= 1.0
    assert "auc" in result


# ---------------------------------------------------------------------------
# Bootstrap AUC CI (n_iter honored, low <= auc <= high)
# ---------------------------------------------------------------------------

def test_bootstrap_auc_ci_bounds_and_iters() -> None:
    records = make_mock_records(80)
    X, y, _ = build_matrix(records, HITTER_T5_FEATURES)
    model = fit_hurdle_model(X, y, l1_ratio=0.3)
    ci = bootstrap_auc_ci(model, X, y, n_iter=60, seed=42)
    assert ci["n_iter"] == 60
    assert ci["low"] <= ci["auc"] <= ci["high"]
    assert 0.0 <= ci["low"] <= 1.0 and 0.0 <= ci["high"] <= 1.0


# ---------------------------------------------------------------------------
# Calibration deciles (10 bins, observed in [0,1], sums to n)
# ---------------------------------------------------------------------------

def test_calibration_deciles_structure() -> None:
    records = make_mock_records(80)
    X, y, _ = build_matrix(records, HITTER_T5_FEATURES)
    model = fit_hurdle_model(X, y, l1_ratio=0.3)
    deciles = calibration_deciles(model, X, y, n_bins=10)
    assert len(deciles) <= 10
    total = sum(d["n"] for d in deciles)
    assert total == len(y)
    for d in deciles:
        assert 0.0 <= d["pred_mean"] <= 1.0
        assert 0.0 <= d["obs_mean"] <= 1.0
        assert d["n"] >= 0


# ---------------------------------------------------------------------------
# AUC gate: >0.65 WARN, never crash
# ---------------------------------------------------------------------------

def test_auc_gate_warns_below_threshold() -> None:
    result = check_auc_gate(0.52)
    assert result["status"] == "WARN"
    assert result["gate"] == HURDLE_AUC_GATE


def test_auc_gate_passes_above_threshold() -> None:
    result = check_auc_gate(0.72)
    assert result["status"] == "PASS"


def test_auc_gate_does_not_raise_on_nan() -> None:
    # Even a degenerate (nan) AUC must warn, not crash.
    result = check_auc_gate(float("nan"))
    assert result["status"] == "WARN"


# ---------------------------------------------------------------------------
# End-to-end train_hurdle on 100 mock rows (no crash, gate present)
# ---------------------------------------------------------------------------

def test_train_hurdle_end_to_end(mock_records) -> None:
    result = train_hurdle(
        mock_records,
        role="hitter",
        features=HITTER_T5_FEATURES,
        l1_grid=(0.3, 0.5),
        n_splits=4,
        n_bootstrap=30,
    )
    for key in ("role", "features", "n_train", "n_heldout",
                "l1_ratio", "heldout", "bootstrap", "calibration", "gate"):
        assert key in result, f"missing key {key}"
    assert result["role"] == "hitter"
    assert result["features"] == HITTER_T5_FEATURES
    # gate is present and has a status (WARN/PASS), never raises
    assert result["gate"]["status"] in {"WARN", "PASS"}
    # bootstrap CI is bounded
    assert result["bootstrap"]["low"] <= result["bootstrap"]["auc"] <= result["bootstrap"]["high"]
