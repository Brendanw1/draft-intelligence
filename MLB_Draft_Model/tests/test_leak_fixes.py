"""One check per URGENT-FIXES.md bug: each fails if its leak comes back."""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from mlb_draft_dashboard.leakfree import OffsetLogit, crossfit_encode, prior_logit, round_rates
from scripts import backtest_tier5 as bt
from scripts import train_tier2_full as t2
from scripts import train_tier3_mlb_arrival as t3

rng = np.random.default_rng(0)


def test_1_round_prior_is_out_of_fold_and_a_true_offset():
    # Pure-noise labels in tiny rounds: an in-sample prior "predicts" them, an out-of-fold one cannot.
    n = 600
    rounds, y = np.repeat(np.arange(n // 3), 3), rng.integers(0, 2, n)
    S = rng.normal(size=(n, 3))
    leaky = prior_logit(round_rates(rounds, y, m=0), rounds)
    oof, _ = crossfit_encode(rounds, S, y)
    assert roc_auc_score(y, leaky) > 0.75
    assert abs(roc_auc_score(y, oof) - 0.5) < 0.1
    # Offset coefficient is fixed at 1 and inputs are scaled (rescaling a column changes nothing).
    feats = ["a", "round_logit_prior"]
    X = np.c_[rng.normal(size=n), rng.normal(size=n)]
    yy = (rng.random(n) < 1 / (1 + np.exp(-X[:, 1]))).astype(int)
    p1 = OffsetLogit(feats).fit(X, yy).predict_proba(X)[:, 1]
    X2 = X.copy(); X2[:, 0] *= 1000
    assert np.allclose(p1, OffsetLogit(feats).fit(X2, yy).predict_proba(X2)[:, 1], atol=1e-4)


def test_2_nn_rate_per_fold_and_censoring():
    n = 400
    S, y = rng.normal(size=(n, 4)), rng.integers(0, 2, n)
    S = np.repeat(S, 2, axis=0) + 1e-9 * rng.normal(size=(2 * n, 4))  # each player's twin sits next to him
    y = np.repeat(y, 2)
    _, rate = crossfit_encode(np.zeros(2 * n), S, y)
    assert roc_auc_score(y, rate) < 0.8  # a row never neighbours itself (twins may share a fold)
    assert t3.label_horizon("2025-05-01", 2021) == 0 and t3.label_horizon("2024-09-01", 2021) == 1
    assert max(t3.TRAIN_CLASSES) + t3.HORIZON <= t3.LAST_SEASON


def test_3_tier2_calibration_folds_are_time_ordered_and_player_grouped(monkeypatch):
    seen = []

    class Rec:
        def fit(self, X, y):
            self.rows = X[:, 0]; return self

        def predict_proba(self, X):
            seen.append((set(self.rows), set(X[:, 0]))); return np.c_[1 - X[:, 1], X[:, 1]]

    monkeypatch.setattr(t2, "make_xgb", lambda y: Rec())
    seasons = np.repeat([2021, 2022, 2023], 20)
    groups = np.array([f"p{i % 25}" for i in range(60)])  # players recur across seasons
    X = np.c_[np.arange(60), rng.random(60)]
    y = rng.integers(0, 2, 60)
    t2.forward_oof(X, y, groups, seasons, [2022, 2023])
    for train_rows, test_rows in seen:
        tr, te = np.array(sorted(train_rows), int), np.array(sorted(test_rows), int)
        assert seasons[tr].max() < seasons[te].min()
        assert not set(groups[tr]) & set(groups[te])


def test_4_backtest_uses_matured_classes_and_keeps_missing_war_missing():
    df = pd.DataFrame({"draft_year": [2021, 2021, 2021, 2022],
                       "war_years_1_through_5": [3.0, None, None, 1.0],
                       "has_mlb_debut": [1, 0, 1, 1]})
    m = bt.matured(df, last_season=2026)
    assert m["draft_year"].tolist() == [2021, 2021, 2021]
    reached, war = bt.actual_war(m)
    assert reached.tolist() == [1, 0, 0]
    assert war[0] == 3.0 and war[1] == 0.0 and np.isnan(war[2])


def test_5_tier5_is_p_times_e_and_ignores_stored_leaky_columns():
    n = 200
    df = pd.DataFrame({c: rng.normal(size=n) for c in set(bt.FEATURES["hitter"] + bt.SIM_STATS["hitter"])})
    df["reached"] = rng.integers(0, 2, n)
    df["war"] = np.where(df["reached"] == 1, rng.normal(1, 1, n), 0.0)
    df["round_logit_prior"] = np.where(df["reached"] == 1, 5.0, -5.0)  # perfectly leaky stored column
    df["nn_mlb_rate"] = df["reached"].astype(float)
    df["draft_round"] = rng.integers(1, 21, n)
    df["person_id"] = np.arange(n)
    p, e, ew = bt.oof_expected_war(df, "hitter")
    assert np.allclose(ew, p * e)
    assert roc_auc_score(df["reached"], p) < 0.7
