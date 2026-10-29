#!/usr/bin/env python3
"""
Tier 5 backtest on matured draft classes only (rewritten 2026-10-29, URGENT-FIXES.md #4 and #5).

Expected WAR = P(reach MLB) x E[WAR years 1-5 | reached], scored against the actual WAR.
  * Matured cohort: a class counts only if all five post-draft seasons are complete
    (draft_year + 5 <= LAST_SEASON). College stats start in 2021, so today that is the 2021 class alone;
    the 2015-2020 classes have WAR but no college stats, and 2022+ are right-censored.
  * The cohort is also the training class, so every number here is player-grouped out-of-fold.
    round_logit_prior and nn_mlb_rate are rebuilt inside each fold (the stored columns leak the label).
  * WAR: a player with no MLB debut has 0 WAR (true for a matured class). A player who debuted but has no
    WAR record is MISSING and is dropped, never set to 0.
  * The hurdle is scored on the label it is trained on (reached MLB); WAR > 2 is scored as a ranking of E[WAR].
  * No try/except: a failure raises.
Tier 5 stays out of the composite and the site until this passes (R2 >= 0 vs the median and vs the mean).

Usage:
    PYTHONPATH=src python scripts/backtest_tier5.py [--report] [--enforce]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, mean_squared_error, r2_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from mlb_draft_dashboard.leakfree import crossfit_encode, encode

BASE = Path(__file__).resolve().parents[1]
TRAINING_DATA_PATH = BASE / "data/training/tier5_training_set.json"
LAST_SEASON = 2026
YEARS = 5
PRODUCTIVE_WAR = 2.0  # the old hurdle label, kept as a ranking check

FEATURES = {
    "hitter": ["Age", "conf_strength", "wOBA_adj", "OPS_adj", "BB_pct_adj", "K_pct_adj",
               "height_inches", "bmi", "round_logit_prior", "nn_mlb_rate"],
    "pitcher": ["Age", "conf_strength", "ERA_adj", "FIP_adj", "K_per_nine_adj", "BB_per_nine_adj",
                "height_inches", "bmi", "round_logit_prior", "nn_mlb_rate"],
}
SIM_STATS = {
    "hitter": ["wOBA_adj", "OPS_adj", "AVG_adj", "SLG_adj", "BB_pct_adj", "K_pct_adj", "ISO_adj"],
    "pitcher": ["ERA_adj", "FIP_adj", "WHIP_adj", "K_per_nine_adj", "BB_per_nine_adj", "K_pct_adj"],
}


def matured(df, last_season=LAST_SEASON, years=YEARS):
    return df[df["draft_year"] + years <= last_season].copy()


def actual_war(df):
    """reached = has a WAR record; WAR = recorded value, 0 if never debuted, NaN if debuted without a record."""
    war = df["war_years_1_through_5"].astype(float)
    reached = war.notna()
    war = war.where(reached, np.where(df["has_mlb_debut"].astype(bool), np.nan, 0.0))
    return reached.astype(int).to_numpy(), war.to_numpy()


def load_cohort():
    df = matured(pd.DataFrame(json.load(open(TRAINING_DATA_PATH))))
    if df.empty:
        raise SystemExit("no matured draft class in the training set")
    df["reached"], df["war"] = actual_war(df)
    missing = int(df["war"].isna().sum())
    print(f"  matured classes {sorted(int(y) for y in df.draft_year.unique())}: {len(df)} players, "
          f"{missing} dropped (debuted, no WAR record)")
    return df[df["war"].notna()].reset_index(drop=True)


def oof_expected_war(df, ptype, n_splits=5, seed=42):
    """Player-grouped (one row per player) out-of-fold P(reach), E[WAR|reach] and their product."""
    feats = FEATURES[ptype]
    X = df[feats].astype(float).fillna(0.0).to_numpy()
    S = df[SIM_STATS[ptype]].astype(float).fillna(0.0).to_numpy()
    rounds = df["draft_round"].fillna(20).astype(int).to_numpy()
    reached, war = df["reached"].to_numpy(), df["war"].to_numpy()
    pi, ni = feats.index("round_logit_prior"), feats.index("nn_mlb_rate")
    assert df["person_id"].is_unique
    p_reach, e_war = np.zeros(len(df)), np.zeros(len(df))
    for tr, te in StratifiedKFold(n_splits, shuffle=True, random_state=seed).split(X, reached):
        Xtr, Xte = X[tr].copy(), X[te].copy()
        Xtr[:, pi], Xtr[:, ni] = crossfit_encode(rounds[tr], S[tr], reached[tr], n_splits=3, seed=seed)
        Xte[:, pi], Xte[:, ni] = encode(rounds[tr], S[tr], reached[tr], rounds[te], S[te])
        hurdle = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000)).fit(Xtr, reached[tr])
        up = reached[tr] == 1
        value = RandomForestRegressor(n_estimators=300, max_depth=4, min_samples_leaf=5, random_state=seed,
                                      n_jobs=2).fit(Xtr[up], war[tr][up])
        p_reach[te] = hurdle.predict_proba(Xte)[:, 1]
        e_war[te] = value.predict(Xte)
    return p_reach, e_war, p_reach * e_war


def compute_metrics(df, p_reach, e_war_given, expected):
    reached, war = df["reached"].to_numpy(), df["war"].to_numpy()
    median, mean = np.full(len(war), np.median(war)), np.full(len(war), war.mean())
    productive = (war > PRODUCTIVE_WAR).astype(int)
    up = reached == 1
    return {
        "n": int(len(war)), "n_reached": int(up.sum()), "n_war_gt_2": int(productive.sum()),
        "r2": float(r2_score(war, expected)),
        "r2_median_baseline": float(r2_score(war, median)),
        "rmse": float(np.sqrt(mean_squared_error(war, expected))),
        "rmse_median_baseline": float(np.sqrt(mean_squared_error(war, median))),
        "rmse_mean_baseline": float(np.sqrt(mean_squared_error(war, mean))),
        "spearman": float(spearmanr(war, expected).correlation),
        "reach_auc": float(roc_auc_score(reached, p_reach)),
        "reach_brier": float(brier_score_loss(reached, p_reach)),
        "reach_brier_base_rate": float(brier_score_loss(reached, np.full(len(war), reached.mean()))),
        "value_r2_given_reach": float(r2_score(war[up], e_war_given[up])),
        "war_gt_2_auc": float(roc_auc_score(productive, expected)) if 0 < productive.sum() < len(war) else None,
    }


def check_kill_criteria(metrics):
    warnings = [f"{pt}: R² = {m['r2']:.4f} < 0.0 (kill criterion failed)" for pt, m in metrics.items() if m["r2"] < 0]
    warnings += [f"{pt}: RMSE {m['rmse']:.3f} not below median baseline {m['rmse_median_baseline']:.3f}"
                 for pt, m in metrics.items() if m["rmse"] >= m["rmse_median_baseline"]]
    return not warnings, warnings


def generate_report(metrics, passed, warnings, classes):
    lines = ["# Tier 5 Backtest Report (matured classes, out of fold)", "",
             f"- **Cohort**: draft classes {classes} (all {YEARS} post-draft seasons complete through {LAST_SEASON})",
             "- **Model**: P(reach MLB) x E[WAR yrs 1-5 | reached], player-grouped 5-fold OOF, "
             "prior and NN rate rebuilt per fold",
             "- **Kill criteria**: R² ≥ 0.0 and RMSE below the median baseline",
             f"- **Status**: {'PASS' if passed else 'FAIL'} (Tier 5 stays off the site until PASS)", ""]
    for pt, m in metrics.items():
        lines += [f"## {pt.title()}"] + [f"- {k}: {v:.4f}" if isinstance(v, float) else f"- {k}: {v}"
                                         for k, v in m.items()] + [""]
    if warnings:
        lines += ["## Warnings", ""] + [f"- {w}" for w in warnings] + [""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Tier 5 backtest (matured classes)")
    parser.add_argument("--report", action="store_true", help="write docs/tier5_backtest_report.md")
    parser.add_argument("--enforce", action="store_true", help="exit 1 if kill criteria fail")
    args = parser.parse_args()

    df = load_cohort()
    metrics = {}
    for pt in ("hitter", "pitcher"):
        sub = df[df["player_type"] == pt].reset_index(drop=True)
        metrics[pt] = compute_metrics(sub, *oof_expected_war(sub, pt))
        print(f"  {pt}: " + "  ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                                      for k, v in metrics[pt].items()))
    passed, warnings = check_kill_criteria(metrics)
    print("Kill criteria", "PASSED" if passed else "FAILED", *warnings, sep="\n  ")
    if args.report:
        path = BASE / "docs/tier5_backtest_report.md"
        path.write_text(generate_report(metrics, passed, warnings, sorted(df.draft_year.unique().tolist())))
        print(f"Report saved: {path}")
    if args.enforce and not passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
