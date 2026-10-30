#!/usr/bin/env python3
"""
train_tier2_full.py — Tier 2: P(drafted in the top 315 picks within three drafts), full D1 population.

Rebuilt 2026-10-29 after the split audit (scratchpad AUDIT-splits.md, Tier 2 row):
  - Label: a fixed horizon, "drafted at pick <= 315 in this season's draft or the next two" (HORIZON = 2), from
    every pick a player received. Seasons whose horizon is not yet observed (season + 2 > last draft) are dropped,
    so every training season is equally observed and the 2026 draftees are back in their earlier seasons.
  - One eligibility rule for both classes: every season kept, freshmen included. Age is not a feature: FanGraphs
    has Age for every drafted player but for under half of the undrafted, so Age (and whether it is missing)
    records the outcome rather than predicting it. Height and BMI are out for the same reason: drafted rows carry
    measured whole-inch heights from the draft record, undrafted rows carry random imputed draws (87% non-integer),
    so the model learned "whole-inch height = drafted" and the board, where every height is a roster whole inch,
    read 2-3x hot (board raw mean 0.37 vs 0.11 on training rows; found 2026-10-29).
  - No player-based row drops. The holdout trains on seasons whose labels were complete before the test season's
    draft (season <= TEST_SEASON - HORIZON - 1), because a later season's label would contain the test draft itself.
  - Calibrators (Platt, isotonic) are fit on out-of-fold scores from player-grouped folds of the training rows;
    the site shows the Platt output (`mlb_prob_platt`), the one scored on the holdout.
  - conf_strength (target-encoded conference rate) is fit on the training rows of each fit only.

Usage:
  PYTHONPATH=src python3 scripts/train_tier2_full.py
"""
import json, pickle, warnings
from collections import defaultdict
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.model_selection import GroupKFold
import xgboost as xgb

from mlb_draft_dashboard.leakfree import OOFCalibrated, conf_strength_table, ece, reliability

warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parents[1]
POSITIVES_PATH = BASE / "data" / "training" / "expanded_training_set.json"
NEGATIVES_PATH = BASE / "data" / "training" / "tier2_negatives.json"
DRAFT_PATH = BASE / "data" / "draft" / "draft_all_picks.json"
OUTPUT_DIR = BASE / "models" / "artifacts_full"
CONF_STATS_PATH = BASE / "models" / "artifacts_full" / "conference_stats.json"
CONF_STRENGTH_PATH = BASE / "models" / "artifacts_full" / "conference_strength.json"
THRESHOLD = 315
HORIZON = 2            # label window: this season's draft and the next two
LAST_DRAFT = 2026      # latest draft in data/draft/draft_all_picks.json
LAST_LABELLED = LAST_DRAFT - HORIZON   # 2024: latest season whose label is fully observed
TEST_SEASON = 2024     # re-read (post hoc); next clean read: the 2025 class after the 2027 draft
MIN_CELL_N = 5

# ── Conference-adjusted features ──
HITTER_ADJ = ["wOBA_adj", "OPS_adj", "AVG_adj", "SLG_adj",
              "BB_pct_adj", "K_pct_adj", "ISO_adj", "wRC_plus_adj"]
PITCHER_ADJ = ["ERA_adj", "FIP_adj", "WHIP_adj",
               "K_per_nine_adj", "BB_per_nine_adj", "K_pct_adj", "BB_pct_adj"]

# ── Interaction features: conf_strength × adjusted stat ──
HITTER_INTERACTIONS = ["strength_x_" + s.replace("_adj", "") for s in HITTER_ADJ]
PITCHER_INTERACTIONS = ["strength_x_" + s.replace("_adj", "") for s in PITCHER_ADJ]

# Full feature lists
T2_HITTER_FEATURES = [
    "G", "AB", "PA", "H", "1B", "2B", "3B", "HR",
    "R", "RBI", "BB", "SO", "HBP", "SF", "SH", "SB", "CS", "GDP",
    "AVG", "BB_pct", "K_pct", "OBP", "SLG", "OPS", "ISO", "Spd",
    "BABIP", "wOBA", "wRC_plus", "wRC", "wRAA", "wBsR", "BB/K",
    "conf_strength",
] + HITTER_ADJ + HITTER_INTERACTIONS

T2_PITCHER_FEATURES = [
    "G", "GS", "CG", "SHO", "SV", "IP", "TBF",
    "H", "R", "ER", "HR", "BB", "SO", "HBP", "WP", "BK",
    "W", "L", "ERA", "WHIP", "FIP", "ERA_minus_FIP",
    "K_pct", "BB_pct", "KBB", "K_per_nine", "BB_per_nine", "HR_per_nine",
    "AVG", "BABIP", "LOB_pct", "K_minus_BB_pct",
    "conf_strength",
] + PITCHER_ADJ + PITCHER_INTERACTIONS


def load_json(path):
    with open(path) as f:
        return json.load(f)


def safe_float(v):
    if v is None:
        return None
    try:
        return float(v)
    except (ValueError, TypeError):
        return None


# ── Conference-adjusted stats ──

def load_conference_stats(path):
    return load_json(path)


def get_conf_avg(conf_stats, conf, season, ptype, stat):
    per_season = conf_stats.get("per_season", {})
    conf_ov = conf_stats.get("conference_overall", {})
    tier_fb = conf_stats.get("tier_fallback", {})

    season_data = per_season.get(conf, {}).get(str(season), {})
    if isinstance(season_data, dict):
        ptype_data = season_data.get(ptype, {})
        if stat in ptype_data:
            return ptype_data[stat]
    conf_data = conf_ov.get(conf, {}).get(ptype, {})
    if stat in conf_data:
        return conf_data[stat]
    # Fallback to tier 3 average
    tier_data = tier_fb.get("3", {}).get(ptype, {})
    return tier_data.get(stat, 0.0)


ADJ_FEATURE_MAP = {
    "wOBA_adj": "wOBA", "OPS_adj": "OPS", "AVG_adj": "AVG", "SLG_adj": "SLG",
    "BB_pct_adj": "BB_pct", "K_pct_adj": "K_pct", "ISO_adj": "ISO", "wRC_plus_adj": "wRC_plus",
    "ERA_adj": "ERA", "FIP_adj": "FIP", "WHIP_adj": "WHIP",
    "K_per_nine_adj": "K_per_nine", "BB_per_nine_adj": "BB_per_nine",
}

# Map interaction names back to their base adjusted stat
INTERACTION_BASE_MAP = {
    "strength_x_wOBA": "wOBA_adj", "strength_x_OPS": "OPS_adj",
    "strength_x_AVG": "AVG_adj", "strength_x_SLG": "SLG_adj",
    "strength_x_BB_pct": "BB_pct_adj", "strength_x_K_pct": "K_pct_adj",
    "strength_x_ISO": "ISO_adj", "strength_x_wRC_plus": "wRC_plus_adj",
    "strength_x_ERA": "ERA_adj", "strength_x_FIP": "FIP_adj",
    "strength_x_WHIP": "WHIP_adj",
    "strength_x_K_per_nine": "K_per_nine_adj", "strength_x_BB_per_nine": "BB_per_nine_adj",
}


def add_features(records, conf_stats, conf_strength):
    """Add conf_strength, conf-adjusted stats, and interactions to records in-place."""
    for rec in records:
        ptype = rec.get("player_type", "hitter")
        conf = rec.get("conference", "")
        season = rec.get("season")

        # 1. conf_strength
        strength = conf_strength.get(conf, {}).get("strength", 1.0)
        rec["conf_strength"] = strength

        # 2. Conference-adjusted stats
        adj_list = HITTER_ADJ if ptype == "hitter" else PITCHER_ADJ
        for adj_feat in adj_list:
            raw_stat = ADJ_FEATURE_MAP.get(adj_feat, adj_feat.replace("_adj", ""))
            raw_val = safe_float(rec.get(raw_stat))
            if raw_val is None:
                rec[adj_feat] = 0.0
            else:
                conf_avg = get_conf_avg(conf_stats, conf, season, ptype, raw_stat)
                rec[adj_feat] = round(raw_val - conf_avg, 4)

        # 3. Interaction features: conf_strength × adjusted stat
        interaction_list = HITTER_INTERACTIONS if ptype == "hitter" else PITCHER_INTERACTIONS
        for int_feat in interaction_list:
            base_adj = INTERACTION_BASE_MAP.get(int_feat)
            if base_adj and base_adj in rec:
                rec[int_feat] = round(strength * rec[base_adj], 4)
            else:
                rec[int_feat] = 0.0

    return records


def load_picks(path=DRAFT_PATH):
    """person_id -> [(draft year, pick number)], every time he was drafted (785 players were drafted twice)."""
    picks = defaultdict(list)
    for p in load_json(path):
        if p.get("person_id") and p.get("pick_number"):
            picks[int(p["person_id"])].append((int(p["year"]), int(p["pick_number"])))
    return picks


def horizon_label(season, person_id, picks, horizon=HORIZON, threshold=THRESHOLD):
    """1 if drafted at pick <= threshold in this season's draft or the next `horizon` drafts, else 0."""
    if person_id is None:
        return 0
    return int(any(season <= yr <= season + horizon and pk <= threshold for yr, pk in picks.get(int(person_id), [])))


def labelled_rows(records, picks, last_season=None):
    """Rows whose fixed-horizon label is observed (season + HORIZON <= last draft in the data), with y attached.
    Same rule for both classes: no age filter, every season kept."""
    last_season = LAST_LABELLED if last_season is None else last_season
    out = []
    for r in records:
        s = r.get("season")
        if s is None or int(s) > last_season:
            continue
        pid = None if r.get("is_undrafted_verified") else r.get("person_id")
        if pid is None and not r.get("is_undrafted_verified"):
            continue
        r["y"] = horizon_label(int(s), pid, picks)
        out.append(r)
    return out


def arrays(rows):
    y = np.array([r["y"] for r in rows])
    groups = np.array([str(r["person_id"]) if r.get("person_id") is not None else "name:" + r.get("player_name", "")
                       for r in rows])
    seasons = np.array([int(r["season"]) for r in rows])
    confs = np.array([r.get("conference") or "" for r in rows], dtype=object)
    return y, groups, seasons, confs


def matrix(rows, feature_cols, conf_stats, strength):
    """Feature matrix with conf_strength (and its interactions) from the given strength table."""
    add_features(rows, conf_stats, strength)
    X = np.zeros((len(rows), len(feature_cols)))
    for i, r in enumerate(rows):
        for j, col in enumerate(feature_cols):
            v = safe_float(r.get(col))
            X[i, j] = 0.0 if v is None or np.isnan(v) else v
    return X


def load_all():
    """Positives + negatives with fixed-horizon labels, the conference stats, and the pick table."""
    picks = load_picks()
    rows = labelled_rows(load_json(POSITIVES_PATH) + load_json(NEGATIVES_PATH), picks)
    return rows, load_conference_stats(CONF_STATS_PATH)


def print_feature_importance(features, importances, top_n=20):
    feat_imp = sorted(zip(features, importances), key=lambda x: x[1], reverse=True)
    print(f"\n  Top {top_n} Features:")
    for feat, imp in feat_imp[:top_n]:
        marker = ""
        if feat in ("height_inches", "bmi", "conf_strength"):
            marker = " ★"
        elif feat.startswith("strength_x_"):
            marker = " ◆"
        elif feat.endswith("_adj") and not feat.startswith("strength_x_"):
            marker = " ▲"
        print(f"    {feat:<26s} {imp:.4f}{marker}")


def make_xgb(y_tr):
    return xgb.XGBClassifier(
        n_estimators=500, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
        reg_alpha=0.1, reg_lambda=1.0,
        scale_pos_weight=(len(y_tr) - y_tr.sum()) / max(y_tr.sum(), 1),
        random_state=42, n_jobs=4, verbosity=0,
    )


def grouped_oof(X, y, groups, n_splits=5):
    """Out-of-fold scores from player-grouped folds: no player is in both the fitting and the scored rows."""
    oof = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits).split(X, y, groups):
        oof[te] = make_xgb(y[tr]).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    return oof


def holdout_split(seasons, test_season=TEST_SEASON, horizon=HORIZON):
    """Train on seasons whose labels were complete before the test season's draft; score the test season."""
    return np.where(seasons <= test_season - horizon - 1)[0], np.where(seasons == test_season)[0]


def score(y, p):
    return {"auc": float(roc_auc_score(y, p)), "brier": float(brier_score_loss(y, p)), "ece": ece(y, p),
            "mean_pred": float(np.mean(p)), "obs_rate": float(np.mean(y))}


def fit_calibrated(X, y, groups):
    base = make_xgb(y).fit(X, y)
    oof = grouped_oof(X, y, groups)
    return base, {m: OOFCalibrated(base, m, oof, y) for m in ("sigmoid", "isotonic")}, oof


def holdout_eval(rows, features, conf_stats, test_season=TEST_SEASON):
    y, groups, seasons, confs = arrays(rows)
    tr, te = holdout_split(seasons, test_season)
    strength = conf_strength_table(confs[tr], y[tr])   # training seasons only
    X = matrix(rows, features, conf_stats, strength)
    base, cals, _ = fit_calibrated(X[tr], y[tr], groups[tr])
    raw = base.predict_proba(X[te])[:, 1]
    out = {"test_season": int(test_season), "train_seasons": sorted(int(v) for v in np.unique(seasons[tr])),
           "n_test": int(len(te)), "n_test_pos": int(y[te].sum()), "n_train": int(len(tr)),
           "read": "re-read (post hoc): the 2024 class was read before on earlier specs", "raw": score(y[te], raw)}
    for m, cal in cals.items():
        p = cal.calibrate(raw)
        out[f"oof_{m}"] = score(y[te], p)
        out[f"oof_{m}"]["reliability"] = reliability(y[te], p)
    return out


def train_model(rows, features, conf_stats, strength, pt):
    """Holdout on a later class, then the shipped model: all labelled rows + player-grouped OOF calibrators."""
    hold = holdout_eval(rows, features, conf_stats)
    for k, v in hold.items():
        if isinstance(v, dict):
            print(f"    {k:<14s} AUC={v['auc']:.4f} Brier={v['brier']:.4f} ECE={v['ece']:.4f} "
                  f"mean_pred={v['mean_pred']:.4f} obs={v['obs_rate']:.4f}")

    y, groups, seasons, _ = arrays(rows)
    X = matrix(rows, features, conf_stats, strength)
    base, cals, oof = fit_calibrated(X, y, groups)
    print_feature_importance(features, base.feature_importances_)
    base.save_model(str(OUTPUT_DIR / f"tier2_full_{pt}.json"))
    for name, m in (("platt", "sigmoid"), ("isotonic", "isotonic")):
        with open(OUTPUT_DIR / f"calibrator_{name}_{pt}.pkl", "wb") as f:
            pickle.dump(cals[m], f)

    feat_imp = sorted(zip(features, base.feature_importances_), key=lambda x: x[1], reverse=True)
    meta = {
        "model_type": f"tier2_full_{pt}",
        "features": features,
        "n_features": len(features),
        "label": f"drafted at pick <= {THRESHOLD} in this season's draft or the next {HORIZON}",
        "train_seasons": sorted(int(v) for v in np.unique(seasons)),
        "n_train": int(len(y)),
        "n_positives": int(y.sum()),
        "n_negatives": int(len(y) - y.sum()),
        "cv_auc_mean": score(y, oof)["auc"],
        "cv_note": "player-grouped 5-fold OOF AUC (raw scores), all labelled seasons",
        "calibration": "Platt/isotonic fit on player-grouped OOF scores; the site shows Platt",
        "holdout": hold,
        "feature_importance": [{"feature": f, "importance": float(i)} for f, i in feat_imp],
    }
    with open(OUTPUT_DIR / f"tier2_full_features_{pt}.json", "w") as f:
        json.dump(meta, f, indent=2)
    return meta


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("TIER 2: P(top 315 within 3 drafts), full population")
    rows, conf_stats = load_all()
    y, _, seasons, confs = arrays(rows)
    strength = conf_strength_table(confs, y)   # shipped: every labelled season (compute_conference_strength.py writes it)
    for s in sorted(set(seasons)):
        print(f"  season {s}: n={int((seasons == s).sum()):,} positive rate={y[seasons == s].mean():.4f}")

    results = {}
    for pt, features in (("hitter", T2_HITTER_FEATURES), ("pitcher", T2_PITCHER_FEATURES)):
        pt_rows = [r for r in rows if r.get("player_type") == pt]
        print(f"\n{pt}: {len(pt_rows):,} rows, {sum(r['y'] for r in pt_rows):,} positive")
        results[pt] = train_model(pt_rows, features, conf_stats, strength, pt)
    for pt, m in results.items():
        print(f"  {pt}: n={m['n_train']:,} pos={m['n_positives']:,} OOF AUC={m['cv_auc_mean']:.4f}")


if __name__ == "__main__":
    main()
