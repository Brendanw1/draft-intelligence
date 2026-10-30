#!/usr/bin/env python3
"""
train_tier3_mlb_arrival.py — Tier 3: P(MLB debut within 3 seasons | drafted).

Leak-free since 2026-10-29 (URGENT-FIXES.md #1, #2):
  * round_logit_prior is a TRUE offset (coefficient fixed at 1), not a feature, and it is computed out of
    fold: each player's prior comes from the other folds' players, never from his own label.
  * nn_mlb_rate is rebuilt per fold: neighbours come only from rows that are not being scored.
  * inputs are standardised before the penalised (L2) logistic fit.
  * censoring: the label is a fixed horizon (debut by the end of draft_year + 3), so 2021-2023 classes are
    equally observed; 2024+ draftees are censored and excluded from training, the prior and the NN pool.

Artifacts go to models/artifacts_fixed/ (same filenames infer_2026.py reads); the live
models/artifacts_full/ is untouched until the owner switches.

Usage:
    PYTHONPATH=src python3 scripts/train_tier3_mlb_arrival.py
"""
import json, pickle, warnings
from pathlib import Path
import numpy as np
from collections import defaultdict
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, brier_score_loss

from mlb_draft_dashboard.leakfree import (
    OffsetLogit, crossfit_encode, encode, ece, fit_nn, round_rates,
)

warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parents[1]
TRAIN_PATH = BASE / "data" / "training" / "expanded_training_set.json"
DRAFT_PATH = BASE / "data" / "draft" / "draft_all_picks.json"
CONF_STATS_PATH = BASE / "models" / "artifacts_full" / "conference_stats.json"
CONF_STRENGTH_PATH = BASE / "models" / "artifacts_full" / "conference_strength.json"
OUTPUT_DIR = BASE / "models" / "artifacts_full"
HORIZON = 3          # seasons after the draft year
LAST_SEASON = 2025   # last MLB season with complete debut data: draft_all_picks.json was scraped 2026-07-14, so 2026 is partial
TRAIN_CLASSES = [y for y in (2021, 2022, 2023) if y + HORIZON <= LAST_SEASON]

# Key stats for nearest-neighbor similarity (used for both NN rate and Elastic Net)
# These are the stats that best capture "profile similarity" across hitters/pitchers
HITTER_SIM_STATS = ["wOBA_adj", "OPS_adj", "AVG_adj", "SLG_adj", "BB_pct_adj", "K_pct_adj", "ISO_adj"]
PITCHER_SIM_STATS = ["ERA_adj", "FIP_adj", "WHIP_adj", "K_per_nine_adj", "BB_per_nine_adj", "K_pct_adj"]

HITTER_ADJ = HITTER_SIM_STATS + ["wRC_plus_adj"]
PITCHER_ADJ = PITCHER_SIM_STATS + ["BB_pct_adj"]

HITTER_INTERACTIONS = ["strength_x_" + s.replace("_adj", "") for s in HITTER_ADJ]
PITCHER_INTERACTIONS = ["strength_x_" + s.replace("_adj", "") for s in PITCHER_ADJ]

# Tier 3 features — round_logit_prior replaces raw draft_round with empirical logit
HITTER_T3_FEATURES = [
    "Age", "conf_strength",
    "wOBA_adj", "OPS_adj", "BB_pct_adj", "K_pct_adj",
    "height_inches", "bmi", "round_logit_prior", "nn_mlb_rate",
]
PITCHER_T3_FEATURES = [
    "Age", "conf_strength",
    "ERA_adj", "FIP_adj", "K_per_nine_adj", "BB_per_nine_adj",
    "height_inches", "bmi", "round_logit_prior", "nn_mlb_rate",
]

ADJ_FEATURE_MAP = {
    "wOBA_adj": "wOBA", "OPS_adj": "OPS", "AVG_adj": "AVG", "SLG_adj": "SLG",
    "BB_pct_adj": "BB_pct", "K_pct_adj": "K_pct", "ISO_adj": "ISO", "wRC_plus_adj": "wRC_plus",
    "ERA_adj": "ERA", "FIP_adj": "FIP", "WHIP_adj": "WHIP",
    "K_per_nine_adj": "K_per_nine", "BB_per_nine_adj": "BB_per_nine",
}
INTERACTION_BASE_MAP = {k: v for k, v in [
    ("strength_x_wOBA", "wOBA_adj"), ("strength_x_OPS", "OPS_adj"),
    ("strength_x_AVG", "AVG_adj"), ("strength_x_SLG", "SLG_adj"),
    ("strength_x_BB_pct", "BB_pct_adj"), ("strength_x_K_pct", "K_pct_adj"),
    ("strength_x_ISO", "ISO_adj"), ("strength_x_wRC_plus", "wRC_plus_adj"),
    ("strength_x_ERA", "ERA_adj"), ("strength_x_FIP", "FIP_adj"),
    ("strength_x_WHIP", "WHIP_adj"),
    ("strength_x_K_per_nine", "K_per_nine_adj"), ("strength_x_BB_per_nine", "BB_per_nine_adj"),
]}


def load_json(path):
    return json.load(open(path))


def safe_float(v):
    if v is None: return None
    try: return float(v)
    except: return None


def get_conf_avg(conf_stats, conf, season, ptype, stat):
    ps = conf_stats.get("per_season", {})
    co = conf_stats.get("conference_overall", {})
    tf = conf_stats.get("tier_fallback", {})
    sd = ps.get(conf, {}).get(str(season), {})
    if isinstance(sd, dict) and stat in sd.get(ptype, {}): return sd[ptype][stat]
    cd = co.get(conf, {}).get(ptype, {})
    if stat in cd: return cd[stat]
    return tf.get("3", {}).get(ptype, {}).get(stat, 0.0)


def add_features(records, conf_stats, conf_strength):
    for rec in records:
        ptype = rec.get("player_type", "hitter")
        conf = rec.get("conference", "")
        season = rec.get("season")
        strength = conf_strength.get(conf, {}).get("strength", 1.0)
        rec["conf_strength"] = strength
        adj_list = HITTER_ADJ if ptype == "hitter" else PITCHER_ADJ
        for af in adj_list:
            rs = ADJ_FEATURE_MAP.get(af, af.replace("_adj", ""))
            rv = safe_float(rec.get(rs))
            rec[af] = round(rv - get_conf_avg(conf_stats, conf, season, ptype, rs), 4) if rv is not None else 0.0
        il = HITTER_INTERACTIONS if ptype == "hitter" else PITCHER_INTERACTIONS
        for inf in il:
            ba = INTERACTION_BASE_MAP.get(inf)
            rec[inf] = round(strength * rec.get(ba, 0), 4) if ba else 0.0
    return records


def get_player_latest(records):
    """Group records by person_id, return latest season per player."""
    groups = defaultdict(list)
    for r in records:
        pid = r.get("person_id") or hash(r.get("player_name", "")) % (10**10)
        groups[pid].append(r)
    result = []
    for pid, recs in groups.items():
        result.append(max(recs, key=lambda x: x.get("season", 0) or 0))
    return result



def label_horizon(debut_date, draft_year, horizon=HORIZON):
    """1 if the debut came by the end of draft_year + horizon, else 0. Never 'unknown' for TRAIN_CLASSES."""
    if not debut_date or debut_date == "None":
        return 0
    return int(int(str(debut_date)[:4]) <= draft_year + horizon)


def build_xy(pool, en_features, sim_stats):
    X = np.array([[safe_float(p.get(f)) or 0.0 for f in en_features] for p in pool])
    S = np.array([[safe_float(p.get(s)) or 0.0 for s in sim_stats] for p in pool])
    rounds = np.array([p["draft_round"] for p in pool])
    y = np.array([p["has_mlb_debut"] for p in pool])
    return X, S, rounds, y


def with_encodings(X, en_features, prior, rate):
    X = X.copy()
    X[:, en_features.index("round_logit_prior")] = prior
    X[:, en_features.index("nn_mlb_rate")] = rate
    return X


def oof_predictions(X, S, rounds, y, en_features, n_splits=5, seed=42):
    """Outer CV. Train rows get cross-fit encodings inside the fold; test rows are encoded from the train fold."""
    pred, prior_only = np.zeros(len(y)), np.zeros(len(y))
    for tr, te in StratifiedKFold(n_splits, shuffle=True, random_state=seed).split(X, y):
        tr_prior, tr_rate = crossfit_encode(rounds[tr], S[tr], y[tr], seed=seed)
        te_prior, te_rate = encode(rounds[tr], S[tr], y[tr], rounds[te], S[te])
        m = OffsetLogit(en_features).fit(with_encodings(X[tr], en_features, tr_prior, tr_rate), y[tr])
        pred[te] = m.predict_proba(with_encodings(X[te], en_features, te_prior, te_rate))[:, 1]
        prior_only[te] = 1 / (1 + np.exp(-te_prior))
    return pred, prior_only


def main():
    print("TIER 3: MLB ARRIVAL (leak-free offset model)")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    train = load_json(TRAIN_PATH)
    draft = load_json(DRAFT_PATH)
    debut_idx = {p["person_id"]: p.get("mlb_debut_date") for p in draft if p.get("person_id")}

    train = add_features(train, load_json(CONF_STATS_PATH), load_json(CONF_STRENGTH_PATH))
    drafted = [r for r in train if r.get("draft_pick") and r["draft_pick"] > 0]
    players = [p for p in get_player_latest(drafted)
               if p.get("draft_year") in TRAIN_CLASSES and p.get("person_id") in debut_idx]
    for p in players:
        p["has_mlb_debut"] = label_horizon(debut_idx[p["person_id"]], p["draft_year"])
        try:
            p["draft_round"] = int(float(p.get("draft_round")))
        except (TypeError, ValueError):
            p["draft_round"] = 20
    print(f"  classes {TRAIN_CLASSES}, horizon {HORIZON}: {len(players)} players, "
          f"{sum(p['has_mlb_debut'] for p in players)} debuts")

    results = {}
    for pt, en_features, sim_stats in [("hitter", HITTER_T3_FEATURES, HITTER_SIM_STATS),
                                       ("pitcher", PITCHER_T3_FEATURES, PITCHER_SIM_STATS)]:
        pool = [p for p in players if p.get("player_type") == pt]
        X, S, rounds, y = build_xy(pool, en_features, sim_stats)
        pred, prior_only = oof_predictions(X, S, rounds, y, en_features)
        cv = {"auc": roc_auc_score(y, pred), "brier": brier_score_loss(y, pred), "ece": ece(y, pred),
              "prior_only_auc": roc_auc_score(y, prior_only),
              "prior_only_brier": brier_score_loss(y, prior_only),
              "base_rate_brier": brier_score_loss(y, np.full(len(y), y.mean()))}
        print(f"  {pt}: n={len(y)} pos={y.sum()}  OOF AUC={cv['auc']:.4f} Brier={cv['brier']:.4f} "
              f"ECE={cv['ece']:.4f} | prior-only AUC={cv['prior_only_auc']:.4f} "
              f"Brier={cv['prior_only_brier']:.4f} | base-rate Brier={cv['base_rate_brier']:.4f}")

        # Final model: cross-fit encodings for its own training rows; full-data prior + NN pool for inference.
        prior, rate = crossfit_encode(rounds, S, y)
        final = OffsetLogit(en_features).fit(with_encodings(X, en_features, prior, rate), y)
        rates = round_rates(rounds, y)
        nn_scaler, nn_model = fit_nn(S, k=21)  # infer_2026 asks for 21 neighbours
        coefs = sorted(zip([f for f in en_features if f != "round_logit_prior"], final.coef_[0]),
                       key=lambda x: abs(x[1]), reverse=True)
        with open(OUTPUT_DIR / f"tier3_mlb_{pt}.pkl", "wb") as f:
            pickle.dump({"model": final, "features": en_features, "round_rates": rates,
                         "label": f"debut by draft_year+{HORIZON}", "offset": "round_logit_prior"}, f)
        with open(OUTPUT_DIR / f"tier3_nn_{pt}.pkl", "wb") as f:
            pickle.dump({"nn": nn_model, "scaler": nn_scaler, "sim_stats": sim_stats,
                         "mlb_debut_labels": y.tolist(),
                         "player_names": [p.get("player_name", "") for p in pool]}, f)
        meta = {"model_type": f"tier3_mlb_{pt}", "features": en_features,
                "offset": "round_logit_prior (fixed coefficient 1)",
                "label": f"MLB debut by end of draft_year+{HORIZON}", "train_classes": TRAIN_CLASSES,
                "feature_coefficients_standardised": [{"feature": f, "coefficient": float(c)} for f, c in coefs],
                "n_train": int(len(y)), "n_positives": int(y.sum()),
                "cv_auc_mean": float(cv["auc"]), "cv_brier": float(cv["brier"]), "cv_ece": float(cv["ece"]),
                "cv_prior_only_auc": float(cv["prior_only_auc"]),
                "cv_prior_only_brier": float(cv["prior_only_brier"]),
                "base_rate_brier": float(cv["base_rate_brier"]),
                "cv_note": "nested out-of-fold: prior and NN rate never see the scored row's label",
                "nn_reference_pool": int(len(y))}
        with open(OUTPUT_DIR / f"tier3_features_{pt}.json", "w") as f:
            json.dump(meta, f, indent=2)
        results[pt] = meta
    print(f"  saved to {OUTPUT_DIR}")
    return results


if __name__ == "__main__":
    main()
