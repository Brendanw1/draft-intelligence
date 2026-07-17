#!/usr/bin/env python3
"""
train_tier3_mlb_arrival.py — Train Tier 3 model predicting MLB arrival
for drafted players using Elastic Net + nearest-neighbor MLB rates.

Two-component ensemble:
  1) Elastic Net logistic regression on college stats + conf_strength + NN rate
  2) Nearest-neighbor MLB rate: proportion of 20 most similar drafted
     players who reached MLB.

This is trained on 2021-2023 drafted players where enough time has passed
for MLB debuts to occur (right-censoring is minimal).

Usage:
    python3 scripts/train_tier3_mlb_arrival.py
"""
import json, pickle, warnings
from pathlib import Path
import numpy as np
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, GroupKFold
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import cdist

warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parents[1]
TRAIN_PATH = BASE / "data" / "training" / "expanded_training_set.json"
DRAFT_PATH = BASE / "data" / "draft" / "draft_all_picks.json"
CONF_STATS_PATH = BASE / "models" / "artifacts_full" / "conference_stats.json"
CONF_STRENGTH_PATH = BASE / "models" / "artifacts_full" / "conference_strength.json"
OUTPUT_DIR = BASE / "models" / "artifacts_full"

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


def compute_nn_mlb_rates(players, sim_stats):
    """For each player, find 20 nearest neighbors and compute MLB debut rate."""
    n = len(players)
    print(f"  Building feature matrix ({len(sim_stats)} similarity stats)...")

    # Build feature matrix for similarity
    X_sim = []
    for p in players:
        row = [safe_float(p.get(s, 0)) or 0 for s in sim_stats]
        X_sim.append(row)
    X_sim = np.array(X_sim)

    # Z-score normalize
    scaler = StandardScaler()
    X_sim_norm = scaler.fit_transform(X_sim)

    # Fit NearestNeighbors
    print(f"  Fitting NearestNeighbors on {n} players...")
    nn = NearestNeighbors(n_neighbors=min(21, n), metric="euclidean", n_jobs=-1)
    nn.fit(X_sim_norm)

    # For each player, find 20 nearest neighbors and compute MLB rate
    # Use 21 to exclude self (nearest neighbor is self at distance 0)
    print(f"  Computing NN MLB rates...")
    distances, indices = nn.kneighbors(X_sim_norm, n_neighbors=min(21, n))

    nn_mlb_rates = []
    for i in range(n):
        # Exclude self (index 0)
        neighbor_indices = indices[i][1:] if len(indices[i]) > 1 else indices[i]
        neighbor_rates = [players[j].get("has_mlb_debut", 0) for j in neighbor_indices]
        nn_mlb_rates.append(np.mean(neighbor_rates) if neighbor_rates else 0.0)

    return nn_mlb_rates, scaler, nn


def main():
    print("=" * 60)
    print("TIER 3: MLB ARRIVAL PREDICTOR")
    print("Elastic Net + Nearest-Neighbor MLB Rates")
    print("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. Load training data ──
    print("\n1. Loading training data...")
    train = load_json(TRAIN_PATH)
    draft = load_json(DRAFT_PATH)

    # Index draft data by person_id for MLB debut lookup
    debut_idx = {}
    for p in draft:
        pid = p.get("person_id")
        if pid:
            debut = p.get("mlb_debut_date")
            debut_idx[pid] = debut if debut and debut != "None" else None

    print(f"  Training records: {len(train)}")
    print(f"  Draft records with debut data: {sum(1 for v in debut_idx.values() if v)}")

    # ── 2. Build drafted-player dataset with MLB outcome ──
    print("\n2. Building drafted-player dataset...")
    conf_stats = load_json(CONF_STATS_PATH)
    conf_strength = load_json(CONF_STRENGTH_PATH)

    # Add features to all records first
    train = add_features(train, conf_stats, conf_strength)

    # Get one record per drafted player (latest season)
    drafted = [r for r in train if r.get("draft_pick") and r["draft_pick"] > 0]
    players = get_player_latest(drafted)
    print(f"  Unique drafted players: {len(players)}")

    # Add MLB debut label
    for p in players:
        pid = p.get("person_id") or hash(p.get("player_name", "")) % (10**10)
        debut = debut_idx.get(pid)
        p["has_mlb_debut"] = 1 if debut else 0

    debuted = sum(1 for p in players if p["has_mlb_debut"])
    print(f"  Reached MLB: {debuted}/{len(players)} ({100*debuted/len(players):.1f}%)")

    # Add draft_round for those who have it
    for p in players:
        rnd = p.get("draft_round")
        if rnd:
            try:
                p["draft_round"] = int(float(rnd))
            except:
                p["draft_round"] = 20
        else:
            p["draft_round"] = 20

    # ── 3. Compute round-specific MLB debut rates for prior ──
    print("\n3. Computing round-specific MLB debut rates (prior)...")
    round_rates = {}
    for p in players:
        rnd = p.get("draft_round", 20)
        if rnd not in round_rates:
            round_rates[rnd] = {"total": 0, "debut": 0}
        round_rates[rnd]["total"] += 1
        if p["has_mlb_debut"]:
            round_rates[rnd]["debut"] += 1

    print(f"  {'Round':>6s} {'Debut':>6s} {'Total':>6s} {'Rate':>7s} {'Logit':>7s}")
    for rnd in sorted(round_rates.keys()):
        rr = round_rates[rnd]
        rate = rr["debut"] / max(rr["total"], 1)
        logit = np.log(max(rate, 0.001) / max(1-rate, 0.001))
        round_rates[rnd]["rate"] = rate
        round_rates[rnd]["logit"] = float(logit)
        print(f"  {rnd:>6d} {rr['debut']:>6d} {rr['total']:>6d} {rate:>6.2%} {logit:>+7.3f}")

    # Apply same filtering: 2021-2023 (all picks, offset handles round)
    recent = [p for p in players if p.get("draft_year", 0) in (2021, 2022, 2023)]
    r_debuted = sum(1 for p in recent if p["has_mlb_debut"])
    print(f"  Players: {len(recent)}")
    print(f"  Reached MLB: {r_debuted}/{len(recent)} ({100*r_debuted/len(recent):.1f}%)")

    results = {}
    for pt, en_features, sim_stats, label in [
        ("hitter", HITTER_T3_FEATURES, HITTER_SIM_STATS, "HITTERS"),
        ("pitcher", PITCHER_T3_FEATURES, PITCHER_SIM_STATS, "PITCHERS"),
    ]:
        print(f"\n{'=' * 60}")
        print(f"{label}")
        print(f"{'=' * 60}")

        pool = [p for p in recent if p.get("player_type") == pt]
        # Also include 2024+ players for NN reference pool (they contribute to similarity,
        # but we only train on 2021-2023)
        all_pt = [p for p in players if p.get("player_type") == pt]

        # Add round_logit_prior based on empirical rates
        for p in recent:
            rnd = p.get("draft_round", 10)
            rr = round_rates.get(rnd, {"logit": -1.0})
            p["round_logit_prior"] = rr["logit"]

        print(f"\n  Training set (2021-2023): {len(pool)}")
        print(f"  Full reference pool: {len(all_pt)}")
        train_debuted = sum(1 for p in pool if p["has_mlb_debut"])
        print(f"  Training set debuted: {train_debuted}/{len(pool)} ({100*train_debuted/len(pool):.1f}%)")

        # ── 3a. Compute NN MLB rates from the FULL reference pool ──
        print(f"\n  Computing nearest-neighbor MLB rates...")
        nn_rates, nn_scaler, nn_model = compute_nn_mlb_rates(all_pt, sim_stats)

        for i, p in enumerate(all_pt):
            p["nn_mlb_rate"] = nn_rates[i]

        # Build training X, y from the 2021-2023 subset
        X_train, y_train = [], []
        for p in pool:
            row = [safe_float(p.get(f)) or 0 for f in en_features]
            X_train.append(row)
            y_train.append(p["has_mlb_debut"])
        X_train = np.array(X_train)
        y_train = np.array(y_train)

        print(f"  Features: {len(en_features)}")
        print(f"  Positive rate: {100*y_train.mean():.1f}%")

        # ── 3b. Cross-validation ──
        print(f"\n  ── 5-Fold CV ──")
        cv_model = LogisticRegression(
            penalty="elasticnet", solver="saga",
            C=1.0, l1_ratio=0.3, max_iter=2000, random_state=42, n_jobs=-1,
        )
        cv_scores = cross_val_score(cv_model, X_train, y_train, cv=5,
                                     scoring="roc_auc", n_jobs=-1)
        print(f"    CV AUC: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")

        # Brier score via manual CV
        from sklearn.model_selection import KFold
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        brier_scores = []
        for train_idx, test_idx in kf.split(X_train):
            m = LogisticRegression(
                penalty="elasticnet", solver="saga",
                C=1.0, l1_ratio=0.3, max_iter=2000, random_state=42, n_jobs=-1,
            )
            m.fit(X_train[train_idx], y_train[train_idx])
            preds = m.predict_proba(X_train[test_idx])[:, 1]
            brier_scores.append(brier_score_loss(y_train[test_idx], preds))
        print(f"    CV Brier: {np.mean(brier_scores):.4f}")

        # ── 3c. Full training ──
        print(f"\n  ── Final Training ──")
        final = LogisticRegression(
            penalty="elasticnet", solver="saga",
            C=1.0, l1_ratio=0.3, max_iter=2000, random_state=42, n_jobs=-1,
        )
        final.fit(X_train, y_train)

        # Feature coefficients
        coefs = sorted(zip(en_features, final.coef_[0]),
                       key=lambda x: abs(x[1]), reverse=True)
        print(f"\n  Top coefficients (|β|):")
        for feat, coef in coefs[:12]:
            print(f"    {feat:<22s} β={coef:+.6f}")

        # Baseline: just the average
        baseline_brier = brier_score_loss(y_train, np.ones_like(y_train) * y_train.mean())
        print(f"\n  Baseline Brier (mean): {baseline_brier:.4f}")
        print(f"  Model Brier: {np.mean(brier_scores):.4f}")
        print(f"  Improvement: {baseline_brier - np.mean(brier_scores):.4f}")

        # ── 3d. Save artifacts ──
        model_path = OUTPUT_DIR / f"tier3_mlb_{pt}.pkl"
        with open(model_path, "wb") as f:
            pickle.dump({
                "model": final,
                "features": en_features,
                "round_rates": {int(k): v for k, v in round_rates.items()},
            }, f)
        print(f"\\n  Saved model: {model_path}")

        nn_path = OUTPUT_DIR / f"tier3_nn_{pt}.pkl"
        with open(nn_path, "wb") as f:
            pickle.dump({
                "nn": nn_model,
                "scaler": nn_scaler,
                "sim_stats": sim_stats,
                "mlb_debut_labels": [p["has_mlb_debut"] for p in all_pt],
                "player_names": [p.get("player_name", "") for p in all_pt],
            }, f)
        print(f"  Saved NN index: {nn_path}")

        # Save metadata
        meta = {
            "model_type": f"tier3_mlb_{pt}",
            "features": en_features,
            "feature_coefficients": [{"feature": f, "coefficient": float(c)} for f, c in coefs],
            "n_train": int(len(y_train)),
            "n_positives": int(y_train.sum()),
            "cv_auc_mean": float(np.mean(cv_scores)),
            "cv_auc_std": float(np.std(cv_scores)),
            "cv_brier": float(np.mean(brier_scores)),
            "nn_reference_pool": len(all_pt),
        }
        meta_path = OUTPUT_DIR / f"tier3_features_{pt}.json"
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        print(f"  Saved metadata: {meta_path}")

        results[pt] = meta

    # ── Summary ──
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    print(f"  {'Model':<20s} {'n':>6s} {'Pos':>6s} {'CV AUC':>7s} {'Brier':>7s}")
    print(f"  {'-'*46}")
    for pt, m in results.items():
        print(f"  {pt:<20s} {m['n_train']:>6d} {m['n_positives']:>6d} {m['cv_auc_mean']:.4f} {m['cv_brier']:.4f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
