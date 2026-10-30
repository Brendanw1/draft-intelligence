"""Leak-free building blocks shared by the Tier 2, 3 and 5 trainers (URGENT-FIXES.md, 2026-10-29).

Every label-derived feature (round prior, neighbour MLB rate) and every calibrator is fit on rows that
exclude the row being scored. The classes here are pickled into artifacts, so they live in the package
(importable with PYTHONPATH=src) rather than in a script.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

PRIOR_SHRINK = 20  # pseudo-count pulling each round's rate toward the pooled rate
N_NEIGHBORS = 20


# ── label-derived features ──────────────────────────────────────────────────

def round_rates(ref_rounds, ref_y, m=PRIOR_SHRINK):
    """Shrunk per-round debut rate and logit from the reference rows only. Same dict shape infer_2026 reads."""
    ref_rounds, ref_y = np.asarray(ref_rounds), np.asarray(ref_y, dtype=float)
    base = ref_y.mean()
    out = {}
    for r in np.unique(ref_rounds):
        sel = ref_rounds == r
        rate = (ref_y[sel].sum() + m * base) / (sel.sum() + m)
        out[int(r)] = {"total": int(sel.sum()), "debut": int(ref_y[sel].sum()),
                       "rate": float(rate), "logit": float(logit(np.clip(rate, 1e-3, 1 - 1e-3)))}
    out["_pooled"] = {"rate": float(base), "logit": float(logit(np.clip(base, 1e-3, 1 - 1e-3)))}
    return out


def prior_logit(rates, rounds):
    pooled = rates["_pooled"]["logit"]
    return np.array([rates.get(int(r), {"logit": pooled})["logit"] for r in rounds])


def fit_nn(ref_X, k=N_NEIGHBORS):
    scaler = StandardScaler().fit(ref_X)
    nn = NearestNeighbors(n_neighbors=min(k, len(ref_X))).fit(scaler.transform(ref_X))
    return scaler, nn


def nn_rate(ref_X, ref_y, X, k=N_NEIGHBORS):
    """Share of the k nearest REFERENCE rows that reached MLB. The scored rows are never in ref."""
    scaler, nn = fit_nn(ref_X, k)
    _, idx = nn.kneighbors(scaler.transform(X))
    return np.asarray(ref_y, dtype=float)[idx].mean(axis=1)


def encode(ref_rounds, ref_X, ref_y, rounds, X):
    """(round_logit_prior, nn_mlb_rate) for the target rows, using reference labels only."""
    return prior_logit(round_rates(ref_rounds, ref_y), rounds), nn_rate(ref_X, ref_y, X)


def crossfit_encode(rounds, X, y, n_splits=5, seed=42):
    """Out-of-fold encoding: each row's prior and neighbour rate come from the other folds."""
    rounds, X, y = np.asarray(rounds), np.asarray(X, dtype=float), np.asarray(y)
    prior, rate = np.zeros(len(y)), np.zeros(len(y))
    for tr, te in StratifiedKFold(n_splits, shuffle=True, random_state=seed).split(X, y):
        prior[te], rate[te] = encode(rounds[tr], X[tr], y[tr], rounds[te], X[te])
    return prior, rate


# ── Tier 3: logistic regression with a true offset ─────────────────────────

class OffsetLogit:
    """L2 logistic regression with a fixed logit offset (coefficient 1, not learned) and scaled inputs.

    `features` is the full column list infer_2026 builds; the `offset` column is taken as the offset and
    every other column is standardised. predict_proba(X) keeps the sklearn interface infer_2026 calls.
    """

    def __init__(self, features, offset="round_logit_prior", C=1.0):
        self.features, self.offset, self.C = list(features), offset, C
        self.off_idx = self.features.index(offset)
        self.x_idx = [i for i in range(len(self.features)) if i != self.off_idx]

    def _split(self, X):
        X = np.asarray(X, dtype=float)
        return X[:, self.x_idx], X[:, self.off_idx]

    def fit(self, X, y):
        Z, off = self._split(X)
        self.scaler = StandardScaler().fit(Z)
        Z = self.scaler.transform(Z)
        y = np.asarray(y, dtype=float)

        def loss(w):
            eta = off + w[0] + Z @ w[1:]
            p = expit(eta)
            nll = np.sum(np.logaddexp(0, eta) - y * eta)
            return nll + 0.5 / self.C * w[1:] @ w[1:], np.r_[np.sum(p - y), Z.T @ (p - y) + w[1:] / self.C]

        w = minimize(loss, np.zeros(Z.shape[1] + 1), jac=True, method="L-BFGS-B").x
        self.intercept_, self.coef_ = np.array([w[0]]), w[1:][None, :]
        return self

    def predict_proba(self, X):
        Z, off = self._split(X)
        # Clip standardised inputs to +/-4 SD: a 1-PA 2026 line (BB% 1.0) is 20+ SD out and would
        # otherwise push a late-round pick to P ~ 0.99. Training rows sit inside this range.
        z = np.clip(self.scaler.transform(Z), -4, 4)
        p = expit(off + self.intercept_[0] + z @ self.coef_[0])
        return np.c_[1 - p, p]


# ── Tier 2: calibrator fit on out-of-fold scores ────────────────────────────

class OOFCalibrated:
    """A base classifier plus a score->probability map fit on out-of-fold predictions only."""

    def __init__(self, base, method, oof_scores, oof_y):
        self.base, self.method = base, method
        s = np.clip(np.asarray(oof_scores, dtype=float), 1e-6, 1 - 1e-6)
        if method == "sigmoid":
            self.map = LogisticRegression(C=1e6).fit(logit(s)[:, None], oof_y)
        else:
            self.map = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(s, oof_y)

    def calibrate(self, scores):
        s = np.clip(np.asarray(scores, dtype=float), 1e-6, 1 - 1e-6)
        if self.method == "sigmoid":
            return self.map.predict_proba(logit(s)[:, None])[:, 1]
        return self.map.predict(s)

    def predict_proba(self, X):
        p = self.calibrate(self.base.predict_proba(X)[:, 1])
        return np.c_[1 - p, p]


# ── metrics ─────────────────────────────────────────────────────────────────

def reliability(y, p, n_bins=10):
    """Equal-width bins: n, mean predicted, observed rate."""
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    b = np.minimum((p * n_bins).astype(int), n_bins - 1)
    return [{"bin": int(i), "n": int((b == i).sum()), "pred_mean": float(p[b == i].mean()),
             "obs_rate": float(y[b == i].mean())} for i in range(n_bins) if (b == i).any()]


def ece(y, p, n_bins=10):
    rows = reliability(y, p, n_bins)
    return float(sum(r["n"] * abs(r["pred_mean"] - r["obs_rate"]) for r in rows) / len(y))


# ── conference strength (target-encoded) ────────────────────────────────────

def conf_strength_table(conferences, y, min_n=10):
    """Per-conference positive rate / pooled rate, from the rows passed in only (fit on training seasons).

    Same dict shape as models/artifacts_full/conference_strength.json; conferences with < min_n rows are left
    out and read as 1.0 by every consumer.
    """
    conferences, y = np.asarray(conferences, dtype=object), np.asarray(y, dtype=float)
    keep = np.array([bool(c) for c in conferences])
    base = y[keep].mean() if keep.any() else 0.0
    out = {}
    for c in np.unique(conferences[keep]):
        sel = conferences == c
        if sel.sum() < min_n:
            continue
        rate = y[sel].mean()
        out[str(c)] = {"strength": round(float(rate / base) if base > 0 else 1.0, 4),
                       "draft_rate": round(float(rate), 4), "global_rate": round(float(base), 4),
                       "total": int(sel.sum()), "drafted": int(y[sel].sum())}
    return out
