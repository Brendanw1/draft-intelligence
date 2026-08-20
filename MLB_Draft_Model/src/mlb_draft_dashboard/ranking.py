from __future__ import annotations

from typing import Dict

import pandas as pd


def normalize_weights(raw_weights: Dict[str, float]) -> Dict[str, float]:
    sanitized = {key: max(float(value), 0.0) for key, value in raw_weights.items()}
    total = sum(sanitized.values())
    if total <= 0:
        even_weight = 1.0 / len(sanitized) if sanitized else 0.0
        return {key: even_weight for key in sanitized}
    return {key: value / total for key, value in sanitized.items()}


def compute_custom_rank(
    frame: pd.DataFrame,
    positive_components: Dict[str, str],
    risk_component: str,
    raw_weights: Dict[str, float],
) -> pd.Series:
    if frame.empty:
        return pd.Series(dtype=float)

    weights = normalize_weights(raw_weights)
    score = pd.Series(0.0, index=frame.index)

    for weight_key, column_name in positive_components.items():
        score = score.add(frame[column_name].fillna(0.0) * weights.get(weight_key, 0.0), fill_value=0.0)

    score = score.sub(frame[risk_component].fillna(0.0) * weights.get("risk", 0.0), fill_value=0.0)
    return score.clip(lower=0.0, upper=100.0)


def add_rank_columns(frame: pd.DataFrame, custom_score: pd.Series) -> pd.DataFrame:
    ranked = frame.copy()
    ranked["custom_rank_score"] = custom_score.round(2)
    ranked["custom_rank"] = ranked["custom_rank_score"].rank(method="min", ascending=False).astype(int)
    return ranked.sort_values(["custom_rank", "production_rank", "player_name"])
