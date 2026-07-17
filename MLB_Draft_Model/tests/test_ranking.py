import pandas as pd

from mlb_draft_dashboard.ranking import add_rank_columns, compute_custom_rank, normalize_weights


def test_weight_normalization_sums_to_one() -> None:
    normalized = normalize_weights({"reach": 35, "impact": 35, "contact": 20, "risk": 10})
    assert round(sum(normalized.values()), 6) == 1.0


def test_custom_rank_penalizes_risk() -> None:
    frame = pd.DataFrame(
        {
            "player_name": ["A", "B"],
            "reach_score": [80.0, 80.0],
            "impact_score": [80.0, 80.0],
            "contact_score": [80.0, 80.0],
            "risk_score": [10.0, 35.0],
            "production_rank": [1, 2],
        }
    )
    custom = compute_custom_rank(
        frame,
        positive_components={"reach": "reach_score", "impact": "impact_score", "contact": "contact_score"},
        risk_component="risk_score",
        raw_weights={"reach": 35, "impact": 35, "contact": 20, "risk": 10},
    )
    ranked = add_rank_columns(frame, custom)
    assert ranked.iloc[0]["player_name"] == "A"
    assert ranked.iloc[0]["custom_rank"] == 1
