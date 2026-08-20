"""Test factories for Tier 5 WAR contracts."""

from __future__ import annotations

import random

from mlb_draft_dashboard.tier5_contracts import (
    TIER5_TRAINING_FIELDS,
    WAR_GROUND_TRUTH_FIELDS,
    signed_log_war,
)


def make_mock_war_record(person_id, war=None, draft_year=2019) -> dict:
    """Build a dict with all WAR_GROUND_TRUTH_FIELDS populated."""
    return {
        "person_id": person_id,
        "key_bbref": f"bbref-{person_id}",
        "career_war_to_date": war if war is not None else 0.0,
        "war_years_1_through_5": war,
        "war_year_1_through_3": war if war is not None else 0.0,
        "peak_single_season_war": war if war is not None else 0.0,
        "seasons_with_positive_war": 1 if (war is not None and war > 0) else 0,
        "total_mlb_seasons": 1 if war is not None else 0,
        "mlb_debut_season": draft_year + 2,
        "draft_year": draft_year,
    }


def make_mock_war_records(n, seed=42) -> list:
    """Build a list of n varied WAR records (nulls, negatives, positives)."""
    rng = random.Random(seed)
    records = []
    for i in range(n):
        roll = rng.random()
        if roll < 0.3:
            war = None
        elif roll < 0.5:
            war = -rng.uniform(0.1, 2.0)
        else:
            war = rng.uniform(0.0, 8.0)
        records.append(make_mock_war_record(person_id=i, war=war, draft_year=2019))
    return records


def make_mock_training_record(person_id, draft_year, role, war) -> dict:
    """Build a dict with all TIER5_TRAINING_FIELDS populated."""
    record = {
        "person_id": person_id,
        "draft_year": draft_year,
        "role": role,
        "meaningfully_productive": war is not None and war > 2.0,
        "war_years_1_through_5": war,
        "signed_log_war": signed_log_war(war) if war is not None else 0.0,
        "age": 21.0,
        "conf_strength": 1.0,
        "wOBA_adj": 0.350,
        "ERA_adj": 4.0,
        "height_inches": 72.0,
        "bmi": 24.0,
        "round_logit_prior": -1.0,
        "nn_mlb_rate": 0.1,
        "draft_pick_number": 100,
        "draft_round": 5,
    }
    return record
