"""Tier 5 WAR contracts — pinned field names and gate thresholds.

Wave 0 schema work only. These constants and validators pin the field names
and gate thresholds that downstream WAR-pull and training scripts must honor.
No data pull, no model training lives here.
"""

from __future__ import annotations

import math
from typing import Dict, List

# ---------------------------------------------------------------------------
# Field contracts
# ---------------------------------------------------------------------------

#: Required keys on a raw WAR ground-truth record (one per player).
WAR_GROUND_TRUTH_FIELDS: List[str] = [
    "person_id",
    "key_bbref",
    "career_war_to_date",
    "war_years_1_through_5",
    "war_year_1_through_3",
    "peak_single_season_war",
    "seasons_with_positive_war",
    "total_mlb_seasons",
    "mlb_debut_season",
    "draft_year",
]

#: Required keys on a Tier 5 training record (features + WAR label).
TIER5_TRAINING_FIELDS: List[str] = [
    "person_id",
    "draft_year",
    "role",  # "hitter" | "pitcher"
    "meaningfully_productive",  # bool
    "war_years_1_through_5",  # float
    "signed_log_war",  # float
    # Existing Tier 3 feature names
    "age",
    "conf_strength",
    "wOBA_adj",
    "ERA_adj",
    "height_inches",
    "bmi",
    "round_logit_prior",
    "nn_mlb_rate",
    "draft_pick_number",
    "draft_round",
]

#: Hurdle threshold (WAR) for "meaningfully productive".
HURDLE_THRESHOLD: float = 2.0


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def validate_war_record(record: dict) -> bool:
    """Return True if ``record`` is a valid WAR ground-truth record.

    A record is invalid if:
      * it is missing ``person_id``, or
      * ``war_years_1_through_5`` is None while the record claims to be in
        the training population (i.e. it carries a ``split`` field).
    """
    if "person_id" not in record or record.get("person_id") is None:
        return False
    if record.get("war_years_1_through_5") is None and "split" in record:
        return False
    return True


def validate_war_distribution(records: list) -> dict:
    """Return summary stats over non-null ``war_years_1_through_5`` values.

    Returns ``{"median": float, "count": int, "positive_count": int}``.
    """
    values = [
        r["war_years_1_through_5"]
        for r in records
        if r.get("war_years_1_through_5") is not None
    ]
    if not values:
        return {"median": 0.0, "count": 0, "positive_count": 0}
    ordered = sorted(values)
    n = len(ordered)
    mid = n // 2
    if n % 2 == 1:
        median = float(ordered[mid])
    else:
        median = (ordered[mid - 1] + ordered[mid]) / 2.0
    return {
        "median": median,
        "count": n,
        "positive_count": sum(1 for v in values if v > 0),
    }


def validate_temporal_no_leakage(records: list, train_max_year: int) -> bool:
    """Return True only if no training-candidate record leaks future data.

    A record is a training candidate if it has a ``split`` field equal to
    ``"train"``. If any such record has ``draft_year > train_max_year`` the
    check fails (returns False).
    """
    for record in records:
        if record.get("split") == "train":
            draft_year = record.get("draft_year")
            if draft_year is not None and draft_year > train_max_year:
                return False
    return True


def validate_hurdle_rate(records: list) -> float:
    """Return the fraction of records flagged ``meaningfully_productive``."""
    if not records:
        return 0.0
    positive = sum(1 for r in records if r.get("meaningfully_productive") is True)
    return positive / len(records)


def signed_log_war(war: float) -> float:
    """Signed log transform: ``sign(war) * log1p(abs(war))``.

    Guard: ``war == 0.0`` returns ``0.0``.
    """
    if war == 0.0:
        return 0.0
    sign = -1.0 if war < 0 else 1.0
    return sign * math.log1p(abs(war))
