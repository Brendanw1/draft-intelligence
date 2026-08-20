"""Tests for Tier 5 WAR contracts."""

from __future__ import annotations

import math

from mlb_draft_dashboard.tier5_contracts import (
    HURDLE_THRESHOLD,
    TIER5_TRAINING_FIELDS,
    WAR_GROUND_TRUTH_FIELDS,
    signed_log_war,
    validate_hurdle_rate,
    validate_temporal_no_leakage,
    validate_war_distribution,
    validate_war_record,
)
from tests.helpers_tier5 import (
    make_mock_training_record,
    make_mock_war_record,
    make_mock_war_records,
)


def test_war_record_accepts_valid_record() -> None:
    record = make_mock_war_record(person_id=1, war=3.5)
    assert validate_war_record(record) is True


def test_war_record_rejects_null_war_in_training_population() -> None:
    record = make_mock_war_record(person_id=1, war=None)
    record["split"] = "train"
    assert validate_war_record(record) is False


def test_war_record_rejects_missing_person_id() -> None:
    record = make_mock_war_record(person_id=1, war=3.5)
    del record["person_id"]
    assert validate_war_record(record) is False


def test_signed_log_war_sign_and_zero() -> None:
    assert signed_log_war(2.0) > 0
    assert signed_log_war(-2.0) < 0
    assert signed_log_war(0.0) == 0.0
    assert signed_log_war(2.0) == math.log1p(2.0)


def test_temporal_no_leakage_detects_future_train_record() -> None:
    train_ok = make_mock_training_record(person_id=1, draft_year=2020, role="hitter", war=3.0)
    train_ok["split"] = "train"
    train_leak = make_mock_training_record(person_id=2, draft_year=2022, role="pitcher", war=4.0)
    train_leak["split"] = "train"
    assert validate_temporal_no_leakage([train_ok], train_max_year=2021) is True
    assert validate_temporal_no_leakage([train_ok, train_leak], train_max_year=2021) is False


def test_hurdle_rate_fraction() -> None:
    rec_a = make_mock_training_record(person_id=1, draft_year=2019, role="hitter", war=5.0)
    rec_b = make_mock_training_record(person_id=2, draft_year=2019, role="pitcher", war=0.5)
    assert validate_hurdle_rate([rec_a, rec_b]) == 0.5


def test_war_distribution_zero_inflated_median() -> None:
    records = [
        make_mock_war_record(person_id=1, war=0.0),
        make_mock_war_record(person_id=2, war=-1.0),
        make_mock_war_record(person_id=3, war=0.5),
        make_mock_war_record(person_id=4, war=None),
    ]
    summary = validate_war_distribution(records)
    assert summary["count"] == 3
    assert summary["median"] <= 0
    assert summary["positive_count"] == 1


def test_field_contracts_are_pinned() -> None:
    assert "person_id" in WAR_GROUND_TRUTH_FIELDS
    assert "war_years_1_through_5" in TIER5_TRAINING_FIELDS
    assert "signed_log_war" in TIER5_TRAINING_FIELDS
    assert HURDLE_THRESHOLD == 2.0
