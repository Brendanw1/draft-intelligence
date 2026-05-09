from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

from .config import EXPORT_FILES


@dataclass(frozen=True)
class ValidationMessage:
    level: str
    table: str
    message: str


REQUIRED_COLUMNS: Dict[str, List[str]] = {
    "hitters_board": [
        "player_uid",
        "player_name",
        "team_code",
        "school_name",
        "conference",
        "season",
        "bats",
        "throws",
        "class_year",
        "production_rank",
        "draft_value_score",
        "reach_score",
        "impact_score",
        "contact_score",
        "risk_score",
        "plate_events",
        "bbe_count",
        "p90_ev_wood_adj",
        "avg_ev_wood_adj",
        "barrel_rate_proxy_wood_adj",
        "contact_rate",
        "whiff_rate",
        "chase_rate",
        "trend_delta",
        "data_completeness_score",
        "one_season_only_flag",
        "missing_critical_count",
        "export_ts",
    ],
    "pitchers_board": [
        "player_uid",
        "player_name",
        "team_code",
        "school_name",
        "conference",
        "season",
        "throws",
        "class_year",
        "production_rank",
        "draft_value_score",
        "reach_score",
        "stuff_score",
        "command_score",
        "risk_score",
        "pitch_count",
        "avg_fb_velo",
        "max_fb_velo",
        "avg_ivb",
        "avg_hb",
        "extension",
        "csw_pct",
        "whiff_pct",
        "zone_pct",
        "arsenal_count",
        "trend_delta",
        "data_completeness_score",
        "one_season_only_flag",
        "missing_critical_count",
        "export_ts",
    ],
    "player_trends": ["player_uid", "player_name", "role", "season", "metric_key", "metric_label", "metric_value"],
    "hitter_bbe_detail": ["player_uid", "player_name", "season", "exit_speed", "angle", "direction"],
    "pitcher_pitchtype_detail": [
        "player_uid",
        "player_name",
        "season",
        "pitch_type",
        "usage_pct",
        "avg_velo",
        "avg_ivb",
        "avg_hb",
        "extension",
        "rel_height",
        "rel_side",
        "zone_pct",
        "whiff_pct",
        "csw_pct",
        "hard_hit_allowed_pct",
    ],
    "benchmarks_acc_sec": ["season", "role", "metric_key", "benchmark_scope", "benchmark_value", "benchmark_label"],
    "explanations": [
        "player_uid",
        "role",
        "sample_size_text",
        "data_completeness_score",
        "match_confidence",
        "positive_driver_1",
        "positive_driver_2",
        "negative_driver_1",
        "negative_driver_2",
        "warning_text",
    ],
    "diagnostics": ["role", "record_type", "section", "label"],
    "qa": ["record_type", "section", "label"],
}


def missing_export_files(exports_dir: Path) -> List[str]:
    missing: List[str] = []
    for key, filename in EXPORT_FILES.items():
        if resolve_export_path(exports_dir, filename) is None:
            missing.append(key)
    return missing


def missing_columns(columns: Iterable[str], required: Iterable[str]) -> List[str]:
    available = set(columns)
    return [column for column in required if column not in available]


def candidate_export_paths(exports_dir: Path, filename: str) -> List[Path]:
    parquet_path = exports_dir / filename
    candidates = [parquet_path]
    if filename.endswith(".parquet"):
        candidates.append(exports_dir / filename.replace(".parquet", ".csv"))
    return candidates


def resolve_export_path(exports_dir: Path, filename: str) -> Path | None:
    for candidate in candidate_export_paths(exports_dir, filename):
        if candidate.exists():
            return candidate
    return None
