from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import pandas as pd

from .config import EXPORT_FILES
from .contracts import REQUIRED_COLUMNS, missing_columns, missing_export_files
from .data_access import load_dashboard_bundle


@dataclass(frozen=True)
class ValidationIssue:
    level: str
    table: str
    message: str


def validate_bundle(bundle: Dict[str, pd.DataFrame]) -> List[ValidationIssue]:
    issues: List[ValidationIssue] = []

    for table_name, required in REQUIRED_COLUMNS.items():
        frame = bundle.get(table_name, pd.DataFrame())
        missing = missing_columns(frame.columns, required)
        if missing:
            issues.append(ValidationIssue("error", table_name, f"Missing columns: {', '.join(missing)}"))
            continue
        if frame.empty and table_name not in {"diagnostics", "qa", "benchmarks_acc_sec"}:
            issues.append(ValidationIssue("warning", table_name, "Export exists but has no rows."))

    for table_name in ("hitters_board", "pitchers_board"):
        frame = bundle.get(table_name, pd.DataFrame())
        if frame.empty or "player_uid" not in frame.columns:
            continue
        duplicate_count = int(frame["player_uid"].duplicated().sum())
        if duplicate_count:
            issues.append(ValidationIssue("error", table_name, f"Duplicate player_uid rows detected: {duplicate_count}"))

    hitters = bundle.get("hitters_board", pd.DataFrame())
    pitchers = bundle.get("pitchers_board", pd.DataFrame())
    if not hitters.empty and not pitchers.empty and "player_uid" in hitters.columns and "player_uid" in pitchers.columns:
        overlap = set(hitters["player_uid"]).intersection(set(pitchers["player_uid"]))
        if overlap:
            preview = ", ".join(sorted(list(overlap))[:5])
            issues.append(ValidationIssue("error", "boards", f"Player IDs overlap across hitter/pitcher boards: {preview}"))

    benchmarks = bundle.get("benchmarks_acc_sec", pd.DataFrame())
    if not benchmarks.empty:
        invalid_scopes = benchmarks.loc[~benchmarks["benchmark_scope"].isin(["ACC_SEC"]), "benchmark_scope"].dropna().unique().tolist()
        if invalid_scopes:
            issues.append(
                ValidationIssue("warning", "benchmarks_acc_sec", f"Unexpected benchmark_scope values: {', '.join(map(str, invalid_scopes))}")
            )
        invalid_roles = benchmarks.loc[~benchmarks["role"].isin(["hitters", "pitchers"]), "role"].dropna().unique().tolist()
        if invalid_roles:
            issues.append(
                ValidationIssue("warning", "benchmarks_acc_sec", f"Unexpected role values: {', '.join(map(str, invalid_roles))}")
            )

    explanations = bundle.get("explanations", pd.DataFrame())
    if not explanations.empty:
        missing_driver_text = explanations[
            explanations[["positive_driver_1", "negative_driver_1"]].fillna("").eq("").any(axis=1)
        ]
        if not missing_driver_text.empty:
            issues.append(ValidationIssue("warning", "explanations", "Some explanation rows are missing top driver text."))

    trends = bundle.get("player_trends", pd.DataFrame())
    if not trends.empty:
        missing_roles = trends.loc[~trends["role"].isin(["hitters", "pitchers"]), "role"].dropna().unique().tolist()
        if missing_roles:
            issues.append(ValidationIssue("warning", "player_trends", f"Unexpected role values: {', '.join(map(str, missing_roles))}"))

    return issues


def validate_exports_dir(exports_dir: Path) -> List[ValidationIssue]:
    issues: List[ValidationIssue] = []
    missing_files = missing_export_files(exports_dir)
    for missing in missing_files:
        issues.append(ValidationIssue("error", missing, f"Missing expected export file `{EXPORT_FILES[missing]}`"))
    if missing_files:
        return issues
    return validate_bundle(load_dashboard_bundle(exports_dir))


def validation_report_text(issues: List[ValidationIssue]) -> str:
    if not issues:
        return "Dashboard export validation passed with no issues."
    return "\n".join(f"[{issue.level.upper()}] {issue.table}: {issue.message}" for issue in issues)
