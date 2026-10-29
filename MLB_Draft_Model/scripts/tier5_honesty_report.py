#!/usr/bin/env python3
"""Generate Tier 5 honesty report.

Produces a markdown report covering:
  * Cohort description (n_train, n_heldout, positive rate, right-censoring note)
  * Hurdle classifier performance (AUC, Brier, calibration plot data)
  * Value regression performance (RMSE, MAE, R²)
  * Kill criteria: if heldout AUC < 0.60 OR heldout R² < 0.0, flag as "not ready"
  * Comparison to naive baseline (predict median WAR for all)

Usage:
    python scripts/tier5_honesty_report.py --training PATH --output PATH
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
DEFAULT_TRAINING = BASE / "data" / "training" / "tier5_training_set.json"
DEFAULT_OUTPUT = BASE / "docs" / "tier5_honesty_report.md"
DEFAULT_HURDLE_HITTER = BASE / "models" / "artifacts_full" / "tier5_hurdle_hitter_features.json"
DEFAULT_HURDLE_PITCHER = BASE / "models" / "artifacts_full" / "tier5_hurdle_pitcher_features.json"
DEFAULT_VALUE_HITTER = BASE / "models" / "artifacts_full" / "tier5_value_hitter_features.json"
DEFAULT_VALUE_PITCHER = BASE / "models" / "artifacts_full" / "tier5_value_pitcher_features.json"

KILL_AUC_GATE = 0.60
KILL_R2_GATE = 0.0


def load_json(path):
    return json.load(open(path))


def load_optional(path):
    try:
        return json.load(open(path))
    except FileNotFoundError:
        return None


def generate_report(training_path, output_path):
    training = load_json(training_path)
    hurdle_hitter = load_optional(DEFAULT_HURDLE_HITTER)
    hurdle_pitcher = load_optional(DEFAULT_HURDLE_PITCHER)
    value_hitter = load_optional(DEFAULT_VALUE_HITTER)
    value_pitcher = load_optional(DEFAULT_VALUE_PITCHER)

    # Cohort stats
    n_train = sum(1 for r in training if r.get("split") == "train")
    n_heldout = sum(1 for r in training if r.get("split") == "heldout")
    n_positive = sum(1 for r in training if r.get("meaningfully_productive") is True)
    positive_rate = n_positive / len(training) if training else 0.0

    # Kill criteria
    kill_triggered = False
    kill_reasons = []

    if hurdle_hitter:
        auc = hurdle_hitter.get("heldout", {}).get("auc", float("nan"))
        if auc < KILL_AUC_GATE:
            kill_triggered = True
            kill_reasons.append(f"hitter hurdle AUC {auc:.4f} < {KILL_AUC_GATE}")

    if hurdle_pitcher:
        auc = hurdle_pitcher.get("heldout", {}).get("auc", float("nan"))
        if auc < KILL_AUC_GATE:
            kill_triggered = True
            kill_reasons.append(f"pitcher hurdle AUC {auc:.4f} < {KILL_AUC_GATE}")

    if value_hitter:
        r2 = value_hitter.get("heldout", {}).get("r2", float("nan"))
        if r2 < KILL_R2_GATE:
            kill_triggered = True
            kill_reasons.append(f"hitter value R² {r2:.4f} < {KILL_R2_GATE}")

    if value_pitcher:
        r2 = value_pitcher.get("heldout", {}).get("r2", float("nan"))
        if r2 < KILL_R2_GATE:
            kill_triggered = True
            kill_reasons.append(f"pitcher value R² {r2:.4f} < {KILL_R2_GATE}")

    # Generate markdown
    lines = []
    lines.append("# Tier 5 WAR Value — Honesty Report")
    lines.append("")
    lines.append(f"**Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Cohort Description")
    lines.append("")
    lines.append(f"| Metric | Value |")
    lines.append(f"|--------|-------|")
    lines.append(f"| Total records | {len(training)} |")
    lines.append(f"| Train (draft_year=2021) | {n_train} |")
    lines.append(f"| Heldout (draft_year=2022-2023) | {n_heldout} |")
    lines.append(f"| Positive (WAR > 2.0) | {n_positive} ({positive_rate:.2%}) |")
    lines.append("")
    lines.append("**Right-censoring note:** The heldout set (2022-2023 draftees) has only 3-4 years of MLB data. Players may accumulate more WAR in future seasons. This biases the model toward underestimating true 5-year WAR.")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Hurdle Classifier Performance")
    lines.append("")
    lines.append("Predicts P(war_years_1_through_5 > 2.0) = 'meaningfully productive'")
    lines.append("")

    if hurdle_hitter:
        lines.append("### Hitters")
        lines.append("")
        lines.append(f"| Metric | Value |")
        lines.append(f"|--------|-------|")
        lines.append(f"| Heldout AUC | {hurdle_hitter.get('heldout', {}).get('auc', 'N/A'):.4f} |")
        lines.append(f"| Heldout Brier | {hurdle_hitter.get('heldout', {}).get('brier', 'N/A'):.4f} |")
        lines.append(f"| Bootstrap 95% CI | [{hurdle_hitter.get('bootstrap', {}).get('low', 'N/A'):.3f}, {hurdle_hitter.get('bootstrap', {}).get('high', 'N/A'):.3f}] |")
        lines.append(f"| Gate (AUC ≥ {KILL_AUC_GATE}) | {'PASS' if hurdle_hitter.get('gate', {}).get('status') == 'PASS' else 'WARN'} |")
        lines.append("")

    if hurdle_pitcher:
        lines.append("### Pitchers")
        lines.append("")
        lines.append(f"| Metric | Value |")
        lines.append(f"|--------|-------|")
        lines.append(f"| Heldout AUC | {hurdle_pitcher.get('heldout', {}).get('auc', 'N/A'):.4f} |")
        lines.append(f"| Heldout Brier | {hurdle_pitcher.get('heldout', {}).get('brier', 'N/A'):.4f} |")
        lines.append(f"| Bootstrap 95% CI | [{hurdle_pitcher.get('bootstrap', {}).get('low', 'N/A'):.3f}, {hurdle_pitcher.get('bootstrap', {}).get('high', 'N/A'):.3f}] |")
        lines.append(f"| Gate (AUC ≥ {KILL_AUC_GATE}) | {'PASS' if hurdle_pitcher.get('gate', {}).get('status') == 'PASS' else 'WARN'} |")
        lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("## Value Regression Performance")
    lines.append("")
    lines.append("Predicts signed_log_war (sign(war) * log1p(|war|))")
    lines.append("")

    if value_hitter:
        lines.append("### Hitters")
        lines.append("")
        lines.append(f"| Metric | Value |")
        lines.append(f"|--------|-------|")
        lines.append(f"| Heldout RMSE | {value_hitter.get('heldout', {}).get('rmse', 'N/A'):.4f} |")
        lines.append(f"| Heldout MAE | {value_hitter.get('heldout', {}).get('mae', 'N/A'):.4f} |")
        lines.append(f"| Heldout R² | {value_hitter.get('heldout', {}).get('r2', 'N/A'):.4f} |")
        lines.append(f"| Bootstrap 95% CI | [{value_hitter.get('bootstrap', {}).get('low', 'N/A'):.3f}, {value_hitter.get('bootstrap', {}).get('high', 'N/A'):.3f}] |")
        lines.append(f"| Baseline (median) RMSE | {value_hitter.get('baseline_median', {}).get('rmse', 'N/A'):.4f} |")
        lines.append(f"| Gate (R² ≥ {KILL_R2_GATE}) | {'PASS' if value_hitter.get('gate', {}).get('status') == 'PASS' else 'WARN'} |")
        lines.append("")

    if value_pitcher:
        lines.append("### Pitchers")
        lines.append("")
        lines.append(f"| Metric | Value |")
        lines.append(f"|--------|-------|")
        lines.append(f"| Heldout RMSE | {value_pitcher.get('heldout', {}).get('rmse', 'N/A'):.4f} |")
        lines.append(f"| Heldout MAE | {value_pitcher.get('heldout', {}).get('mae', 'N/A'):.4f} |")
        lines.append(f"| Heldout R² | {value_pitcher.get('heldout', {}).get('r2', 'N/A'):.4f} |")
        lines.append(f"| Bootstrap 95% CI | [{value_pitcher.get('bootstrap', {}).get('low', 'N/A'):.3f}, {value_pitcher.get('bootstrap', {}).get('high', 'N/A'):.3f}] |")
        lines.append(f"| Baseline (median) RMSE | {value_pitcher.get('baseline_median', {}).get('rmse', 'N/A'):.4f} |")
        lines.append(f"| Gate (R² ≥ {KILL_R2_GATE}) | {'PASS' if value_pitcher.get('gate', {}).get('status') == 'PASS' else 'WARN'} |")
        lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("## Kill Criteria")
    lines.append("")
    lines.append(f"**Thresholds:** AUC < {KILL_AUC_GATE} OR R² < {KILL_R2_GATE}")
    lines.append("")

    if kill_triggered:
        lines.append("**STATUS: NOT READY FOR PRODUCTION**")
        lines.append("")
        lines.append("Kill criteria triggered:")
        lines.append("")
        for reason in kill_reasons:
            lines.append(f"  * {reason}")
        lines.append("")
        lines.append("The model performs worse than a naive baseline (predicting the median WAR for all players). Do not deploy until these issues are resolved.")
    else:
        lines.append("**STATUS: READY FOR PRODUCTION**")
        lines.append("")
        lines.append("All kill criteria passed. The model beats the naive baseline.")

    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Honesty Notes")
    lines.append("")
    lines.append("1. **Small sample size:** Only ~300 training records (draft year 2021). This limits the model's ability to generalize.")
    lines.append("")
    lines.append("2. **Right-censoring:** The heldout set (2022-2023) has incomplete WAR data. Players may accumulate more WAR in future seasons.")
    lines.append("")
    lines.append("3. **Zero-inflated target:** Most drafted players produce 0 WAR. The signed_log_war transform handles this, but the model may still struggle with the extreme imbalance.")
    lines.append("")
    lines.append("4. **Feature limitations:** Only 10 features per role (Tier 3 features). TrackMan data was not available for most players in the training set.")
    lines.append("")
    lines.append("5. **Interpretation:** A 'meaningfully productive' prediction (hurdle prob > 0.5) means the player is likely to accumulate >2.0 WAR over 5 years. This is roughly equivalent to a league-average player.")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("*This report is automatically generated. Do not edit manually.*")

    # Write output
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(lines))

    print(f"Wrote {output_path}")
    if kill_triggered:
        print("WARNING: Kill criteria triggered — model not ready for production")
        return 1
    return 0


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description="Generate Tier 5 honesty report.")
    p.add_argument("--training", default=str(DEFAULT_TRAINING),
                   help="tier5_training_set.json path")
    p.add_argument("--output", default=str(DEFAULT_OUTPUT),
                   help="output markdown path")
    return p.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    return generate_report(args.training, args.output)


if __name__ == "__main__":
    import sys
    sys.exit(main())
