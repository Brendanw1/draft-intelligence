from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .config import EXPORT_FILES


def _pct(rng: np.random.Generator, low: float, high: float) -> float:
    return float(np.round(rng.uniform(low, high), 2))


def write_demo_exports(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    export_ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

    hitter_rows = []
    pitcher_rows = []
    explanations = []
    trends = []
    bbe_rows = []
    pitchtype_rows = []
    benchmark_rows = []
    diagnostics_rows = []
    qa_rows = []

    hitter_names = [
        ("h_01", "Ethan Miller", "Virginia Tech", "ACC", "L", "R"),
        ("h_02", "Luke Carter", "Florida", "SEC", "R", "R"),
        ("h_03", "Miles Rivera", "Wake Forest", "ACC", "L", "R"),
        ("h_04", "Cole Jackson", "Tennessee", "SEC", "R", "R"),
        ("h_05", "James Holloway", "North Carolina", "ACC", "L", "R"),
    ]
    pitcher_names = [
        ("p_01", "Carson Reid", "Virginia Tech", "ACC", "R"),
        ("p_02", "Jack Turner", "Florida", "SEC", "R"),
        ("p_03", "Mason Hayes", "Wake Forest", "ACC", "L"),
        ("p_04", "Gavin Brooks", "Arkansas", "SEC", "R"),
        ("p_05", "Noah Bennett", "Georgia Tech", "ACC", "R"),
    ]

    seasons = [2024, 2025, 2026]

    for index, (player_uid, player_name, school_name, conference, bats, throws) in enumerate(hitter_names, start=1):
        class_year = ["SO", "JR", "JR", "SR", "SO"][index - 1]
        reach = _pct(rng, 55, 90)
        impact = _pct(rng, 50, 96)
        contact = _pct(rng, 45, 92)
        risk = _pct(rng, 8, 38)
        draft_value = np.clip(0.4 * reach + 0.35 * impact + 0.25 * contact - 0.15 * risk, 0, 100)
        hitter_rows.append(
            {
                "player_uid": player_uid,
                "player_name": player_name,
                "team_code": school_name[:3].upper(),
                "school_name": school_name,
                "conference": conference,
                "season": 2026,
                "bats": bats,
                "throws": throws,
                "class_year": class_year,
                "production_rank": index,
                "draft_value_score": round(draft_value, 2),
                "reach_score": reach,
                "impact_score": impact,
                "contact_score": contact,
                "risk_score": risk,
                "plate_events": int(rng.integers(85, 210)),
                "bbe_count": int(rng.integers(28, 95)),
                "p90_ev_wood_adj": round(rng.uniform(98, 108), 2),
                "avg_ev_wood_adj": round(rng.uniform(87, 96), 2),
                "barrel_rate_proxy_wood_adj": round(rng.uniform(6, 19), 2),
                "contact_rate": round(rng.uniform(68, 91), 2),
                "whiff_rate": round(rng.uniform(12, 34), 2),
                "chase_rate": round(rng.uniform(18, 39), 2),
                "trend_delta": round(rng.uniform(-1.8, 4.2), 2),
                "data_completeness_score": round(rng.uniform(82, 100), 2),
                "one_season_only_flag": False,
                "missing_critical_count": int(rng.integers(0, 2)),
                "export_ts": export_ts,
            }
        )
        explanations.append(
            {
                "player_uid": player_uid,
                "role": "hitters",
                "sample_size_text": f"{hitter_rows[-1]['plate_events']} plate events / {hitter_rows[-1]['bbe_count']} BBE",
                "data_completeness_score": hitter_rows[-1]["data_completeness_score"],
                "match_confidence": round(rng.uniform(0.84, 0.99), 2),
                "positive_driver_1": f"Impact score {impact:.1f}",
                "positive_driver_2": f"P90 EV wood-adj {hitter_rows[-1]['p90_ev_wood_adj']:.1f}",
                "negative_driver_1": f"Whiff rate {hitter_rows[-1]['whiff_rate']:.1f}%",
                "negative_driver_2": f"Chase rate {hitter_rows[-1]['chase_rate']:.1f}%",
                "warning_text": "One-season-only warning will appear when trend history is missing.",
            }
        )
        for season in seasons:
            trends.extend(
                [
                    {"player_uid": player_uid, "player_name": player_name, "role": "hitters", "season": season, "metric_key": "p90_ev_wood_adj", "metric_label": "P90 EV (Wood Adj)", "metric_value": round(rng.uniform(97, 108), 2)},
                    {"player_uid": player_uid, "player_name": player_name, "role": "hitters", "season": season, "metric_key": "contact_rate", "metric_label": "Contact Rate", "metric_value": round(rng.uniform(66, 92), 2)},
                    {"player_uid": player_uid, "player_name": player_name, "role": "hitters", "season": season, "metric_key": "chase_rate", "metric_label": "Chase Rate", "metric_value": round(rng.uniform(18, 40), 2)},
                ]
            )
        for _ in range(36):
            bbe_rows.append(
                {
                    "player_uid": player_uid,
                    "player_name": player_name,
                    "season": 2026,
                    "exit_speed": round(rng.uniform(72, 111), 2),
                    "angle": round(rng.uniform(-15, 38), 2),
                    "direction": round(rng.uniform(-45, 45), 2),
                }
            )

    for index, (player_uid, player_name, school_name, conference, throws) in enumerate(pitcher_names, start=1):
        class_year = ["JR", "SO", "JR", "SR", "SO"][index - 1]
        reach = _pct(rng, 50, 92)
        stuff = _pct(rng, 52, 98)
        command = _pct(rng, 40, 89)
        risk = _pct(rng, 10, 42)
        draft_value = np.clip(0.35 * reach + 0.4 * stuff + 0.25 * command - 0.15 * risk, 0, 100)
        pitcher_rows.append(
            {
                "player_uid": player_uid,
                "player_name": player_name,
                "team_code": school_name[:3].upper(),
                "school_name": school_name,
                "conference": conference,
                "season": 2026,
                "throws": throws,
                "class_year": class_year,
                "production_rank": index,
                "draft_value_score": round(draft_value, 2),
                "reach_score": reach,
                "stuff_score": stuff,
                "command_score": command,
                "risk_score": risk,
                "pitch_count": int(rng.integers(180, 520)),
                "avg_fb_velo": round(rng.uniform(89, 97), 2),
                "max_fb_velo": round(rng.uniform(92, 100), 2),
                "avg_ivb": round(rng.uniform(12, 21), 2),
                "avg_hb": round(rng.uniform(-16, 16), 2),
                "extension": round(rng.uniform(5.8, 7.1), 2),
                "csw_pct": round(rng.uniform(24, 37), 2),
                "whiff_pct": round(rng.uniform(18, 39), 2),
                "zone_pct": round(rng.uniform(41, 59), 2),
                "arsenal_count": int(rng.integers(3, 6)),
                "trend_delta": round(rng.uniform(-1.4, 3.4), 2),
                "data_completeness_score": round(rng.uniform(83, 100), 2),
                "one_season_only_flag": False,
                "missing_critical_count": int(rng.integers(0, 2)),
                "export_ts": export_ts,
            }
        )
        explanations.append(
            {
                "player_uid": player_uid,
                "role": "pitchers",
                "sample_size_text": f"{pitcher_rows[-1]['pitch_count']} pitches",
                "data_completeness_score": pitcher_rows[-1]["data_completeness_score"],
                "match_confidence": round(rng.uniform(0.84, 0.99), 2),
                "positive_driver_1": f"Stuff score {stuff:.1f}",
                "positive_driver_2": f"Avg FB velo {pitcher_rows[-1]['avg_fb_velo']:.1f}",
                "negative_driver_1": f"Risk score {risk:.1f}",
                "negative_driver_2": f"Zone rate {pitcher_rows[-1]['zone_pct']:.1f}%",
                "warning_text": "One-season-only warning will appear when trend history is missing.",
            }
        )
        for season in seasons:
            trends.extend(
                [
                    {"player_uid": player_uid, "player_name": player_name, "role": "pitchers", "season": season, "metric_key": "avg_fb_velo", "metric_label": "Avg FB Velo", "metric_value": round(rng.uniform(88, 97), 2)},
                    {"player_uid": player_uid, "player_name": player_name, "role": "pitchers", "season": season, "metric_key": "csw_pct", "metric_label": "CSW%", "metric_value": round(rng.uniform(22, 38), 2)},
                    {"player_uid": player_uid, "player_name": player_name, "role": "pitchers", "season": season, "metric_key": "zone_pct", "metric_label": "Zone%", "metric_value": round(rng.uniform(40, 60), 2)},
                ]
            )
        for pitch_type in ["Fastball", "Slider", "ChangeUp"]:
            pitchtype_rows.append(
                {
                    "player_uid": player_uid,
                    "player_name": player_name,
                    "season": 2026,
                    "pitch_type": pitch_type,
                    "usage_pct": round(rng.uniform(15, 55), 2),
                    "avg_velo": round(rng.uniform(80, 97), 2),
                    "avg_ivb": round(rng.uniform(3, 21), 2),
                    "avg_hb": round(rng.uniform(-18, 18), 2),
                    "extension": round(rng.uniform(5.7, 7.1), 2),
                    "rel_height": round(rng.uniform(5.0, 6.4), 2),
                    "rel_side": round(rng.uniform(-2.5, 2.5), 2),
                    "zone_pct": round(rng.uniform(39, 61), 2),
                    "whiff_pct": round(rng.uniform(12, 42), 2),
                    "csw_pct": round(rng.uniform(20, 39), 2),
                    "hard_hit_allowed_pct": round(rng.uniform(16, 44), 2),
                }
            )

    for role, metric_keys in {
        "hitters": [
            ("reach_score", "Reach Score"),
            ("impact_score", "Impact Score"),
            ("contact_score", "Contact Score"),
            ("risk_score", "Risk Score"),
            ("p90_ev_wood_adj", "P90 EV (Wood Adj)"),
            ("avg_ev_wood_adj", "Avg EV (Wood Adj)"),
            ("barrel_rate_proxy_wood_adj", "Barrel Proxy"),
            ("contact_rate", "Contact Rate"),
            ("whiff_rate", "Whiff Rate"),
            ("chase_rate", "Chase Rate"),
        ],
        "pitchers": [
            ("reach_score", "Reach Score"),
            ("stuff_score", "Stuff Score"),
            ("command_score", "Command Score"),
            ("risk_score", "Risk Score"),
            ("avg_fb_velo", "Avg FB Velo"),
            ("max_fb_velo", "Max FB Velo"),
            ("avg_ivb", "Avg IVB"),
            ("avg_hb", "Avg HB"),
            ("extension", "Extension"),
            ("csw_pct", "CSW%"),
            ("whiff_pct", "Whiff%"),
            ("zone_pct", "Zone%"),
        ],
    }.items():
        source = pd.DataFrame(hitter_rows if role == "hitters" else pitcher_rows)
        for metric_key, metric_label in metric_keys:
            benchmark_rows.append(
                {
                    "season": 2026,
                    "role": role,
                    "metric_key": metric_key,
                    "benchmark_scope": "ACC_SEC",
                    "benchmark_value": round(source[metric_key].mean(), 2),
                    "benchmark_label": metric_label,
                }
            )
        diagnostics_rows.append({"role": role, "record_type": "metric", "section": "availability", "label": "Model diagnostics status", "value_num": np.nan, "value_text": "Deterministic component-score export; calibration/ROC can be added once trained model outputs are exported.", "player_uid": None, "player_name": None})
        for component in ["draft_value_score"] + [column for column in source.columns if column.endswith("_score")]:
            diagnostics_rows.append({"role": role, "record_type": "metric", "section": "component_average", "label": component, "value_num": round(source[component].mean(), 2), "value_text": None, "player_uid": None, "player_name": None})
        distribution_source = source["draft_value_score"]
        bins = np.linspace(distribution_source.min(), distribution_source.max(), 6)
        counts, edges = np.histogram(distribution_source, bins=bins)
        for count, left, right in zip(counts, edges[:-1], edges[1:]):
            diagnostics_rows.append({"role": role, "record_type": "distribution", "section": "score_distribution", "label": "Draft Value Score", "bucket_label": f"{left:.1f}-{right:.1f}", "bucket_start": round(float(left), 2), "bucket_end": round(float(right), 2), "value_num": int(count), "value_text": None, "player_uid": None, "player_name": None})
        high_risk = source.sort_values(["risk_score", "draft_value_score"], ascending=[False, False]).head(3)
        for _, row in high_risk.iterrows():
            diagnostics_rows.append({"role": role, "record_type": "example", "section": "high_score_high_risk_examples", "label": "High-risk profile", "value_num": float(row["risk_score"]), "value_text": f"Draft value {row['draft_value_score']:.1f}", "player_uid": row["player_uid"], "player_name": row["player_name"]})

    qa_rows.extend(
        [
            {"record_type": "metric", "section": "freshness", "label": "Export Timestamp", "value_num": np.nan, "value_text": export_ts, "role": None, "player_uid": None, "player_name": None},
            {"record_type": "metric", "section": "coverage", "label": "Hitters on Board", "value_num": len(hitter_rows), "value_text": None, "role": "hitters", "player_uid": None, "player_name": None},
            {"record_type": "metric", "section": "coverage", "label": "Pitchers on Board", "value_num": len(pitcher_rows), "value_text": None, "role": "pitchers", "player_uid": None, "player_name": None},
            {"record_type": "metric", "section": "coverage", "label": "Benchmark Scope", "value_num": np.nan, "value_text": "ACC/SEC same-season role averages", "role": None, "player_uid": None, "player_name": None},
        ]
    )

    pd.DataFrame(hitter_rows).to_parquet(output_dir / EXPORT_FILES["hitters_board"], index=False)
    pd.DataFrame(pitcher_rows).to_parquet(output_dir / EXPORT_FILES["pitchers_board"], index=False)
    pd.DataFrame(trends).to_parquet(output_dir / EXPORT_FILES["player_trends"], index=False)
    pd.DataFrame(bbe_rows).to_parquet(output_dir / EXPORT_FILES["hitter_bbe_detail"], index=False)
    pd.DataFrame(pitchtype_rows).to_parquet(output_dir / EXPORT_FILES["pitcher_pitchtype_detail"], index=False)
    pd.DataFrame(benchmark_rows).to_parquet(output_dir / EXPORT_FILES["benchmarks_acc_sec"], index=False)
    pd.DataFrame(explanations).to_parquet(output_dir / EXPORT_FILES["explanations"], index=False)
    pd.DataFrame(diagnostics_rows).to_parquet(output_dir / EXPORT_FILES["diagnostics"], index=False)
    pd.DataFrame(qa_rows).to_parquet(output_dir / EXPORT_FILES["qa"], index=False)
