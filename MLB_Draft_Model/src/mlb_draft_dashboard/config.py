from __future__ import annotations

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
EXPORTS_DIR = BASE_DIR / "exports" / "dashboard"
STATE_DB_PATH = BASE_DIR / "app_state.sqlite"

ROLE_FIT_OPTIONS = ["Unassigned", "Early Round", "Value", "Sleeper", "Pass"]

BOARD_DEFAULTS = {
    "hitters": {
        "weights": {"reach": 35.0, "impact": 35.0, "contact": 20.0, "risk": 10.0},
        "components": ["reach_score", "impact_score", "contact_score"],
        "risk_column": "risk_score",
        "search_columns": ["player_name", "school_name", "conference"],
    },
    "pitchers": {
        "weights": {"reach": 30.0, "stuff": 40.0, "command": 20.0, "risk": 10.0},
        "components": ["reach_score", "stuff_score", "command_score"],
        "risk_column": "risk_score",
        "search_columns": ["player_name", "school_name", "conference"],
    },
}

EXPORT_FILES = {
    "hitters_board": "hitters_board.parquet",
    "pitchers_board": "pitchers_board.parquet",
    "player_trends": "player_trends.parquet",
    "hitter_bbe_detail": "hitter_bbe_detail.parquet",
    "pitcher_pitchtype_detail": "pitcher_pitchtype_detail.parquet",
    "benchmarks_acc_sec": "benchmarks_acc_sec.parquet",
    "explanations": "explanations.parquet",
    "diagnostics": "diagnostics.parquet",
    "qa": "qa.parquet",
}

DETAIL_METRICS = {
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
}
