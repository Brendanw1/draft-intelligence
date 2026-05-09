from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pandas as pd
import plotly.express as px
import streamlit as st

from .config import BASE_DIR, BOARD_DEFAULTS, DETAIL_METRICS, EXPORTS_DIR, ROLE_FIT_OPTIONS, STATE_DB_PATH
from .contracts import REQUIRED_COLUMNS, missing_columns, missing_export_files
from .data_access import load_dashboard_bundle, merge_notes
from .ranking import add_rank_columns, compute_custom_rank
from .state_store import StateStore


@st.cache_data(show_spinner=False)
def cached_bundle(exports_dir: str) -> Dict[str, pd.DataFrame]:
    return load_dashboard_bundle(Path(exports_dir))


def _format_percent(value: float) -> str:
    if pd.isna(value):
        return "NA"
    return f"{value:.1f}%"


def _ensure_session_defaults() -> None:
    st.session_state.setdefault("active_page", "Hitters Board")
    st.session_state.setdefault("selected_player_uid", None)
    st.session_state.setdefault("selected_role", "hitters")
    st.session_state.setdefault("hitters_weights", BOARD_DEFAULTS["hitters"]["weights"].copy())
    st.session_state.setdefault("pitchers_weights", BOARD_DEFAULTS["pitchers"]["weights"].copy())


def _show_missing_exports(exports_dir: Path) -> None:
    st.title("MLB Draft Dashboard")
    st.warning("Dashboard exports were not found yet.")
    st.markdown(
        f"""
        Expected Parquet exports live in [`{exports_dir}`]({exports_dir}).

        To get started:
        1. Run `python scripts/generate_demo_exports.py` for sample data, or
        2. Run `R/export_dashboard_data.R` against your SQLite source.
        """
    )


def _validate_bundle(bundle: Dict[str, pd.DataFrame]) -> List[str]:
    issues: List[str] = []
    for table_name, required in REQUIRED_COLUMNS.items():
        frame = bundle.get(table_name, pd.DataFrame())
        if frame.empty:
            if table_name in {"diagnostics", "qa", "benchmarks_acc_sec"}:
                continue
        missing = missing_columns(frame.columns, required)
        if missing:
            issues.append(f"`{table_name}` missing columns: {', '.join(missing)}")
    return issues


def _select_page() -> str:
    page = st.sidebar.radio(
        "Page",
        ["Hitters Board", "Pitchers Board", "Player Detail", "Model Diagnostics", "Data QA"],
        index=["Hitters Board", "Pitchers Board", "Player Detail", "Model Diagnostics", "Data QA"].index(
            st.session_state["active_page"]
        ),
    )
    st.session_state["active_page"] = page
    return page


def _build_player_index(bundle: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    players = []
    for role_key, table_name in [("hitters", "hitters_board"), ("pitchers", "pitchers_board")]:
        frame = bundle[table_name]
        if frame.empty:
            continue
        players.append(
            frame[["player_uid", "player_name", "school_name", "conference", "season"]]
            .assign(role=role_key)
            .drop_duplicates()
        )
    return pd.concat(players, ignore_index=True) if players else pd.DataFrame()


def _default_filters(board_type: str, frame: pd.DataFrame) -> Dict[str, object]:
    season_values = sorted(frame["season"].dropna().unique().tolist()) if not frame.empty else []
    return {
        "season": season_values,
        "conference": [],
        "school_name": [],
        "handedness": [],
        "class_year": [],
        "role_fit": [],
        "favorites_only": False,
        "notes_only": False,
        "search_text": "",
        "score_range": [
            float(frame["draft_value_score"].min()) if not frame.empty else 0.0,
            float(frame["draft_value_score"].max()) if not frame.empty else 100.0,
        ],
        "sample_min": 0,
        "secondary_sample_min": 0,
    }


def _apply_saved_view(board_type: str, view_payload: Dict[str, object], frame: pd.DataFrame) -> None:
    st.session_state[f"{board_type}_weights"] = view_payload["weights"]
    st.session_state[f"{board_type}_filters"] = {**_default_filters(board_type, frame), **view_payload["filters"]}


def _render_saved_views(board_type: str, frame: pd.DataFrame, store: StateStore) -> None:
    saved_views = store.list_views(board_type)
    if not saved_views:
        return

    selected_view = st.selectbox(
        f"Load saved {board_type} view",
        options=[""] + saved_views,
        key=f"{board_type}_saved_view_selector",
    )
    if selected_view:
        payload = store.get_view(board_type, selected_view)
        if payload is not None:
            _apply_saved_view(board_type, payload, frame)
            st.success(f"Loaded saved view: {selected_view}")


def _render_filters(board_type: str, frame: pd.DataFrame) -> Dict[str, object]:
    filters = st.session_state.setdefault(f"{board_type}_filters", _default_filters(board_type, frame))

    handedness_column = "bats" if board_type == "hitters" else "throws"
    sample_label = "Min Plate Events" if board_type == "hitters" else "Min Pitch Count"
    sample_column = "plate_events" if board_type == "hitters" else "pitch_count"
    secondary_label = "Min BBE" if board_type == "hitters" else "Min Arsenal Count"
    secondary_column = "bbe_count" if board_type == "hitters" else "arsenal_count"

    with st.expander("Board Filters", expanded=True):
        col1, col2, col3 = st.columns(3)
        filters["season"] = col1.multiselect("Season", sorted(frame["season"].dropna().unique()), default=filters["season"])
        filters["conference"] = col2.multiselect(
            "Conference", sorted(value for value in frame["conference"].dropna().unique() if value), default=filters["conference"]
        )
        filters["school_name"] = col3.multiselect(
            "School", sorted(value for value in frame["school_name"].dropna().unique() if value), default=filters["school_name"]
        )

        col4, col5, col6 = st.columns(3)
        filters["handedness"] = col4.multiselect(
            "Bats / Throws" if board_type == "hitters" else "Throws",
            sorted(value for value in frame[handedness_column].dropna().unique() if value),
            default=filters["handedness"],
        )
        filters["class_year"] = col5.multiselect(
            "Class Year", sorted(value for value in frame["class_year"].dropna().unique() if value), default=filters["class_year"]
        )
        filters["role_fit"] = col6.multiselect("Role Fit", ROLE_FIT_OPTIONS, default=filters["role_fit"])

        col7, col8, col9 = st.columns(3)
        filters["favorites_only"] = col7.checkbox("Favorites Only", value=filters["favorites_only"])
        filters["notes_only"] = col8.checkbox("Has Notes Only", value=filters["notes_only"])
        filters["search_text"] = col9.text_input("Search", value=filters["search_text"])

        min_score = float(frame["draft_value_score"].min())
        max_score = float(frame["draft_value_score"].max())
        filters["score_range"] = st.slider(
            "Draft Value Score Range",
            min_value=float(min_score),
            max_value=float(max_score),
            value=tuple(filters["score_range"]),
        )

        col10, col11 = st.columns(2)
        filters["sample_min"] = int(
            col10.number_input(sample_label, min_value=0, value=int(filters["sample_min"]), step=1)
        )
        filters["secondary_sample_min"] = int(
            col11.number_input(secondary_label, min_value=0, value=int(filters["secondary_sample_min"]), step=1)
        )

    filtered = frame.copy()
    if filters["season"]:
        filtered = filtered[filtered["season"].isin(filters["season"])]
    if filters["conference"]:
        filtered = filtered[filtered["conference"].isin(filters["conference"])]
    if filters["school_name"]:
        filtered = filtered[filtered["school_name"].isin(filters["school_name"])]
    if filters["handedness"]:
        filtered = filtered[filtered[handedness_column].isin(filters["handedness"])]
    if filters["class_year"]:
        filtered = filtered[filtered["class_year"].isin(filters["class_year"])]
    if filters["role_fit"]:
        filtered = filtered[filtered["role_fit"].isin(filters["role_fit"])]
    if filters["favorites_only"]:
        filtered = filtered[filtered["is_favorite"]]
    if filters["notes_only"]:
        filtered = filtered[filtered["has_note"]]
    if filters["search_text"]:
        search_value = filters["search_text"].lower().strip()
        filtered = filtered[
            filtered["player_name"].fillna("").str.lower().str.contains(search_value)
            | filtered["school_name"].fillna("").str.lower().str.contains(search_value)
            | filtered["conference"].fillna("").str.lower().str.contains(search_value)
        ]

    filtered = filtered[
        filtered["draft_value_score"].between(filters["score_range"][0], filters["score_range"][1], inclusive="both")
    ]
    filtered = filtered[filtered[sample_column] >= filters["sample_min"]]
    filtered = filtered[filtered[secondary_column] >= filters["secondary_sample_min"]]
    return filtered


def _render_weight_panel(board_type: str, store: StateStore, filtered: pd.DataFrame) -> Dict[str, float]:
    default_weights = BOARD_DEFAULTS[board_type]["weights"]
    current_weights = st.session_state.get(f"{board_type}_weights", default_weights.copy())

    with st.expander("Live Board Weights", expanded=True):
        st.caption("These sliders re-rank precomputed component scores only. They do not retrain the underlying model.")
        col1, col2, col3, col4 = st.columns(4)
        weight_keys = [key for key in current_weights.keys()]
        current_weights[weight_keys[0]] = float(col1.slider(weight_keys[0].title(), 0, 100, int(current_weights[weight_keys[0]])))
        current_weights[weight_keys[1]] = float(col2.slider(weight_keys[1].title(), 0, 100, int(current_weights[weight_keys[1]])))
        current_weights[weight_keys[2]] = float(col3.slider(weight_keys[2].title(), 0, 100, int(current_weights[weight_keys[2]])))
        current_weights[weight_keys[3]] = float(col4.slider(weight_keys[3].title(), 0, 100, int(current_weights[weight_keys[3]])))
        st.session_state[f"{board_type}_weights"] = current_weights

        col5, col6, col7 = st.columns([1, 1, 2])
        if col5.button("Reset to Production", key=f"{board_type}_reset_weights"):
            st.session_state[f"{board_type}_weights"] = default_weights.copy()
            st.rerun()

        show_production = col6.toggle("Show Production vs Custom Rank", value=True, key=f"{board_type}_show_prod")

        with col7:
            with st.form(f"{board_type}_save_view_form"):
                view_name = st.text_input("Save current view")
                save_view = st.form_submit_button("Save View")
                if save_view and view_name.strip():
                    filters = st.session_state.get(f"{board_type}_filters", {})
                    store.save_view(board_type, view_name.strip(), current_weights, filters)
                    st.success(f"Saved {board_type} view: {view_name.strip()}")

    st.session_state[f"{board_type}_show_production"] = show_production
    return current_weights


def _prepare_board(board_type: str, bundle: Dict[str, pd.DataFrame], store: StateStore) -> pd.DataFrame:
    board_df = bundle[f"{board_type}_board"]
    notes_df = store.get_player_notes()
    merged = merge_notes(board_df, notes_df)
    filtered = _render_filters(board_type, merged)
    weights = _render_weight_panel(board_type, store, filtered)
    positive_components = {key: value for key, value in zip(weights.keys(), BOARD_DEFAULTS[board_type]["components"] + [BOARD_DEFAULTS[board_type]["risk_column"]])}
    custom = compute_custom_rank(
        filtered,
        positive_components={key: column for key, column in positive_components.items() if key != "risk"},
        risk_component=BOARD_DEFAULTS[board_type]["risk_column"],
        raw_weights=weights,
    )
    ranked = add_rank_columns(filtered, custom)
    return ranked


def _display_board(board_type: str, ranked: pd.DataFrame) -> None:
    st.title("Hitters Board" if board_type == "hitters" else "Pitchers Board")
    st.caption("Production rank stays fixed. Custom rank reorders the board using your live weights.")

    if ranked.empty:
        st.info("No players match the current filters.")
        return

    ranked = ranked.copy()
    ranked["favorite"] = ranked["is_favorite"].map(lambda value: "★" if value else "")
    ranked["notes"] = ranked["has_note"].map(lambda value: "Yes" if value else "")
    if not st.session_state.get(f"{board_type}_show_production", True):
        ranked = ranked.drop(columns=["production_rank"], errors="ignore")

    if board_type == "hitters":
        columns = [
            "favorite",
            "player_name",
            "school_name",
            "conference",
            "bats",
            "throws",
            "class_year",
            "custom_rank",
            "reach_score",
            "impact_score",
            "contact_score",
            "risk_score",
            "p90_ev_wood_adj",
            "avg_ev_wood_adj",
            "barrel_rate_proxy_wood_adj",
            "contact_rate",
            "whiff_rate",
            "chase_rate",
            "trend_delta",
            "role_fit",
            "notes",
        ]
    else:
        columns = [
            "favorite",
            "player_name",
            "school_name",
            "conference",
            "throws",
            "class_year",
            "custom_rank",
            "reach_score",
            "stuff_score",
            "command_score",
            "risk_score",
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
            "role_fit",
            "notes",
        ]

    if st.session_state.get(f"{board_type}_show_production", True):
        columns.insert(7 if board_type == "hitters" else 6, "production_rank")

    display = ranked[["player_uid"] + columns].rename(
        columns={
            "player_name": "Player",
            "school_name": "School",
            "conference": "Conf",
            "class_year": "Class",
            "production_rank": "Prod Rank",
            "custom_rank": "Custom Rank",
            "reach_score": "Reach",
            "impact_score": "Impact",
            "contact_score": "Contact",
            "risk_score": "Risk",
            "stuff_score": "Stuff",
            "command_score": "Command",
            "p90_ev_wood_adj": "P90 EV WAdj",
            "avg_ev_wood_adj": "Avg EV WAdj",
            "barrel_rate_proxy_wood_adj": "Barrel WAdj",
            "contact_rate": "Contact%",
            "whiff_rate": "Whiff%",
            "chase_rate": "Chase%",
            "avg_fb_velo": "Avg FB",
            "max_fb_velo": "Max FB",
            "avg_ivb": "IVB",
            "avg_hb": "HB",
            "extension": "Ext",
            "csw_pct": "CSW%",
            "whiff_pct": "Whiff%",
            "zone_pct": "Zone%",
            "arsenal_count": "Arsenal",
            "trend_delta": "Trend",
            "role_fit": "Role Fit",
            "favorite": "Fav",
            "notes": "Notes",
        }
    )

    selection_payload = None
    try:
        selection_payload = st.dataframe(
            display.drop(columns=["player_uid"]),
            use_container_width=True,
            hide_index=True,
            on_select="rerun",
            selection_mode="single-row",
        )
    except TypeError:
        st.dataframe(display.drop(columns=["player_uid"]), use_container_width=True, hide_index=True)

    if selection_payload is not None and selection_payload.selection.rows:
        selected_index = selection_payload.selection.rows[0]
        selected_player_uid = display.iloc[selected_index]["player_uid"]
        st.session_state["selected_player_uid"] = selected_player_uid
        st.session_state["selected_role"] = board_type
        st.session_state["active_page"] = "Player Detail"
        st.rerun()

    options = ranked[["player_uid", "player_name", "school_name"]].copy()
    options["label"] = options["player_name"] + " | " + options["school_name"]
    selected_label = st.selectbox(f"Open {board_type[:-1]} detail", [""] + options["label"].tolist(), index=0)
    if selected_label:
        selected_uid = options.loc[options["label"] == selected_label, "player_uid"].iloc[0]
        st.session_state["selected_player_uid"] = selected_uid
        st.session_state["selected_role"] = board_type
        st.session_state["active_page"] = "Player Detail"
        st.rerun()


def _metric_card_row(player_row: pd.Series, role: str) -> None:
    metrics = [
        ("Production Rank", int(player_row["production_rank"])),
        ("Custom Rank", int(player_row["custom_rank"])),
        ("Draft Value", f"{player_row['draft_value_score']:.1f}"),
        ("Data Completeness", f"{player_row['data_completeness_score']:.1f}"),
    ]
    if role == "hitters":
        metrics.extend([("Reach", f"{player_row['reach_score']:.1f}"), ("Impact", f"{player_row['impact_score']:.1f}")])
    else:
        metrics.extend([("Reach", f"{player_row['reach_score']:.1f}"), ("Stuff", f"{player_row['stuff_score']:.1f}")])

    columns = st.columns(len(metrics))
    for column, (label, value) in zip(columns, metrics):
        column.metric(label, value)


def _render_notes_panel(player_uid: str, player_row: pd.Series, store: StateStore) -> None:
    current_note = player_row.get("note_text", "")
    current_role_fit = player_row.get("role_fit", "Unassigned")
    current_favorite = bool(player_row.get("is_favorite", False))
    with st.form(f"note_form_{player_uid}"):
        cols = st.columns([1, 1, 2])
        is_favorite = cols[0].checkbox("Favorite", value=current_favorite)
        role_fit = cols[1].selectbox("Role Fit", ROLE_FIT_OPTIONS, index=ROLE_FIT_OPTIONS.index(current_role_fit))
        note_text = cols[2].text_area("Analyst Notes", value=current_note, height=120)
        save = st.form_submit_button("Save Notes")
        if save:
            store.save_player_note(player_uid, is_favorite, role_fit, note_text)
            st.success("Saved analyst state.")
            st.rerun()


def _render_trends(player_uid: str, role: str, trends_df: pd.DataFrame) -> None:
    trend_rows = trends_df[(trends_df["player_uid"] == player_uid) & (trends_df["role"] == role)]
    st.subheader("Trend View")
    if trend_rows.empty:
        st.info("No trend history exported for this player yet.")
        return
    figure = px.line(trend_rows, x="season", y="metric_value", color="metric_label", markers=True)
    figure.update_layout(legend_title_text="")
    st.plotly_chart(figure, use_container_width=True)


def _render_hitter_detail(player_row: pd.Series, bundle: Dict[str, pd.DataFrame]) -> None:
    st.subheader("Score Summary")
    summary = pd.DataFrame(
        {
            "Component": ["Reach", "Impact", "Contact", "Risk"],
            "Score": [
                player_row["reach_score"],
                player_row["impact_score"],
                player_row["contact_score"],
                player_row["risk_score"],
            ],
        }
    )
    st.plotly_chart(px.bar(summary, x="Component", y="Score", range_y=[0, 100]), use_container_width=True)

    _render_trends(player_row["player_uid"], "hitters", bundle["player_trends"])

    st.subheader("EV / Launch Angle")
    bbe = bundle["hitter_bbe_detail"]
    bbe = bbe[bbe["player_uid"] == player_row["player_uid"]]
    if bbe.empty:
        st.info("No hitter BBE detail exported for this player.")
    else:
        st.plotly_chart(
            px.scatter(
                bbe,
                x="angle",
                y="exit_speed",
                color="direction",
                color_continuous_scale="Viridis",
                labels={"angle": "Launch Angle", "exit_speed": "Exit Speed", "direction": "Direction"},
            ),
            use_container_width=True,
        )

    st.subheader("Plate Discipline Panel")
    discipline = pd.DataFrame(
        {
            "Metric": ["Contact Rate", "Whiff Rate", "Chase Rate"],
            "Value": [player_row["contact_rate"], player_row["whiff_rate"], player_row["chase_rate"]],
        }
    )
    st.plotly_chart(px.bar(discipline, x="Metric", y="Value"), use_container_width=True)


def _render_pitcher_detail(player_row: pd.Series, bundle: Dict[str, pd.DataFrame]) -> None:
    st.subheader("Score Summary")
    summary = pd.DataFrame(
        {
            "Component": ["Reach", "Stuff", "Command", "Risk"],
            "Score": [
                player_row["reach_score"],
                player_row["stuff_score"],
                player_row["command_score"],
                player_row["risk_score"],
            ],
        }
    )
    st.plotly_chart(px.bar(summary, x="Component", y="Score", range_y=[0, 100]), use_container_width=True)

    _render_trends(player_row["player_uid"], "pitchers", bundle["player_trends"])

    st.subheader("Arsenal Table")
    pitch_types = bundle["pitcher_pitchtype_detail"]
    pitch_types = pitch_types[pitch_types["player_uid"] == player_row["player_uid"]]
    if pitch_types.empty:
        st.info("No pitch-type detail exported for this pitcher.")
    else:
        st.dataframe(pitch_types, use_container_width=True, hide_index=True)
        movement = px.scatter(
            pitch_types,
            x="avg_hb",
            y="avg_ivb",
            size="usage_pct",
            color="pitch_type",
            hover_data=["avg_velo", "csw_pct", "whiff_pct"],
        )
        movement.update_layout(xaxis_title="Average HB", yaxis_title="Average IVB")
        st.plotly_chart(movement, use_container_width=True)

    st.subheader("Command / Miss-Bat Panel")
    command = pd.DataFrame(
        {
            "Metric": ["CSW%", "Whiff%", "Zone%"],
            "Value": [player_row["csw_pct"], player_row["whiff_pct"], player_row["zone_pct"]],
        }
    )
    st.plotly_chart(px.bar(command, x="Metric", y="Value"), use_container_width=True)


def _render_benchmark_table(player_row: pd.Series, role: str, bundle: Dict[str, pd.DataFrame], board_df: pd.DataFrame) -> None:
    st.subheader("ACC / SEC Same-Season Benchmark")
    benchmark_df = bundle["benchmarks_acc_sec"]
    benchmark_df = benchmark_df[
        (benchmark_df["role"] == role)
        & (benchmark_df["season"] == player_row["season"])
        & (benchmark_df["benchmark_scope"] == "ACC_SEC")
    ]
    if benchmark_df.empty:
        st.info("No ACC/SEC benchmark export was available for this season and role.")
        return

    records = []
    season_board = board_df[board_df["season"] == player_row["season"]]
    for metric_key, metric_label in DETAIL_METRICS[role]:
        benchmark_row = benchmark_df[benchmark_df["metric_key"] == metric_key]
        if benchmark_row.empty or metric_key not in player_row.index:
            continue
        benchmark_value = float(benchmark_row["benchmark_value"].iloc[0])
        player_value = float(player_row[metric_key])
        percentile = season_board[metric_key].rank(pct=True).loc[player_row.name] * 100 if metric_key in season_board else float("nan")
        records.append(
            {
                "Metric": metric_label,
                "Player": round(player_value, 2),
                "ACC/SEC Avg": round(benchmark_value, 2),
                "Delta": round(player_value - benchmark_value, 2),
                "Percentile": round(percentile, 1),
            }
        )
    st.dataframe(pd.DataFrame(records), use_container_width=True, hide_index=True)


def _render_explanations(player_uid: str, role: str, bundle: Dict[str, pd.DataFrame]) -> None:
    st.subheader("Drivers and Reliability")
    explanation = bundle["explanations"]
    explanation = explanation[(explanation["player_uid"] == player_uid) & (explanation["role"] == role)]
    if explanation.empty:
        st.info("No explanation export found for this player.")
        return

    row = explanation.iloc[0]
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Top Positives**")
        st.write(row["positive_driver_1"])
        st.write(row["positive_driver_2"])
        st.markdown("**Top Negatives**")
        st.write(row["negative_driver_1"])
        st.write(row["negative_driver_2"])
    with col2:
        st.markdown("**Reliability Panel**")
        st.write(row["sample_size_text"])
        st.write(f"Data completeness: {row['data_completeness_score']:.1f}")
        st.write(f"Match confidence: {row['match_confidence']:.2f}")
        if row.get("warning_text"):
            st.warning(row["warning_text"])


def _render_player_detail(bundle: Dict[str, pd.DataFrame], store: StateStore) -> None:
    st.title("Player Detail")
    players = _build_player_index(bundle)
    if players.empty:
        st.info("No player exports available yet.")
        return

    selected_uid = st.session_state.get("selected_player_uid")
    if selected_uid not in players["player_uid"].tolist():
        selected_uid = players["player_uid"].iloc[0]
        st.session_state["selected_player_uid"] = selected_uid
        st.session_state["selected_role"] = players.loc[players["player_uid"] == selected_uid, "role"].iloc[0]

    labels = {
        row["player_uid"]: f"{row['player_name']} | {row['school_name']} | {row['role'][:-1].title()}"
        for _, row in players.iterrows()
    }
    selected_uid = st.sidebar.selectbox(
        "Player Search",
        options=list(labels.keys()),
        index=list(labels.keys()).index(selected_uid),
        format_func=lambda uid: labels[uid],
    )
    st.session_state["selected_player_uid"] = selected_uid

    role = players.loc[players["player_uid"] == selected_uid, "role"].iloc[0]
    st.session_state["selected_role"] = role
    board_df = merge_notes(bundle[f"{role}_board"], store.get_player_notes())
    weights = st.session_state[f"{role}_weights"]
    board_ranked = add_rank_columns(
        board_df,
        compute_custom_rank(
            board_df,
            positive_components={
                key: column
                for key, column in zip(
                    weights.keys(),
                    BOARD_DEFAULTS[role]["components"] + [BOARD_DEFAULTS[role]["risk_column"]],
                )
                if key != "risk"
            },
            risk_component=BOARD_DEFAULTS[role]["risk_column"],
            raw_weights=weights,
        ),
    )
    player_row = board_ranked[board_ranked["player_uid"] == selected_uid].copy()
    if player_row.empty:
        st.warning("Selected player was not present in the board export.")
        return
    row = player_row.iloc[0]

    st.subheader(f"{row['player_name']} | {row['school_name']} | {role[:-1].title()}")
    st.caption(f"Season {row['season']} export timestamp: {row['export_ts']}")
    _metric_card_row(row, role)
    _render_notes_panel(selected_uid, row, store)

    if role == "hitters":
        _render_hitter_detail(row, bundle)
    else:
        _render_pitcher_detail(row, bundle)

    _render_benchmark_table(row, role, bundle, board_df)
    _render_explanations(selected_uid, role, bundle)


def _render_diagnostics(bundle: Dict[str, pd.DataFrame]) -> None:
    st.title("Model Diagnostics")
    st.caption("Live board weights do not change trained outputs. They only re-rank precomputed normalized components.")
    diagnostics = bundle["diagnostics"]
    if diagnostics.empty:
        st.info("No diagnostics export was found yet.")
        return

    tabs = st.tabs(["Hitters", "Pitchers"])
    for tab, role in zip(tabs, ["hitters", "pitchers"]):
        with tab:
            role_df = diagnostics[diagnostics["role"] == role]
            if role_df.empty:
                st.info(f"No diagnostics exported for {role}.")
                continue

            metrics = role_df[role_df["record_type"] == "metric"]
            for _, row in metrics.iterrows():
                if pd.notna(row.get("value_text")):
                    st.info(f"{row['label']}: {row['value_text']}")

            averages = metrics[metrics["section"] == "component_average"]
            if not averages.empty:
                st.subheader("Component Averages")
                st.plotly_chart(px.bar(averages, x="label", y="value_num"), use_container_width=True)

            distributions = role_df[role_df["record_type"] == "distribution"]
            if not distributions.empty:
                st.subheader("Score Distribution")
                st.plotly_chart(px.bar(distributions, x="bucket_label", y="value_num", color="label"), use_container_width=True)

            examples = role_df[role_df["record_type"] == "example"]
            if not examples.empty:
                st.subheader("High-Score / High-Risk Profiles")
                st.dataframe(examples[["player_name", "value_num", "value_text"]], hide_index=True, use_container_width=True)


def _render_qa(bundle: Dict[str, pd.DataFrame]) -> None:
    st.title("Data QA")
    qa = bundle["qa"]
    if qa.empty:
        st.info("No QA export was found yet.")
        return
    metrics = qa[qa["record_type"] == "metric"]
    if not metrics.empty:
        st.subheader("Coverage and Freshness")
        st.dataframe(metrics[["section", "label", "value_num", "value_text", "role"]], hide_index=True, use_container_width=True)


def run_app() -> None:
    st.set_page_config(page_title="MLB Draft Dashboard", layout="wide")
    _ensure_session_defaults()

    exports_dir = Path(EXPORTS_DIR)
    missing = missing_export_files(exports_dir)
    if missing:
        _show_missing_exports(exports_dir)
        return

    bundle = cached_bundle(str(exports_dir))
    issues = _validate_bundle(bundle)

    st.sidebar.title("Draft Dashboard")
    st.sidebar.caption(f"Workspace: {BASE_DIR.name}")
    if issues:
        st.sidebar.error("Export validation issues detected.")
        with st.sidebar.expander("Validation issues", expanded=False):
            for issue in issues:
                st.write(issue)

    store = StateStore(STATE_DB_PATH)
    page = _select_page()

    if page == "Hitters Board":
        _render_saved_views("hitters", bundle["hitters_board"], store)
        ranked = _prepare_board("hitters", bundle, store)
        _display_board("hitters", ranked)
    elif page == "Pitchers Board":
        _render_saved_views("pitchers", bundle["pitchers_board"], store)
        ranked = _prepare_board("pitchers", bundle, store)
        _display_board("pitchers", ranked)
    elif page == "Player Detail":
        _render_player_detail(bundle, store)
    elif page == "Model Diagnostics":
        _render_diagnostics(bundle)
    else:
        _render_qa(bundle)
