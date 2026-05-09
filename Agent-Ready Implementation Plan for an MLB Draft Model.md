# Agent-Ready Implementation Plan for an MLB Draft Model

## Overview

This document is an implementation-grade plan for building a draft model that uses college Trackman data from 2024–2026 and augments it with public MLB Statcast, FanGraphs, and NCAA data. It is written so a coding agent can take it and begin scaffolding the project immediately. The system is designed around a Python-first stack, with R used only where `baseballr` provides better access to NCAA and Chadwick player ID data. DuckDB queries Parquet files directly, which eliminates the need for a running database server and makes the project easy to keep local and reproducible.[^1][^2][^3][^4]

The plan assumes your private data is a set of Trackman CSV exports. Trackman's V3 glossary documents that these CSV exports contain identifiers, player names, handedness, pitch and hit tags, count context, and the physical measurements needed to build pitch and contact models, including columns such as `PitcherId`, `BatterId`, `PitcherThrows`, `BatterSide`, `RelSpeed`, `SpinRate`, and the default export ordering. Public Trackman examples in college datasets also confirm that pitch-level files usually include manually tagged pitch and hit types and enough context to reconstruct pitch- and player-level summaries.[^5][^6][^7][^8]

The final system should do six things well: ingest raw files, standardize schema, build player/entity crosswalks, create normalized features, train and validate a two-stage model, and serve the outputs in an interactive dashboard. The draft model is not just a single notebook; it is a reproducible data product with versioned inputs, deterministic outputs, and a clear separation between raw data, derived features, trained models, and presentation.

## Build Objectives

The engineering objectives should be explicit before any code is written.

- Build a reproducible pipeline from raw Trackman CSVs to model-ready player-season tables.
- Resolve player identity across Trackman, NCAA, FanGraphs, and MLBAM/Statcast sources using the Chadwick register and deterministic/fuzzy matching layers.[^2][^1]
- Produce separate modeling tracks for pitchers and hitters because the features, targets, and translation adjustments differ materially.
- Keep all intermediate datasets in Parquet and use DuckDB as the analytical engine so the pipeline remains local, fast, and easy to inspect.[^3][^4]
- Use temporal or group-aware validation such as `LeaveOneGroupOut` or grouped CV where the group is draft class or season to avoid leakage across years.[^9][^10]
- Track experiments and model versions with MLflow so every result can be reproduced from a run ID and commit hash.[^11][^12]

## Recommended Repository Layout

The coding agent should create the project as a proper Python package rather than a loose notebook folder. `uv` is a strong package/project manager here because it writes and maintains `pyproject.toml`, creates a lockfile, and supports reproducible installs with `uv sync` and `uv add` workflows.[^13][^14][^15]

```text
mlb_draft_model/
├── pyproject.toml
├── uv.lock
├── README.md
├── .gitignore
├── .python-version
├── configs/
│   ├── paths.yaml
│   ├── schema.yaml
│   ├── features.yaml
│   ├── model_pitchers.yaml
│   ├── model_hitters.yaml
│   └── logging.yaml
├── data/
│   ├── raw/
│   │   ├── trackman/2024/
│   │   ├── trackman/2025/
│   │   ├── trackman/2026/
│   │   ├── ncaa/
│   │   └── statcast/
│   ├── interim/
│   ├── external/
│   ├── features/
│   └── marts/
├── models/
│   ├── stage1/
│   ├── stage2/
│   └── explainability/
├── notebooks/
├── reports/
├── app/
│   ├── streamlit_app.py
│   ├── pages/
│   └── components/
├── scripts/
│   ├── 00_fetch_external_data.py
│   ├── 01_ingest_trackman.py
│   ├── 02_standardize_schema.py
│   ├── 03_build_player_crosswalk.py
│   ├── 04_build_game_context.py
│   ├── 05_build_pitch_features.py
│   ├── 06_build_batted_ball_features.py
│   ├── 07_build_player_season_tables.py
│   ├── 08_build_training_labels.py
│   ├── 09_train_stage1.py
│   ├── 10_train_stage2.py
│   ├── 11_score_current_class.py
│   └── 12_generate_dashboard_tables.py
├── src/
│   └── draft_model/
│       ├── __init__.py
│       ├── io/
│       ├── schema/
│       ├── ids/
│       ├── features/
│       ├── labels/
│       ├── modeling/
│       ├── evaluation/
│       ├── plotting/
│       ├── app/
│       └── utils/
└── tests/
    ├── test_schema.py
    ├── test_crosswalks.py
    ├── test_features.py
    ├── test_labels.py
    └── test_modeling.py
```

This structure matters because the project has both batch pipelines and interactive exploration. Batch logic lives in `scripts/` and reusable logic lives in `src/`. The dashboard only consumes frozen feature and scoring tables from `data/marts/` so the web app stays fast and does not recompute the pipeline on every page load.

## Environment and Dependency Specification

The coding agent should initialize the project with `uv init`, define a `pyproject.toml`, and keep the lockfile under version control because `uv.lock` captures the exact resolved environment for reproducibility. The Python version should be pinned in `project.requires-python`; DuckDB's current Python client requires Python 3.9 or newer.[^14][^16][^15][^3][^13]

Recommended runtime dependencies:

- `duckdb`
- `polars`
- `pandas`
- `pyarrow`
- `numpy`
- `scikit-learn`
- `xgboost`
- `lightgbm`
- `optuna`
- `mlflow`
- `shap`
- `plotly`
- `streamlit`
- `orjson`
- `rapidfuzz`
- `pyyaml`
- `pydantic`
- `joblib`

Recommended dev dependencies:

- `pytest`
- `ruff`
- `mypy`
- `ipykernel`

The coding agent should also create an optional R helper script directory for `baseballr`. It does not need to R-package the project. A lightweight `R/` folder containing fetch scripts is enough.

## Initial Setup Commands

The coding agent should start with a deterministic setup flow.

```bash
uv init
uv add duckdb polars pandas pyarrow numpy scikit-learn xgboost lightgbm optuna mlflow shap plotly streamlit orjson rapidfuzz pyyaml pydantic joblib
uv add --dev pytest ruff mypy ipykernel
uv sync
```

Then create a minimal `pyproject.toml` with project metadata, a pinned Python version, and tool sections for `ruff` and `mypy`.[^17][^14]

## Data Sources and Their Roles

The coding agent should explicitly separate each source by role.

| Source | Role in project | Access path |
|---|---|---|
| Private college Trackman CSVs | Primary pitch and batted-ball measurements | Your exports[^7] |
| Chadwick player register | Crosswalk names to baseball IDs | `baseballr::chadwick_player_lu()`[^2][^1] |
| NCAA play-by-play/schedule | Game context and roster validation | `load_ncaa_baseball_pbp()` and related functions[^18] |
| MLB Statcast | Downstream pro outcomes, comps, pitch/contact context | `pybaseball.statcast*`[^19][^20] |
| FanGraphs season data | Aggregated public stats and historical labels/controls | `pybaseball` FanGraphs loaders[^21][^22] |
| College park/conference context | Opponent and environment normalization | external tables or custom build[^23][^24] |

The project should treat Trackman as the system of record for pitch and batted-ball physics. NCAA and FanGraphs supply context; Statcast supplies downstream outcome and comp information.

## Pipeline Orchestration Philosophy

The project does not need Airflow, Prefect, or Dagster for version 1. A simple script-based DAG is enough if each script is idempotent, writes deterministic outputs, and checks whether the target artifact already exists. The coding agent should implement a lightweight CLI pattern:

```bash
python scripts/01_ingest_trackman.py --season 2024
python scripts/02_standardize_schema.py --season 2024
python scripts/03_build_player_crosswalk.py
python scripts/09_train_stage1.py --group pitchers
```

Each script should:
- read from one or more known input files,
- validate schema,
- write a single well-named Parquet artifact,
- emit logs,
- fail loudly if required inputs are missing,
- never overwrite raw data.

## Step 1: Ingest Raw Trackman Files

Trackman CSV exports are the highest-risk ingestion step because private exports often vary slightly by venue, season, or export configuration even when they are nominally V3 files. Trackman's glossary provides the canonical export naming and order, including columns for pitch number, timestamps, pitcher and batter IDs, team abbreviations, handedness, and downstream measurements. The coding agent should therefore build ingestion as a schema-tolerant but validated process rather than hard-coding a single exact column list.[^7]

### Implementation requirements

1. Scan `data/raw/trackman/{season}/**/*.csv`.
2. Read each file with Polars lazy scans where possible.
3. Normalize column names to snake_case.
4. Build a column alias map so synonymous exports map to one canonical schema.
5. Add file-level metadata columns: `source_file`, `season`, `ingested_at`, `row_hash`.
6. Write a single season-level Parquet file to `data/interim/trackman_raw_{season}.parquet`.

### Canonical schema design

The coding agent should define a YAML schema file with three classes of columns.

**Identity columns**
- game_date
- pitcher_name
- pitcher_id_raw
- batter_name
- batter_id_raw
- pitcher_team
- batter_team
- season
- game_id_raw

**Context columns**
- inning
- top_bottom
- balls
- strikes
- outs
- pa_of_inning
- pitch_of_pa
- stand
- throws
- tagged_pitch_type
- auto_pitch_type
- tagged_hit_type
- play_result

**Physics columns**
- rel_speed
- spin_rate
- spin_axis
- rel_height
- rel_side
- extension
- plate_x
- plate_z
- ivb
- hb
- vaa
- haa
- exit_speed
- launch_angle
- bearing
- distance

If `vaa` is not present in your private export, the schema should still allow nulls and later derive a standardized VAA-like measure if the necessary movement and release parameters are available.

### Suggested ingest function

```python
import polars as pl
from pathlib import Path

CANONICAL_MAP = {
    "PitcherId": "pitcher_id_raw",
    "BatterId": "batter_id_raw",
    "Pitcher": "pitcher_name",
    "Batter": "batter_name",
    "PitcherThrows": "throws",
    "BatterSide": "stand",
    "RelSpeed": "rel_speed",
    "SpinRate": "spin_rate",
    "RelHeight": "rel_height",
    "Extension": "extension",
}

def normalize_columns(cols: list[str]) -> dict[str, str]:
    out = {}
    for c in cols:
        out[c] = CANONICAL_MAP.get(c, c.strip().lower())
    return out

files = list(Path("data/raw/trackman/2025").rglob("*.csv"))
frames = []
for fp in files:
    df = pl.read_csv(fp, infer_schema_length=5000, ignore_errors=False)
    rename_map = normalize_columns(df.columns)
    df = df.rename(rename_map).with_columns([
        pl.lit(str(fp)).alias("source_file"),
        pl.lit(2025).alias("season")
    ])
    frames.append(df)
all_df = pl.concat(frames, how="diagonal_relaxed")
all_df.write_parquet("data/interim/trackman_raw_2025.parquet")
```

The agent should not assume every file contains every field. `diagonal_relaxed` concatenation is appropriate when some exports have slightly different column sets.

## Step 2: Standardize and Validate Schema

After raw ingestion, the coding agent should build a schema standardization pass. The point is to convert all Trackman seasons into a stable table contract that downstream scripts can trust.

### Tasks

- Cast dates and timestamps.
- Standardize handedness values to `R` / `L` / `S`.
- Standardize pitch types into a project-specific taxonomy.
- Standardize team names and conference labels.
- Convert imperial/metric if any venue exports differ.
- Drop duplicate rows using stable keys.

The dedupe key should be a hashed composite of at least: `season`, `game_date`, `pitcher_name`, `batter_name`, `inning`, `top_bottom`, `pa_of_inning`, `pitch_of_pa`, and `pitch_no`. The coding agent should produce a validation report counting missingness and field ranges. For example, `rel_speed` should never be negative, and `spin_rate` should stay within plausible baseball ranges.

### Validation outputs

Write a validation summary table per season with:
- row count
- unique pitchers
- unique batters
- null rate by critical column
- min/max of core physics columns
- count of duplicate dedupe keys

This should be written as `data/interim/trackman_validation_{season}.parquet` and optionally a human-readable CSV.

## Step 3: Build External Data Fetchers

### Chadwick crosswalk

The coding agent should create a small R script that runs `chadwick_player_lu()` and writes the output to `data/external/chadwick_players.parquet`. `baseballr` documents this function explicitly as the public Chadwick register download.[^25][^1][^2]

### NCAA play-by-play and schedule

The coding agent should also use `load_ncaa_baseball_pbp()` for relevant seasons, because the function can load multiple seasons and optionally write to a database. The pipeline only needs the returned tibble; it can be written straight to Parquet.[^18]

### Statcast pulls

`pybaseball` supports `statcast()`, `statcast_pitcher()`, and `statcast_batter()`. The player-specific functions use `start_dt`, `end_dt`, and `player_id`, where `player_id` is the MLBAM ID. The agent should pull public Statcast only for matched players or for a comp pool; it should not pull all MLB history blindly.[^19][^26][^20]

### FanGraphs pulls

`pybaseball` season-level FanGraphs functions such as `batting_stats` and `pitching_stats` allow year ranges, qualification filters, and individual or aggregate views. The agent should fetch historical MLB/Minor-proxy or major-league seasons to support comp generation and baseline distributions.[^21][^22]

## Step 4: Build Player Crosswalks

This is the most important non-modeling component in the project. If player ID resolution is weak, your labels, comps, and historical joins will all degrade.

### Crosswalk strategy

The coding agent should build a three-stage resolver:

**Stage A: Exact deterministic match**
- Use `pitcher_id_raw` and `batter_id_raw` when Trackman IDs correspond to known external IDs.
- Match exact `last, first` strings to Chadwick where available.
- Match by exact full name + season + school/team.

**Stage B: Normalized string match**
- Normalize diacritics, punctuation, and suffixes.
- Convert `Last, First` to `First Last` and vice versa.
- Use RapidFuzz string similarity with school/team and class-year constraints.

**Stage C: Manual review queue**
- Any match below a confidence threshold should be written to `data/interim/player_match_review.parquet`.
- Include candidate scores and evidence columns.

The coding agent should also create a `player_master.parquet` table keyed by an internal `player_uid`. That `player_uid` is the stable project ID and should map to all known identifiers:
- trackman_pitcher_id_raw
- trackman_batter_id_raw
- mlbam_id
- fangraphs_id
- bbref_id
- ncaa_name_variant
- school
- season_first_seen
- season_last_seen

The crosswalk should be versioned because future manual corrections should never be lost.

## Step 5: Build Game and Competition Context

Modeling raw Trackman physics is useful, but a draft model also needs context. The coding agent should construct a game-context table keyed by game and team.

### Inputs
- NCAA schedule / play-by-play[^18]
- Team metadata
- Conference mapping
- Park metadata

### Outputs
Per game and per team, compute:
- opponent team
- home/away/neutral
- conference
- conference tier
- opponent strength proxy
- park ID
- park factor placeholders
- date and season

This context should be joined into pitch- and batted-ball records early so feature engineering can aggregate by situation if needed.

## Step 6: Create Park and Schedule Adjustment Tables

Public research on Division I park factors shows college environments are much more extreme than MLB environments, and FanGraphs now exposes conference-adjusted college metrics via the College Splits integration. The coding agent should therefore separate physics features from environment-sensitive outcome features.[^23][^24]

### Rule set

- Do **not** park-adjust raw pitch physics such as rel_speed, spin_rate, IVB, HB, extension.
- Do **not** park-adjust raw batted-ball physics such as exit speed and launch angle.
- **Do** adjust rate stats derived from outcomes, such as OPS, ISO, HR rate, ERA, and strikeout rates if used as contextual features.

### Implementation options

1. If you have external park factor tables, store them in `data/external/park_factors.parquet` and join by team/venue/season.
2. If not, create a v1 custom proxy using your own Trackman data by estimating venue effects on outcome metrics after controlling for batter/pitcher strength.

The coding agent should treat this as modular: the system should work even if v1 uses a coarse conference-tier adjustment instead of full park factors.

## Step 7: Engineer Pitch-Level Features

Pitcher modeling should start at the pitch level and roll upward. The coding agent should write a feature builder that groups by pitcher, season, and pitch type.

### Required pitch-level derived fields

- count bucket: ahead / even / behind
- zone bucket
- hard contact allowed flag
- swing flag
- whiff flag
- called_strike flag
- csw flag
- in_play flag
- barrel_allowed proxy if EV/LA thresholds are met
- velocity_zscore within season and pitch type
- movement differential from pitch-type mean

### Aggregated pitcher-season features

By pitcher, season, and optionally pitch type:
- average and max rel_speed
- average spin_rate
- average IVB and HB
- average release height and extension
- release consistency (sd of release height and side)
- usage share by pitch type
- CSW% by pitch type
- whiff% by pitch type
- chase% proxy if zone logic is available
- hard-hit-allowed% on contact
- arsenal count
- movement separation score across pitch types

The coding agent should also build multi-year trend features for pitchers with multiple seasons:
- velo_delta_24_25
- velo_delta_25_26
- spin_delta
- IVB_delta
- BB-like control deltas

### Stuff+ style internal model

The project should include an internal college Stuff+ model. The coding agent should train a pitch-level model where the response is a pitch-value proxy such as `delta_run_exp`, `swinging_strike`, or a composite outcome score if direct run expectancy is unavailable. Public Statcast columns include `delta_run_exp` and `delta_home_win_exp` in the pitch-level output, demonstrating the feasibility of using run-value style responses in baseball pitch models.[^27]

For the college data, if direct run value is unavailable in Trackman, the coding agent should create a hierarchical target such as:
- swing-and-miss = +2
- called strike = +1
- foul = 0
- ball = -1
- hard-hit in play = -2

This should be configurable, not hard-coded in business logic. The resulting predicted value per pitch becomes `stuff_college_score`. Aggregate that score by pitch type and overall pitcher season.

## Step 8: Engineer Hitter and Batted-Ball Features

Hitters need a separate feature pipeline because the bat context matters more. College wood/aluminum translation is a known issue and prior public work reports a drop of roughly 2.5–3.2 mph in EV when moving from aluminum to wood. The coding agent should therefore compute both raw and wood-adjusted EV features and keep them side by side.[^28]

### Required hitter-season features

- avg_exit_speed_raw
- avg_exit_speed_wood_adj
- p90_exit_speed_raw
- p90_exit_speed_wood_adj
- avg_launch_angle
- launch_angle_stdev
- hard_hit_rate_raw
- barrel_rate_proxy_raw
- barrel_rate_proxy_wood_adj
- pull/oppo/center directional shares if direction exists
- whiff_rate
- contact_rate
- chase_rate proxy
- bb_rate and k_rate if count/results are present
- xwoba_proxy from EV + LA bins

### Suggested batted-ball feature logic

The coding agent should define project-level thresholds in `features.yaml` so they can be changed later without code edits.

Example:
```yaml
hitting:
  wood_ev_discount_mph: 2.8
  hard_hit_ev_threshold: 95
  barrel:
    min_ev: 98
    min_la: 26
    max_la: 30
```

Then derive:
- `exit_speed_wood_adj = max(exit_speed - wood_ev_discount_mph, 0)`
- `hard_hit_flag = 1 if exit_speed >= threshold`
- `barrel_flag_proxy = 1 if exit_speed and launch_angle fall in configured zone`

The coding agent should preserve both raw and adjusted versions because draft rooms often want to see both.

## Step 9: Assemble Player-Season Tables

After pitch and batted-ball aggregation, the agent should create clean player-season marts.

### Pitchers table

Key: `player_uid`, `season`

Columns:
- player identity fields
- school, conference, throws, height/weight if available
- physical pitch features
- internal stuff features
- contextual stats
- multi-year trend features
- eligibility metadata

### Hitters table

Key: `player_uid`, `season`

Columns:
- player identity fields
- school, conference, stand, primary position if available
- contact quality features
- swing/discipline features
- contextual stats
- multi-year trend features
- eligibility metadata

These should be written to `data/marts/pitchers_player_season.parquet` and `data/marts/hitters_player_season.parquet`.

## Step 10: Build Training Labels

The label design should be modular because you may want to experiment with several outcomes. Public work on scouting report models and draft projection emphasizes that MLB reach rate is heavily imbalanced and many draftees never reach the majors. The coding agent should therefore support multiple label families.[^29]

### Stage 1 labels: survival / reach model

Candidate binary labels:
- reached affiliated ball
- reached High-A within N years
- reached Double-A within N years
- reached MLB within N years

Recommended v1 label:
- `reached_double_a_within_4y`

That is a better compromise than immediate MLB reach because it reduces noise from late bloomers while still measuring organizationally meaningful success.

### Stage 2 labels: performance conditional on survival

Candidate regression labels:
- max level reached ordinal score
- peak wRC+ or FIP- at highest level reached
- WAR through age-26
- draft slot surplus relative to consensus

Recommended v1 targets:
- hitters: best park-adjusted wRC+ at AA/AAA/MLB within 4 years
- pitchers: best FIP- or K-BB% proxy at AA/AAA/MLB within 4 years

### Label data sources

- public Statcast for MLB-reached players[^20][^19]
- FanGraphs season stats for reached players[^22][^21]
- Baseball Reference / MiLB proxies if you add another public source later

The coding agent should version label tables separately from features because label logic will evolve.

## Step 11: Design the Modeling Workflow

The modeling system should be built separately for hitters and pitchers, and each should have two stages.

### Stage 1: binary classifier

Model objective:
- probability player reaches threshold outcome

Recommended model:
- `XGBClassifier`

Why this is appropriate:
- handles tabular data well,
- deals with missingness gracefully,
- supports non-linear interactions,
- works well with SHAP explainability,
- supports class imbalance through `scale_pos_weight`.[^30][^31][^32]

### Stage 2: conditional regressor

Model objective:
- expected conditional performance among players likely to reach

Recommended models:
- `XGBRegressor` or `LGBMRegressor`

### Baseline models

The coding agent should implement baselines first:
- logistic regression for Stage 1,
- ridge regression for Stage 2,
- maybe a Marcel-like weighted-average baseline for sanity checking.

This is important because baseball often rewards simple baselines, and public work has shown that simple projection frameworks can outperform more complex ML in small-data settings.[^33]

## Step 12: Validation Design

The project must avoid leakage. `LeaveOneGroupOut` in scikit-learn is explicitly appropriate when groups represent collection years and you want time-aware splits. Group-aware CV examples also show how to pass groups through nested CV or cross-validation pipelines.[^34][^10][^9]

### Recommended validation groups

Use one of these group definitions:
- draft class year,
- season,
- player_uid for repeated player records.

Recommended v1 strategy:
- Outer CV: `LeaveOneGroupOut` by draft class or season.
- Inner tuning split: grouped fold on remaining groups.

### Metrics

Stage 1:
- ROC AUC
- PR AUC
- Brier score
- calibration curve

Stage 2:
- RMSE
- MAE
- Spearman rank correlation
- top-N hit rate (how many eventual strong outcomes appear in top model bucket)

The coding agent should always log both rank-based and error-based metrics because draft applications care more about ordering than exact point estimates.

## Step 13: Feature Preprocessing Pipeline

Because XGBoost and LightGBM can handle unscaled numeric inputs, the preprocessing pipeline should stay simple.

### Numeric features
- median imputation or explicit missing indicators
- no standard scaling required for tree models

### Categorical features
- small-cardinality features: one-hot encode with scikit-learn `OneHotEncoder`
- large-cardinality identifiers: do not include raw names or IDs
- handedness and conference tier are safe categorical inputs

### Leakage exclusions
Never include:
- actual draft position if predicting draft value,
- post-draft pro outcomes in training features,
- future seasons when scoring current class,
- any feature derived from manual scouting grades unless the project explicitly intends to blend scouting and data.

## Step 14: Experiment Tracking with MLflow

MLflow's XGBoost integration supports autologging of parameters, metrics, feature importance, models, and artifacts with minimal instrumentation. The coding agent should set up separate experiments for hitters and pitchers, and separate runs for stage 1 and stage 2.[^12][^11]

### Experiment names
- `draft_pitchers_stage1`
- `draft_pitchers_stage2`
- `draft_hitters_stage1`
- `draft_hitters_stage2`

### Required logged artifacts
- training config yaml snapshot
- train/validation metrics JSON
- feature list used
- SHAP summary PNG
- calibration plot PNG for classifier
- model binary
- predictions on validation fold

### Example training pattern

```python
import mlflow
import mlflow.xgboost
from xgboost import XGBClassifier

mlflow.set_experiment("draft_pitchers_stage1")
mlflow.xgboost.autolog()

with mlflow.start_run():
    model = XGBClassifier(
        n_estimators=500,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=pos_neg_ratio,
        random_state=42,
        eval_metric="auc"
    )
    model.fit(X_train, y_train)
```

The coding agent should also save a separate lightweight `manifest.json` in each model folder so the dashboard can load the current production version without reading MLflow internals.

## Step 15: Hyperparameter Search with Optuna

Optuna is well-suited to XGBoost tuning and its define-by-run interface is documented with XGBoost examples. The agent should implement Optuna after the baseline model is stable, not before.[^35][^36][^37]

### Tuning plan

Tune only a compact parameter set first:
- `max_depth`
- `learning_rate`
- `min_child_weight`
- `subsample`
- `colsample_bytree`
- `reg_alpha`
- `reg_lambda`
- `n_estimators`

Use grouped CV within the objective and optimize PR AUC for Stage 1, Spearman correlation or MAE for Stage 2.

The coding agent should cap early searches at 30–50 trials. Baseball datasets of this kind are usually not large enough to justify massive HPO until the label table is stable.

## Step 16: Explainability with SHAP

SHAP's documentation shows straightforward use with XGBoost, including waterfall plots for single predictions and beeswarm plots for global summaries. This is especially important in a scouting-facing product because users need to understand why a player scores well.[^38][^39][^40]

### Required outputs

Global:
- SHAP beeswarm plot for model-wide drivers
- mean absolute SHAP importance table

Per player:
- SHAP waterfall plot for the selected prospect
- top positive and negative driver table

The coding agent should precompute SHAP values for the current draft class and store them as Parquet or JSON so the dashboard can render instantly. Generating SHAP on the fly for every page interaction is unnecessary and can be slow.

## Step 17: Scoring Current Prospects

Once models are trained, the scoring script should read the most recent `player_season` marts, apply the saved preprocessing and model objects, and write a final board table.

### Recommended scoring outputs per player
- `reach_score` (0 to 1)
- `performance_score_conditional`
- `draft_value_score` (blended)
- `risk_score`
- `nearest_comp_1`
- `nearest_comp_2`
- `nearest_comp_3`
- `top_positive_driver_1`
- `top_negative_driver_1`

### Blend logic

The coding agent should keep blend logic configurable. Example:

```yaml
scoring:
  draft_value_score:
    reach_weight: 0.60
    performance_weight: 0.40
```

That way you can tune the blend later without retraining both stages.

## Step 18: Comparable Player Engine

The project should not rely solely on model scores. A nearest-neighbor comp engine adds interpretation and practical baseball value. Public prospect analysis has used pitch-shape comparables such as matching college fastball shape profiles to MLB pitchers.[^41]

### Implementation

1. Build a comp feature space separately for pitchers and hitters.
2. Standardize numeric features within role.
3. Use cosine similarity or Euclidean distance.
4. Search over a comp pool of public MLB/Statcast players or historical college-to-pro players.
5. Return nearest matches with distance score.

### Pitcher comp features
- avg fastball rel_speed
- avg fastball IVB
- avg fastball HB
- extension
- release height
- primary breaking ball shape
- arsenal mix

### Hitter comp features
- p90 EV
- avg EV
- launch angle distribution stats
- pull/oppo tendency
- contact/whiff profile

Store the comp table in a separate mart so the dashboard can fetch it directly.

## Step 19: Dashboard Design in Streamlit

The dashboard should consume already-built marts and should cache data-loading functions, not figure objects. Streamlit community discussions and issues show that caching data is stable, while caching Plotly figures themselves can be problematic in some versions.[^42][^43][^44]

### Page structure

**Home / Board**
- sortable table of all draft-eligible players
- filters by season, conference, position, handedness, score range

**Player Card**
- biographical header
- score summary
- movement plot or EV/LA chart
- year-over-year trend lines
- SHAP waterfall
- comps

**Model Diagnostics**
- ROC/PR curves
- calibration plot
- feature importance
- residuals/ranking analysis

**Data QA**
- file coverage
- missingness counts
- unmatched player list

### Streamlit data loading pattern

```python
import streamlit as st
import duckdb

@st.cache_data(ttl="1h")
def load_board():
    return duckdb.sql("SELECT * FROM 'data/marts/draft_board.parquet'").df()
```

The agent should cache dataframes, not `st.plotly_chart` outputs. Build figures on top of cached dataframes at render time.

## Step 20: DuckDB Query Layer

DuckDB's Python API can query Parquet directly using `duckdb.sql("SELECT * FROM 'file.parquet'")` or `read_parquet`, and the docs show both patterns. This should be the default read pattern across scripts and the app.[^4][^3]

### Recommended patterns

- Query one Parquet file directly for marts.
- Use wildcard or explicit file lists for multi-file scans.[^45][^46]
- Prefer SQL for aggregations and joins, then convert to Pandas only when modeling or plotting.

Example:

```python
import duckdb

df = duckdb.sql("""
SELECT
    player_uid,
    season,
    avg_rel_speed,
    avg_ivb,
    avg_hb,
    stuff_college_score
FROM 'data/marts/pitchers_player_season.parquet'
WHERE season IN (2024, 2025, 2026)
""").df()
```

## Step 21: Testing Strategy

The coding agent should not skip tests. This project is especially vulnerable to silent failures in schema mapping and ID resolution.

### Minimum test suite

`test_schema.py`
- verifies required canonical columns exist after standardization
- checks date and numeric casts

`test_crosswalks.py`
- checks deterministic match logic
- checks fuzzy threshold behavior
- asserts no duplicated `player_uid`

`test_features.py`
- validates expected ranges for rel_speed, spin_rate, EV
- validates that multi-year aggregations do not duplicate rows
- checks wood-adjusted EV is never greater than raw EV

`test_labels.py`
- ensures no label leakage from future seasons
- checks that stage 2 labels exist only for qualified stage 1 survivors if using conditional modeling

`test_modeling.py`
- fits a tiny smoke-test model on toy data
- confirms pipeline serializes and deserializes

## Step 22: Logging and Error Handling

The coding agent should use structured logging throughout. Each script should log:
- inputs found
- rows read
- rows dropped
- rows written
- wall-clock time
- warnings for unmatched players or missing critical columns

Any unresolved schema mismatch should raise an exception before writing the artifact. The project should not silently continue if, for example, a season file lacks `rel_speed` or `pitcher_name`.

## Step 23: Configuration Over Hard-Coding

The coding agent should expose all assumptions through config files rather than embedding them in scripts.

### `features.yaml`
- wood EV discount
- barrel thresholds
- minimum pitches for pitcher inclusion
- minimum BBE for hitter inclusion
- recency weights for multi-year aggregation

### `model_pitchers.yaml` and `model_hitters.yaml`
- selected features
- target definitions
- validation group column
- XGBoost parameter bounds
- score blend weights

### `schema.yaml`
- canonical columns
- alias map
- required/non-required fields
- numeric ranges for validation

This makes the project easier to hand off and safer to iterate.

## Step 24: Recommended Build Order for the Coding Agent

The coding agent should work in this exact order:

1. Initialize repo, environment, folder structure.
2. Write schema config and ingestion utilities.
3. Ingest one season of Trackman to validate assumptions.
4. Build schema standardization and validation report.
5. Fetch Chadwick and NCAA public data.
6. Build player crosswalk and review queue.
7. Create pitch and batted-ball feature builders.
8. Build player-season marts.
9. Fetch Statcast/FanGraphs outcomes for matched historical players.
10. Build stage 1 labels.
11. Fit a baseline classifier with grouped CV.
12. Add MLflow logging.
13. Add Optuna tuning.
14. Build SHAP outputs.
15. Build stage 2 model.
16. Score the current class.
17. Build the Streamlit dashboard on frozen marts.
18. Add tests and polish.

This sequencing matters because the biggest risks are in data quality, not model choice. The pipeline should prove the data layer first.

## Step 25: Deliverables the Coding Agent Should Produce

By the end of v1, the coding agent should hand back:

- a working repo with `pyproject.toml` and lockfile[^13][^14]
- one command per pipeline step in `scripts/`
- season-level raw and standardized Trackman Parquet files
- player crosswalk table and manual review table
- hitter and pitcher player-season marts
- label tables for stage 1 and stage 2
- trained stage 1 and stage 2 model artifacts
- MLflow experiment directory
- SHAP global and player-level artifacts
- final draft board Parquet/CSV
- Streamlit dashboard that reads the final marts and model outputs
- README with run instructions and dependency notes

## Concrete Agent Prompt You Can Hand Off

Use the following as the initial instruction block for a coding agent:

```text
Build a Python-first MLB draft modeling project using the repository layout and pipeline described below.

Requirements:
1. Use uv with pyproject.toml and lockfile.
2. Store all intermediate and final analytical data as Parquet.
3. Use DuckDB for analytical queries and marts.
4. Use Polars for raw CSV ingestion and ETL.
5. Create canonical Trackman schema mapping from potentially inconsistent CSV exports.
6. Add validation reports for every season ingested.
7. Build a player crosswalk using Chadwick data from baseballr and a fuzzy matching review queue.
8. Create separate player-season marts for pitchers and hitters.
9. Implement stage 1 XGBoost classification and stage 2 regression with grouped temporal validation.
10. Track experiments with MLflow.
11. Precompute SHAP summary and player-level explanation outputs.
12. Build a Streamlit dashboard that loads frozen marts and model outputs.
13. Add pytest smoke tests for schema, crosswalks, features, and model serialization.

Implementation constraints:
- Never overwrite raw data.
- Use config files for thresholds and feature selection.
- Keep scripts idempotent.
- Use clear logging and fail loudly on schema mismatches.
- Start with one season to validate assumptions before scaling to all years.
```

## Final Recommendation

The strongest implementation path is to treat this as a data engineering project first, a modeling project second, and a dashboard project third. The public APIs and docs are mature enough to support this architecture: `baseballr` gives you Chadwick and NCAA access, `pybaseball` gives you player-level Statcast with known arguments and output shapes, DuckDB gives you direct Parquet querying without infrastructure, and MLflow plus SHAP give you a clean experimentation and explainability layer on top of XGBoost.[^39][^19][^2][^27][^3][^20][^4][^11][^38][^18]

If you want the cleanest handoff to a coding agent, the first thing to request after this plan is: **"Scaffold the repo, pyproject.toml, configs, and the first three scripts: ingest Trackman, standardize schema, and build the player crosswalk."** That narrows the initial scope to the riskiest parts of the project and forces the agent to solve the right problems first.

---

## References

1. [baseballr: Acquiring and Analyzing Baseball Data](https://est.colpos.mx/web/packages/baseballr/baseballr.pdf)

2. [NCAA Baseball](https://billpetti.github.io/baseballr/reference/index.html)

3. [Python API - DuckDB](https://duckdb.org/docs/lts/clients/python/overview) - The latest stable version of the DuckDB Python client is {{ site.current_duckdb_version }}. Installa...

4. [Reading and Writing Parquet Files - DuckDB](https://duckdb.org/docs/current/data/parquet/overview.html) - Examples Read a single Parquet file: SELECT * FROM 'test.parquet'; Figure out which columns/types ar...

5. [Optical Tracking Data from College Baseball Scrimmages](https://data.mendeley.com/datasets/xfnz6mkdzm/1) - The Track_Combo.csv (Track_Combo.txt) dataset contains ball-tracking data from a series of exhibitio...

6. [Dataset Comparison - Mendeley Data](https://data.mendeley.com/datasets/compare/xfnz6mkdzm)

7. [V3 | FAQs | Radar Measurement Glossary Of Terms](https://support.trackmanbaseball.com/hc/en-us/articles/5089413493787-V3-FAQs-Radar-Measurement-Glossary-Of-Terms) - Glossary of Terms Terms are listed by the default order in which they appear upon exporting a .CSV f...

8. [Radar | Glossary Of Terms - Baseball](https://support.trackmanbaseball.com/hc/en-us/articles/5089413493787-Radar-Glossary-Of-Terms) - CSV file from the Trackman Baseball game tracking software. V3: Order, CSV Column Header, Descriptio...

9. [Use GroupKFold in nested cross-validation using sklearn](https://stackoverflow.com/questions/60996995/use-groupkfold-in-nested-cross-validation-using-sklearn) - Let's check those out: So, when you do a GroupKFold it will make sure that all samples from one grou...

10. [LeaveOneGroupOut — scikit-learn 1.8.0 documentation](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.LeaveOneGroupOut.html) - For instance the groups could be the year of collection of the samples and thus allow for cross-vali...

11. [Hyperparameter Tuning](https://mlflow.org/docs/latest/ml/traditional-ml/xgboost/) - Official MLflow documentation for LLM tracing, agent evaluation, prompt management, experiment track...

12. [ML Experiment Tracking | MLflow AI Platform](https://mlflow.org/docs/latest/ml/tracking/) - Official MLflow documentation for LLM tracing, agent evaluation, prompt management, experiment track...

13. [Managing Python Projects With uv: An All-in-One Solution](https://realpython.com/python-uv/) - Learn how to create and manage your Python projects using uv, an extremely fast Python package and p...

14. [Configuring projects | uv - Astral Docs](https://docs.astral.sh/uv/concepts/projects/config/) - uv is an extremely fast Python package and project manager, written in Rust.

15. [Working on projects | uv - Astral Docs](https://docs.astral.sh/uv/guides/projects/) - A guide to using uv to create and manage Python projects, including adding dependencies, running com...

16. [How to migrate from requirements.txt to pyproject.toml with uv](https://pydevtools.com/handbook/how-to/migrate-requirements.txt/) - Convert a requirements.txt-based project to pyproject.toml using uv init and uv add.

17. [Writing your pyproject.toml¶](https://packaging.python.org/en/latest/guides/writing-pyproject-toml/)

18. [load_ncaa_baseball_pbp: *Load cleaned NCAA baseball play-by-play data from the... in baseballr: Acquiring and Analyzing Baseball Data](https://rdrr.io/cran/baseballr/man/load_ncaa_baseball_pbp.html)

19. [pybaseball/docs/statcast_pitcher.md at master - GitHub](https://github.com/jldbc/pybaseball/blob/master/docs/statcast_pitcher.md) - This function retrieves percentile ranks for each player in a given year, including batters with 2.1...

20. [pybaseball/docs/statcast_batter.md at master · jldbc/pybaseball](https://github.com/jldbc/pybaseball/blob/master/docs/statcast_batter.md) - Pull current and historical baseball statistics using Python (Statcast, Baseball Reference, FanGraph...

21. [pybaseball/docs/batting_stats.md at master · jldbc/pybaseball](https://github.com/jldbc/pybaseball/blob/master/docs/batting_stats.md) - Pull current and historical baseball statistics using Python (Statcast, Baseball Reference, FanGraph...

22. [pybaseball/docs/pitching_stats.md at master · jldbc/pybaseball](https://github.com/jldbc/pybaseball/blob/master/docs/pitching_stats.md) - Pull current and historical baseball statistics using Python (Statcast, Baseball Reference, FanGraph...

23. [Making Sense of Division One Park Factors - College Splits Research](https://collegesplits.substack.com/p/making-sense-of-division-one-park) - Park adjustments are one thing, but what about strength of schedule? Even programs within the same c...

24. [We've Got College Data! - FanGraphs Baseball](https://blogs.fangraphs.com/weve-got-college-data/) - Division I data is updated daily and is available going back to 2021. wRC+, ERA-, and FIP- are confe...

25. [[PDF] baseballr: Acquiring and Analyzing Baseball Data - CRAN](https://cran.r-project.org/web/packages/baseballr/baseballr.pdf)

26. [pybaseball 2.0.0 - PyPI](https://pypi.org/project/pybaseball/2.0.0/) - Retrieve baseball data in Python

27. [jldbc/pybaseball: Pull current and historical baseball ... - GitHub](https://github.com/jldbc/pybaseball) - Pull current and historical baseball statistics using Python (Statcast, Baseball Reference, FanGraph...

28. [What Goes Into an MLB Draft Model: Batted Ball Profiles - Magnus](https://www.seemagnus.com/blog-posts-test/what-goes-into-a-mlb-draft-model-batted-ball-profiles)

29. [Predicting Future MLB Players Using Scouting Reports](https://arxiv.org/pdf/1910.12622.pdf)

30. [XGBoost "scale_pos_weight" vs "sample_weight" for Imbalanced ...](https://xgboosting.com/xgboost-scale_pos_weight-vs-sample_weight-for-imbalanced-classification/) - This example demonstrates how to use both parameters and compares their performance using evaluation...

31. [XGBoost for Imbalanced Classification](https://xgboosting.com/xgboost-for-imbalanced-classification/)

32. [How to Configure XGBoost for Imbalanced Classification](https://machinelearningmastery.com/xgboost-for-imbalanced-classification/) - The XGBoost algorithm is effective for a wide range of regression and classification predictive mode...

33. [Why Marcel Beat LightGBM: Building an NPB Player Performance ...](https://dev.to/yasumorishima/why-marcel-beat-lightgbm-building-an-npb-player-performance-prediction-system-2jcb) - I built a Japanese professional baseball (NPB) player performance prediction system using Marcel pro...

34. [Is this a proper cross-validation code with the Leave-One-Group-Out ...](https://github.com/scikit-learn/scikit-learn/discussions/27091) - I am trying to make it conduct. Each group indicates the collection of data coming from a given part...

35. [Optuna + XGBoost on a tabular dataset](https://aetperf.github.io/2021/02/16/Optuna-+-XGBoost-on-a-tabular-dataset.html) - databases, dataviz, datascience

36. [XGBoost Hyperparameter Optimization with Optuna](https://xgboosting.com/xgboost-hyperparameter-optimization-with-optuna/)

37. [Dashboard](https://optuna.org) - Optuna is an automatic hyperparameter optimization software framework, particularly designed for mac...

38. [waterfall plot — SHAP latest documentation](https://shap.readthedocs.io/en/latest/example_notebooks/api_examples/plots/waterfall.html) - Waterfall plots are designed to display explanations for individual predictions, so they expect a si...

39. [beeswarm plot — SHAP latest documentation](https://shap.readthedocs.io/en/latest/example_notebooks/api_examples/plots/beeswarm.html) - The beeswarm plot is designed to display an information-dense summary of how the top features in a d...

40. [Front page example (XGBoost) — SHAP latest documentation](https://shap.readthedocs.io/en/latest/example_notebooks/tabular_examples/tree_based_models/Front%20page%20example%20(XGBoost).html) - The code from the front page example using XGBoost. [1]: import xgboost import shap # train an XGBoo...

41. [Finding Pro Pitching Comps For Top 2024 MLB Draft College Pitchers](https://www.baseballamerica.com/stories/finding-pro-pitching-comps-for-top-2024-mlb-draft-college-pitchers/) - With more data available than ever, we draw similarities between draft arms and pro pitchers, includ...

42. [Problem with functions in st.cache_data - Using Streamlit](https://discuss.streamlit.io/t/problem-with-functions-in-st-cache-data/73552) - To make my app smoother, i'm trying to cache all my plotly charts. Each chart has a function that st...

43. [st.plotly_chart causes AtrributeError if inside of st.cache_data in ...](https://github.com/streamlit/streamlit/issues/8885) - In streamlit 1.34, caching a plotly chart works fine. However, in 1.35, presumably because of the ne...

44. [Plotly Performance Issues Despite Caching - Using Streamlit](https://discuss.streamlit.io/t/plotly-performance-issues-despite-caching/110491) - Hi, I'm encountering a performance bottleneck when using Plotly to display charts in my Streamlit ap...

45. [Querying multiple parquet files in a range using duckdb](https://stackoverflow.com/questions/78509455/querying-multiple-parquet-files-in-a-range-using-duckdb) - I have parquet files arranged in this format /db/{year}/table{date}.parquet In each year folder, the...

46. [Using DuckDB in Python to access Parquet data - Simon Willison: TIL](https://til.simonwillison.net/duckdb/parquet) - 3GB of data in 68 parquet files. Those files are 45MB each. DuckDB can run queries against Parquet d...

