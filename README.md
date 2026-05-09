# MLB Draft Dashboard

Local Streamlit dashboard for MLB draft board building and player deep-dives.

## What This Repo Contains

- `R/export_dashboard_data.R`: reads a SQLite TrackMan-style database and writes frozen Parquet exports to `exports/dashboard/`
- `app/streamlit_app.py`: Streamlit entrypoint
- `src/mlb_draft_dashboard/`: dashboard logic, DuckDB/Parquet loading, ranking, and local state persistence
- `scripts/generate_demo_exports.py`: creates demo Parquet exports when you want to preview the UI without wiring real data first
- `tests/`: smoke tests for export validation, ranking logic, and local state persistence

## Expected Export Files

The app reads these Parquet files from `exports/dashboard/`:

- `hitters_board.parquet`
- `pitchers_board.parquet`
- `player_trends.parquet`
- `hitter_bbe_detail.parquet`
- `pitcher_pitchtype_detail.parquet`
- `benchmarks_acc_sec.parquet`
- `explanations.parquet`
- `diagnostics.parquet`
- `qa.parquet`

## Local Analyst State

The dashboard stores favorites, role-fit tags, notes, and saved weight/filter views in a local SQLite file:

- `app_state.sqlite`

This file is intentionally separate from the exported data so rerunning the R export step does not wipe analyst notes.

## Getting Started

1. Create and activate a virtualenv.
2. Install the app dependencies:

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

3. Create demo exports, or run the R exporter against a real SQLite source:

```bash
PYTHONPATH=src .venv/bin/python scripts/generate_demo_exports.py
```

4. Launch the app:

```bash
PYTHONPATH=src .venv/bin/streamlit run app/streamlit_app.py
```

5. Run tests:

```bash
PYTHONPATH=src .venv/bin/pytest
```

## Real Data Export

Run the R exporter from your normal local R installation:

```bash
Rscript R/export_dashboard_data.R \
  --db ../Apollo/.local/uploads/1775510166-VTBaseball2025-2026.db \
  --table VTData \
  --output exports/dashboard \
  --team-mapping configs/team_mapping_all_teams.csv
```

Optional:

- `--player-metadata <csv>` to enrich players with class year or custom fields

## Recommended Dev Data

For a smaller but still representative source that covers every team in the broader 2025-2026 dataset:

```bash
python3 scripts/create_representative_sample.py \
  ../Apollo/.local/uploads/1775510166-VTBaseball2025-2026.db \
  data_all_teams_sample.db \
  VTData \
  250
```

This writes a smaller SQLite file with all teams still represented.

## Team Mapping Template

The exporter automatically loads built-in defaults from [configs/default_team_mapping.csv](/Users/brendanwaterval/Desktop/vt_baseball/MLB_Draft_Model/configs/default_team_mapping.csv). Your custom CSV acts as an override layer.

If you want to bootstrap a mapping file from a SQLite source:

```bash
python3 scripts/bootstrap_team_mapping.py \
  ../Apollo/.local/uploads/1775510166-VTBaseball2025-2026.db \
  configs/team_mapping_all_teams.csv
```

The team-mapping CSV should include at least:

- `team_code`
- `school_name`
- `conference`

If you want ACC/SEC benchmarking to work immediately, make sure those team codes are mapped correctly.
