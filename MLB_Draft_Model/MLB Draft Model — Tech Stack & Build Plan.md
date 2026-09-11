# MLB Draft Model — Tech Stack & Build Plan

## Overview

This document provides a concrete technology stack recommendation and phased build plan for an MLB draft model built on college Trackman data. The stack is designed around three priorities: (1) keeping everything in Python, with R available for statistical work you already know; (2) minimizing infrastructure complexity for a solo or small-team project; and (3) ensuring the output is shareable as an interactive prospect dashboard. Every tool selected is open-source, free, and has strong 2025–2026 community support.

***

## Core Language Decision: Python as Primary, R as Supplement

**Python is the right primary language for this project.** The reasons are decisive: Python's ML ecosystem (scikit-learn, XGBoost, LightGBM, SHAP) is deeper and more production-grade than R's equivalents, it integrates directly with DuckDB and Streamlit for the data layer and dashboard, and `pybaseball` gives you a one-liner interface to all of MLB Statcast. For a solo project that ends in a deployable dashboard, Python handles the entire stack end-to-end without hand-offs.[^1][^2][^3][^4]

**R remains valuable for two specific tasks:**
- `baseballr`: the best interface for NCAA baseball data and the Chadwick Bureau player ID crosswalk (`chadwick_player_lu()`), which is the cleanest way to link college Trackman player names to MLB Advanced Media (MLBAM) IDs[^5][^6]
- Statistical modeling and visualization using `tidyverse`/`ggplot2` for exploratory analysis, where R's grammar of graphics still produces more polished plots for research purposes[^7]

The practical workflow is: pull NCAA and crosswalk data in R → write to Parquet → ingest into Python for all modeling and dashboard work. This hybrid is well-supported since both R and Python can read/write Parquet natively.[^8]

***

## Layer 1: Data Ingestion & Storage

### File Format: Parquet

Store all persistent data — raw Trackman CSVs, normalized feature tables, player crosswalks, Statcast pulls — as **Apache Parquet files**. Parquet's columnar storage makes analytical queries (filtering by conference, player, season, pitch type) 10–100x faster than CSV for the same file. File sizes compress 2–5x vs. CSV, and crucially, both DuckDB and Python's Pandas/Polars read Parquet natively without any schema conversion.[^9][^10][^8]

Directory structure:
```
data/
  raw/           # Original Trackman CSVs, as-received
  interim/       # Cleaned, merged, park/conf-adjusted
  features/      # Final player-season feature tables
  external/      # Statcast pulls, Chadwick crosswalk, park factors
models/
  artifacts/     # Trained XGBoost/LightGBM model files
mlruns/          # MLflow experiment tracking
app/             # Streamlit dashboard
notebooks/       # EDA and research notebooks
```

### Query Engine: DuckDB

Use **DuckDB** as your in-process analytical query engine. Unlike SQLite (which is optimized for transactional OLTP workloads), DuckDB uses vectorized columnar execution purpose-built for analytical queries (OLAP). On complex aggregations and joins — exactly what you're doing when computing per-player multi-year Trackman summaries or running conference-level normalization — DuckDB outperforms SQLite by 5–100x depending on query complexity. It queries Parquet files directly without loading them first, integrates with Pandas and Polars dataframes as zero-copy, and requires zero database server setup — it's an embedded library like SQLite.[^11][^12][^13][^14][^9]

One real-world baseball analytics platform (BaseballIQ) reported DuckDB handling 30-day rolling averages and cross-pitcher percentiles "in seconds, zero database at runtime". That's the exact query pattern for your normalization pipeline.[^15]

**DuckDB for this project:**
```python
import duckdb
# Query Parquet files directly — no import step
df = duckdb.query("""
    SELECT player_id, season, AVG(release_speed) as avg_velo,
           AVG(induced_vert_break) as avg_ivb,
           PERCENTILE_CONT(0.9) WITHIN GROUP (ORDER BY exit_speed) as ev_p90
    FROM 'data/interim/trackman_2024_2026.parquet'
    WHERE pitch_type = 'FF'
    GROUP BY player_id, season
""").df()
```

### Data Ingestion Libraries

| Source | Tool | Notes |
|--------|------|-------|
| Trackman CSVs (yours) | `polars` or `pandas` | Parse and merge raw exports[^16] |
| MLB Statcast | `pybaseball` (`statcast()`, `statcast_pitcher()`, `statcast_batter()`) | Last commit Jan 2026, 30 contributors[^5][^3] |
| NCAA game/schedule data | `baseballr` (R) → Parquet | `load_ncaa_baseball_pbp()`, `load_ncaa_baseball_schedule()`[^6] |
| Player ID crosswalk | `baseballr::chadwick_player_lu()` (R) → Parquet | Links name/school to MLBAM ID[^6] |
| College park factors | College Splits (manual CSV download or scrape) | Park factor tables by D1 school[^17] |
| FanGraphs conf-adj stats | `pybaseball.fg_batting_data()` / `fg_pitching_data()` | Conference-adjusted wRC+, ERA-, FIP- for D1 (added 2025)[^18] |

### Pandas vs. Polars

Use **Polars for the ETL pipeline** (loading raw Trackman CSVs, merging multi-year files, computing normalization passes) and **Pandas for modeling** (scikit-learn, XGBoost, SHAP all expect Pandas DataFrames). Polars is 4.6x faster than Pandas on row filtering and 2.6x faster on group-aggregation operations on 1GB+ datasets. Its Rust-based multi-threaded execution is the right tool when reading and merging 3 seasons of pitch-level Trackman data (which can easily reach 500K–2M rows). Converting back to Pandas for modeling is a one-liner: `polars_df.to_pandas()`.[^19][^20]

***

## Layer 2: Feature Engineering

All feature engineering runs in Python, orchestrated with standard notebooks or scripts. The key pipeline steps are:

1. **Park factor join**: Merge D1 park factors onto each game's outcome metrics by home team
2. **Conference tier encoding**: Assign each team a conference tier (P5 = 1, high-mid = 2, low-mid = 3, low = 4) based on historical FPI or custom SOS scores
3. **Metal bat EV discount**: Apply a 2.5–3.0 mph downward adjustment to all EV-based features for hitters
4. **Multi-year aggregation**: For players with 2024+2025+2026 data, compute weighted season averages (more recent season weighted higher) and year-over-year trend features
5. **College Stuff+ computation**: Train a within-dataset XGBoost model predicting run value per pitch from physical metrics; use these scores as features in the draft model itself
6. **Chadwick crosswalk join**: Match Trackman player names to MLBAM IDs; pull Statcast outcomes for historically drafted players to build training labels

***

## Layer 3: Modeling

### Core Framework: XGBoost + LightGBM + scikit-learn

Both XGBoost and LightGBM are the right algorithms for this data. The key comparison points:[^21]

| Criterion | XGBoost | LightGBM |
|-----------|---------|----------|
| Classification (reach-MLB probability) | **Preferred** — outperforms LightGBM on AUC in most sports benchmarks[^21] | Strong alternative |
| Regression (performance projection) | Good | **Preferred** — slightly faster training on larger feature sets |
| Handling imbalanced classes (~80% negative) | `scale_pos_weight` parameter | `is_unbalance=True` or `class_weight` |
| Feature importance | Built-in gain importance | Built-in split importance |
| Interpretability via SHAP | **TreeExplainer is fastest on XGBoost** | Supported but slightly slower |
| Speed | Slower than LightGBM | 2–5x faster training |

**Recommendation**: Use XGBoost for the Stage 1 classification model (reach-MLB probability) because interpretability via SHAP is critical and XGBoost produces cleaner, more focused SHAP outputs than neural networks or even LightGBM. Use LightGBM for Stage 2 regression (performance tier) where you want faster iteration. Wrap both in a scikit-learn Pipeline for reproducibility.[^22][^23]

One important empirical finding from a comparable baseball prediction project (NPB player performance): Marcel-style projection outperformed ML on a pure MAE basis (OPS MAE 0.048 vs. 0.062 for XGBoost). This is a known phenomenon in baseball analytics when the training set is small. **Mitigation**: use multi-year weighted averages as baseline features (effectively embedding Marcel-style regression toward the mean into the feature set) before passing to XGBoost.[^24]

### Experiment Tracking: MLflow

Use **MLflow** to track all model runs, hyperparameter searches, and validation metrics. MLflow 3.9.0 (current as of 2026) provides:[^25][^26]
- Automatic logging of XGBoost/LightGBM parameters, metrics, and model artifacts via `mlflow.xgboost.autolog()`
- A local web UI to compare runs side-by-side (AUC, PR-AUC, RMSE, Spearman rank correlation)
- Model Registry for versioning the Stage 1 and Stage 2 models separately, with stage transitions (dev → staging → production)
- Git commit linking so every model version traces back to the exact code that produced it[^26]

This matters because you'll want to retrain every year as new draft classes confirm or refute projections. MLflow lets you systematically compare the 2025-trained model vs. the 2026-retrained model.

```python
import mlflow
import mlflow.xgboost

mlflow.set_experiment("draft_model_stage1_classification")
with mlflow.start_run():
    mlflow.xgboost.autolog()
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
    mlflow.log_metric("val_auc", roc_auc_score(y_val, model.predict_proba(X_val)[:,1]))
    mlflow.log_metric("val_prauc", average_precision_score(y_val, model.predict_proba(X_val)[:,1]))
```

### Hyperparameter Tuning

Use **Optuna** for Bayesian hyperparameter optimization. It's lightweight, integrates directly with XGBoost/LightGBM, and is far more sample-efficient than grid search — critical when your training set size (drafted players with outcomes) may be in the hundreds rather than thousands of rows.

### Validation

Use **leave-one-draft-class-out cross-validation** (temporal CV). For each fold, train on all draft classes except one, validate on the held-out class. Report AUC-ROC and PR-AUC for Stage 1; Spearman rank correlation and RMSE for Stage 2.

***

## Layer 4: Interpretability

### SHAP

Install the `shap` library and use `shap.TreeExplainer` for XGBoost. XGBoost's tree structure enables exact Shapley value computation (not an approximation), making the explanation fast and provably correct. The three plot types you'll use most:[^27][^23]

- **`shap.summary_plot`** (beeswarm): global feature importance with directional arrows — shows that higher IVB raises draft probability, higher BB% lowers it
- **`shap.waterfall_plot`**: per-player explanation — "Player X's score is above baseline primarily because of 97 mph velo (+0.12) and elite IVB (+0.09), offset by poor BB% (−0.06)"
- **`shap.dependence_plot`**: shows the non-linear relationship between a feature (e.g., EV percentile) and its SHAP contribution

These outputs are designed to be embedded directly in the Streamlit dashboard player card view.

***

## Layer 5: Dashboard

### Streamlit

**Streamlit is the right choice over R Shiny for this project** given that the entire modeling stack is Python-native. The guidance from current sources is unambiguous: "Use Shiny if you work in R. Use Streamlit if you work in Python." Streamlit acquired by Snowflake in 2022 has become the dominant choice for Python data scientists building production analytics apps. Key advantages for this use case:[^28]

- Zero-config deployment to Streamlit Community Cloud (free tier) — shareable URL immediately
- `st.plotly_chart()` renders interactive Plotly figures (movement plots, EV/LA scatter, trend lines) natively
- Widget system (sliders, multiselect, radio buttons) maps directly to prospect filter controls (conference, position, class year, model score threshold)
- File upload widget enables loading a new player's Trackman CSV and running the model on-the-fly

Streamlit's main limitation — full script re-execution on any widget change — is manageable with `@st.cache_data` decorators on your DuckDB query functions and `@st.cache_resource` on the loaded model objects.[^29]

### Visualization Libraries

| Purpose | Library | Notes |
|---------|---------|-------|
| Movement plots (IVB vs HB scatter) | `plotly.express` | Interactive hover tooltips with player name/pitch type |
| EV/LA scatter | `plotly.express` | Color by barrel/hard-hit/weak contact zone |
| Year-over-year trends | `plotly.graph_objects` | Line charts with confidence bands |
| SHAP waterfall plots | `shap` + `matplotlib` | Embed as static PNG in Streamlit |
| Movement plot on baseball field | `matplotlib` + custom | Use GeomMLBStadiums-style coordinate system |

***

## Full Stack Summary

| Layer | Tool | Why |
|-------|------|-----|
| Primary language | Python 3.11+ | Full ML + dashboard stack[^1][^2] |
| Secondary language | R (`baseballr`) | NCAA data, Chadwick ID crosswalk[^5][^6] |
| File format | Apache Parquet | 10–100x faster than CSV for analytical queries[^8][^10] |
| Query engine | DuckDB | Vectorized OLAP, queries Parquet directly, zero setup[^11][^13] |
| Data ingestion (MLB) | `pybaseball` | Statcast, FanGraphs, Baseball Reference[^3] |
| Data manipulation (ETL) | `polars` | 3–5x faster than pandas on large files[^19][^20] |
| Data manipulation (modeling) | `pandas` | Required by scikit-learn, XGBoost, SHAP[^7] |
| ML core | `xgboost`, `lightgbm`, `scikit-learn` | Best tabular ML for this data type[^21] |
| Hyperparameter tuning | `optuna` | Bayesian search, efficient on small training sets |
| Experiment tracking | `MLflow` | Model versioning, run comparison, Git linking[^25][^26] |
| Explainability | `shap` | TreeExplainer for XGBoost; waterfall + beeswarm plots[^27][^23] |
| Dashboard | `streamlit` | Python-native, free deployment, fast iteration[^28] |
| Visualization | `plotly` | Interactive movement plots, EV/LA scatter, trend lines |
| Version control | `git` + GitHub | Track code + link to MLflow runs |
| Environment management | `uv` or `conda` | Reproducible Python environments |

***

## Phased Build Plan

### Phase 1: Data Foundation (Weeks 1–2)
1. Set up `git` repo and Python environment (`uv` or `conda`) with all dependencies pinned
2. Write Parquet ingestion script for your raw Trackman CSVs (2024, 2025, 2026) using Polars
3. In R: pull Chadwick crosswalk via `baseballr::chadwick_player_lu()`, pull NCAA schedule and team data via `baseballr::load_ncaa_baseball_*()`, write to Parquet
4. Pull MLB Statcast data for all players matching your crosswalk via `pybaseball`; write to Parquet
5. Download D1 park factors from College Splits; store in `data/external/`

### Phase 2: Normalization Pipeline (Weeks 3–4)
1. Build DuckDB-based normalization queries: park factor joins, conference tier encoding, EV adjustment
2. Compute per-player-season feature tables (physical Trackman metrics, adjusted outcome stats, multi-year trends)
3. Train within-dataset college Stuff+ model; add scores as features
4. Run data quality audit: flag players with sparse Trackman coverage (<100 pitches or <30 BBEs)

### Phase 3: Training Data Assembly (Weeks 5–6)
1. Identify all players in your dataset who subsequently played in affiliated baseball (via Chadwick crosswalk + Baseball Reference MiLB data)
2. Pull minor league performance outcomes for 2019–2023 draft class players
3. Define target variable: binary reach-AA-within-3-years or continuous park-adjusted wRC+/FIP- at highest level reached
4. Merge features and labels into final training table

### Phase 4: Model Training (Weeks 7–8)
1. Set up MLflow experiment tracking
2. Train Stage 1 XGBoost classifier (reach-MLB probability); tune with Optuna; log all runs
3. Train Stage 2 LightGBM regressor (performance level); tune with Optuna; log all runs
4. Run leave-one-draft-class-out CV; report AUC-ROC, PR-AUC (Stage 1) and Spearman rank correlation (Stage 2)
5. Compute SHAP values; produce beeswarm and waterfall plots for top prospects

### Phase 5: Dashboard Build (Weeks 9–10)
1. Build Streamlit app with: prospect board (sortable by model score), player card view (movement plot, EV/LA scatter, SHAP waterfall, comp player), trend view (multi-year metric trajectories)
2. Add filters: position, class year, conference, handedness, model score threshold
3. Cache DuckDB queries and model inference with `@st.cache_data` / `@st.cache_resource`
4. Deploy to Streamlit Community Cloud (free, public URL) or keep local for private use

### Phase 6: 2026 Draft Scoring & Iteration (Week 11+)
1. Run the trained model on 2026 draft-eligible players in your Trackman dataset
2. Generate ranked prospect board with scores, SHAP explanations, and comp players
3. After the 2026 draft, collect draft slot data as ground truth; compute rank correlation between model scores and actual draft order
4. Retrain annually as new draft class outcomes become available

---

## References

1. [Python or R ?](https://www.reddit.com/r/sportsanalytics/comments/1ik045u/python_or_r/) - Python or R ?

2. [Python vs R for Sports Analytics - YouTube](https://www.youtube.com/watch?v=0A5nQOQIHLM) - ... Sports Data Forever by Building Your Own Web Scraping Pipeline: https://mckay-s-site.thinkific.c...

3. [jldbc/pybaseball: Pull current and historical baseball ... - GitHub](https://github.com/jldbc/pybaseball) - Pull current and historical baseball statistics using Python (Statcast, Baseball Reference, FanGraph...

4. [Introducing pybaseball: an Open Source Package for Baseball Data ...](https://jamesrledoux.com/projects/open-source/introducing-pybaseball/) - The stats that this library provides range from the classics (BA, RBI, HR, W, L, K, IP), to the slig...

5. [PySport Opensource Overview](https://opensource.pysport.org/?sports=Baseball)

6. [NCAA Baseball](https://billpetti.github.io/baseballr/reference/index.html)

7. [R vs Python in 2025: A Complete Comparison for Data Science](https://www.r-bloggers.com/2025/09/r-vs-python-in-2025-a-complete-comparison-for-data-science/) - In this article, we compare them across multiple dimensions—usability, visualization, machine learni...

8. [Parquet Data Format: Exploring Its Pros and Cons for 2025](https://edgedelta.com/company/blog/parquet-data-format) - Parquet shrinks file size. Its built-in compression cuts storage by 2–5x, saving money and speeding ...

9. [DuckDB vs SQLite: A Complete Database Comparison - DataCamp](https://www.datacamp.com/blog/duckdb-vs-sqlite-complete-database-comparison) - SQLite prioritizes transactional simplicity and reliability, while DuckDB is optimized for analytica...

10. [Parquet File Format – Everything You Need to Know!](https://towardsdatascience.com/parquet-file-format-everything-you-need-to-know/) - Data compression – by applying various encoding and compression algorithms, Parquet file provides re...

11. [DuckDB vs SQLite: Which Embedded Database Should You Use?](https://motherduck.com/learn-more/duckdb-vs-sqlite-databases/) - DuckDB consistently outperforms SQLite for analytical queries on larger datasets due to its columnar...

12. [DuckDB vs SQLite: Choosing the Right Embedded Database](https://betterstack.com/community/guides/scaling-python/duckdb-vs-sqlite/) - Compare DuckDB and SQLite to find the best embedded database for your needs. Learn their key differe...

13. [DuckDB vs SQLite: Performance, Speed, and Use Cases Compared](https://www.hakunamatatatech.com/our-resources/blog/sqlite) - SQLite is optimized for high-speed transactional operations (OLTP), while DuckDB is built for high-p...

14. [Why We Moved from SQLite to DuckDB: 5x Faster Queries - Reddit](https://www.reddit.com/r/dataengineering/comments/1ixbrkc/why_we_moved_from_sqlite_to_duckdb_5x_faster/) - SQLite is only 2x as fast as duckdb for transactional workloads while duck is 77% smaller size but u...

15. [MLB Analytics Platform with AI-Generated Scouting Reports - LinkedIn](https://www.linkedin.com/posts/ifrg_dataengineering-machinelearning-sportsanalytics-activity-7439671910713655297-CL0_) - The data pipeline was the easy part — teaching the AI how to reason about sport-specific context too...

16. [Why Data Teams Are Moving from Pandas to Polars in 2025 - LinkedIn](https://www.linkedin.com/pulse/why-data-teams-moving-from-pandas-polars-2025-datumlabsio-anorf) - Technical view: Pandas runs single-threaded. Polars automatically uses all CPU cores. On joins and a...

17. [Making Sense of Division One Park Factors - College Splits Research](https://collegesplits.substack.com/p/making-sense-of-division-one-park) - Park adjustments are one thing, but what about strength of schedule? Even programs within the same c...

18. [We've Got College Data! - FanGraphs Baseball](https://blogs.fangraphs.com/weve-got-college-data/) - Division I data is updated daily and is available going back to 2021. wRC+, ERA-, and FIP- are confe...

19. [Pandas vs. Polars: Benchmarking Dataframe Libraries with Real ...](https://pipeline2insights.substack.com/p/pandas-vs-polars-benchmarking-dataframe) - Polars is 5x faster than Pandas when loading a 1GB CSV file. Memory consumption is way lower with Po...

20. [Pandas vs Polars: Which Data Processor Runs Faster - Shuttle.dev](https://www.shuttle.dev/blog/2025/09/24/pandas-vs-polars) - Our benchmarks showed Polars delivering a 3.3x speedup over Pandas for this ETL workload, with signi...

21. [Integration of machine learning XGBoost and SHAP models for NBA ...](https://pmc.ncbi.nlm.nih.gov/articles/PMC11265715/) - This study investigated the application of artificial intelligence in real-time prediction of profes...

22. [Understanding Model Predictions with SHAP - XGBoost vs Neural ...](https://www.youtube.com/watch?v=eLrtufRomh8) - Train XGBoost and Neural Network models for classification · Use SHAP to explain predictions from bo...

23. [A Gentle Introduction to SHAP for Tree-Based Models](https://machinelearningmastery.com/a-gentle-introduction-to-shap-for-tree-based-models/) - In this article, we'll explore how to apply SHAP to tree-based models using a well-optimized XGBoost...

24. [Why Marcel Beat LightGBM: Building an NPB Player Performance ...](https://dev.to/yasumorishima/why-marcel-beat-lightgbm-building-an-npb-player-performance-prediction-system-2jcb) - I built a Japanese professional baseball (NPB) player performance prediction system using Marcel pro...

25. [Model Versioning with MLflow: Tracking and Managing Your ML Models - Java Code Geeks](https://www.javacodegeeks.com/2025/06/model-versioning-with-mlflow-tracking-and-managing-your-ml-models.html) - A practical guide to model versioning with MLflow: Learn how to track ML experiments, visualize resu...

26. [ML Model Versioning and Experiment Tracking with MLflow - Dasroot!](https://dasroot.net/posts/2026/02/ml-model-versioning-experiment-tracking-mlflow/) - Learn how MLflow enables robust model versioning and experiment tracking for reproducible, scalable ...

27. [XGBoost Feature Importance with SHAP Values](https://xgboosting.com/xgboost-feature-importance-with-shap-values/)

28. [Shiny vs Streamlit for Data Apps in 2026: Which Should You Use?](https://rguides.dev/articles/shiny-vs-streamlit/) - Streamlit and Shiny both let you build data apps fast. Here is how to choose the right framework for...

29. [Streamlit vs Shiny for Python: The Best Choice for Lab Apps and ...](https://evo-byte.com/streamlit-vs-shiny-for-python-the-best-choice-for-lab-apps-and-dashboards/) - Intro: Why lab teams ask this question You want a fast way to turn notebooks into usable tools. Your...

