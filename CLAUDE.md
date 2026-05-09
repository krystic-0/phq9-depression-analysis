# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Streamlit web app for PHQ-9 depression analysis and suicide detection. Two datasets are preprocessed, then analyzed via clustering, classification, and regression models, with results displayed in a multi-page UI.

## Running the app

```bash
# Install dependencies
pip install -r requirements.txt

# Start the app (main entry point with login/auth)
streamlit run main.py

# Standalone login page
streamlit run login.py

# Run all tests
python -m pytest test_system.py -v
# or
python test_system.py

# Run a single test
python -m pytest test_system.py::TestDataPreprocessing::test_missing_value_handling -v
```

## Run data pipeline (CLI, no UI)

```bash
python data_preprocessing.py          # Process raw CSVs → outputs/csv/preprocessed_*.csv
python model_building.py              # Run all ML pipelines → outputs/csv/*.csv + outputs/images/*.png
python optimized_clustering_suicide.py # Alternative clustering with t-SNE/PCA visuals
```

## Architecture

### Entry points and routing

- `main.py` — App entry point. Sets `st.set_page_config`, renders login/register UI, stores session in `st.session_state` + browser `localStorage`, then delegates to `app.main()` after auth. Uses `_inject_js()` helper (via `st.components.v1.html`) to persist auth tokens to browser localStorage for page-refresh survival.
- `login.py` — Standalone login page with its own `st.set_page_config`. Can be run independently; redirects to `/` on success.
- `app.py` — Main application shell. Checks login state, renders sidebar navigation with 4 pages, and dispatches to page render functions. Designed to be both imported (by `main.py`) and run directly.

### UI navigation (4 pages in sidebar)

1. **🔬 聚类相关** — K-Means clustering (K=3) on PHQ-9 features, t-SNE visualization, cluster profile stats
2. **🧪 分类模型相关** — Binary suicide text classification with LR/RF/XGBoost, confusion matrix, feature importance
3. **📈 预测模型相关** — PHQ-9 total score regression with RF/XGBoost/Stacking, prediction vs actual scatter, residual plots
4. **📊 应用内可视化** — PHQ-9 distribution, severity pie, symptom correlation heatmap, time-series trends, demographic analysis, word cloud

### Core processing modules

- `data_preprocessing.py` — Reads the two raw CSVs, handles encoding detection (chardet), missing value imputation (group-mean), text preprocessing (jieba for Chinese, NLTK for English), psycholinguistic feature extraction, SMOTE balancing for suicide data, and optional BERT/TF-IDF feature extraction. Outputs go to `outputs/csv/`.
- `model_building.py` — Contains `DepressionModelBuilder`, the central ML orchestrator. Public API:
  - `load_data()` — loads preprocessed CSVs into `self.depression_df` / `self.suicide_df`
  - `data_preparation()` — scales features, handles missing values via IterativeImputer
  - `clustering_analysis()` — K-Means (K=3) + t-SNE, returns (tsne_result, kmeans_labels)
  - `classification_modeling()` — LR/RF/XGBoost with SMOTE + 5-fold stratified CV
  - `prediction_modeling()` — RF/XGBoost/Stacking regressors with 5-fold CV, SHAP analysis
- `visualization.py` — All Plotly chart functions. Consistent purple color scheme (`#9370DB`). Functions accept DataFrames and return Plotly figures.
- `optimized_clustering.py` / `optimized_clustering_suicide.py` — Standalone clustering scripts with PCA/t-SNE visualization. `app.py` calls `optimized_clustering_suicide.py` via `subprocess.run` when the user clicks "重新聚类分析" (to avoid blocking the main thread).

### Auth

- Simple JSON file-based auth at `users.json` (dict of `username: password`).
- Default account: `admin` / `admin123`.
- Session persisted via Streamlit `session_state` + browser `localStorage` for page-refresh survival.
- Both `main.py` and `app.py` use a shared `_inject_js(code)` helper that wraps `st.components.v1.html` to run JavaScript in the browser (used for localStorage persistence).

### Session state pattern

`app.py` initializes session_state keys from `_SESSION_DEFAULTS` dict at module level (runs on import). Keys include: `logged_in`, `username`, `cluster_*`, `classification_*`, `prediction_*`, and `current_page`. This ensures keys exist before any Streamlit widget accesses them.

### Data flow

```
Raw CSVs → data_preprocessing.py → outputs/csv/preprocessed_*.csv
                                    ↓
                          model_building.py (or UI "rebuild" buttons)
                                    ↓
                    outputs/csv/*_results.csv + outputs/images/*.png
                                    ↓
                          app.py loads + renders in Streamlit
```

### Output files

- `outputs/csv/` — `preprocessed_depression_data.csv`, `preprocessed_suicide_data.csv`, `cluster_profiles.csv`, `classification_results.csv`, `classification_cv_results.csv`, `prediction_results.csv`, `prediction_cv_results.csv`, `feature_importance.csv`
- `outputs/images/` — t-SNE/PCA plots, clustering evaluation, confusion matrix, feature importance, prediction scatter, residual plots, SHAP summary, word cloud

### Key patterns

- `DepressionModelBuilder` is instantiated in both `app.py` (for on-demand rebuild) and `model_building.py __main__` (for CLI batch run). It caches loaded data as instance attributes.
- The UI in `app.py` checks for pre-generated output files (images, CSVs) and displays them if they exist, otherwise shows warnings. "Rebuild" buttons re-run the pipeline and call `st.rerun()` to refresh.
- All randomness is seeded (42) across the codebase for reproducibility.
- Text preprocessing handles both Chinese (jieba) and English (NLTK) with custom suicide/depression keyword dictionaries and synonym normalization.
- Data loading in `app.py` uses `@st.cache_data(ttl=3600)` to cache preprocessed CSVs for 1 hour.
- Tests in `test_system.py` use `unittest` (can be run with either `python test_system.py` or `python -m pytest`).
- `kaleido` is required for Plotly static image export (`plotly.io.write_image`).
