# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Change point detection research for queueing systems (M/M/1, Cox/M/1 queues). The project detects regime shifts in arrival-rate time series and predicts queue-length distributions (PMFs).

## Setup

```bash
pip install -r requirements.txt
# Additional deps used but not listed: scipy, statsmodels
```

## Key Notebooks (entry points)

| Notebook | Purpose |
|---|---|
| `data_generation.ipynb` | Simulate arrival data (M/M/1, CoxM/1, step-change processes) |
| `cpd_mmd_version 2.ipynb` | Change point detection via MMD |
| `pmf_notebook.ipynb` | Plot queue-length PMFs |
| `Z_t_predict.ipynb` | Predict the Cox process intensity Z(t) |
| `notebook0819karman_filter.ipynb` | Kalman filter–based predictor experiments |

## Architecture

### `src/` — main importable package

```
src/
  bayesian_online_predictor.py   # BayesianOnlinePredictor (BOCPD, Student-t)
  unified_predictor.py           # UnifiedPredictor + PredictionConfig (method abstraction)
  change_point_detection.py      # MMD-based CPD with OLS / diff-OLS model fitting
  compute_pmf_functions.py       # PMF computation for M/m/1-type queues (uniformization)
  config_templates.py            # ConfigTemplates + get_config(scenario) presets
  utils.py                       # Metrics, KL divergence, load/save helpers
```

**Three prediction methods** selectable via `PredictionConfig(method=...)`:
- `'bayesian'` — `BayesianOnlinePredictor`: online BOCPD with run-length distribution
- `'kalman'` — `RollingKalmanPredictor` (imported lazily from `kalman_predictor.py`)
- `'cpd'` — `OLScpdPredictor` using MMD deviation detection (imported lazily from `cpd_predictor.py`)

**Quick usage:**
```python
from src import UnifiedPredictor, PredictionConfig
config = PredictionConfig(method='bayesian', hazard_lambda=50, alarm_threshold=0.1)
results = UnifiedPredictor(config).predict(df)   # df has 'time' and 'value' columns

# Or use presets:
from src.config_templates import get_config
config = get_config('queue')   # 'default'|'sensitive'|'robust'|'high_freq'|'queue'|'arrival'
```

### `Simulation_code/` — standalone simulation scripts

Scripts generate `.pickle` histogram files saved to `data_integrated/Simulation_histograms/`:
- `MM1-simulation-t0t1.py` — M/M/1 queue
- `CoxM1-simulation-t0t1.py` — Cox/M/1 queue
- `step2M1-simulation-t0t1-v2.py` — step-change arrival process

### `data_integrated/` — data files

- `arrival_data/` — CSV files of simulated arrival time series (naming: `initial_value_<Z0>_samples_<N>_a_<a>_b_<b>...csv`)
- `Simulation_histograms/` — `.pickle` histogram files from simulation scripts
- `pmf_data/` — computed PMF arrays
- `result_data/` — prediction result outputs

### `utils/plot.py`

Lightweight plotting helpers (separate from `src/utils.py`).

## Data Format Conventions

- DataFrames passed to predictors must have `'time'` and `'value'` columns.
- Arrival data CSVs encode parameters in filename: `a` = mean-reversion speed, `b` = long-term mean, `delta` = noise magnitude.
- PMF/histogram pickles store numpy arrays of queue-length distributions at observation times `t1`, `t5`, etc.
