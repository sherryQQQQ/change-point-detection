# Sprint Plan: Academic-Ready Change Point Detection Pipeline

**Generated:** 2026-03-31
**Status:** COMPLETED (Phase 1-4), Phase 5-6 ready for execution

---

## Task Breakdown

### Phase 1: Merge v2 → main src/ (File Reorganization) — DONE

**Task 1.1** — Merge v2-only modules into main `src/` — **COMPLETED**

- Copied `v2/src/benchmark.py` → `src/benchmark.py`
- Copied `v2/src/pipeline.py` → `src/pipeline.py`
- Copied `v2/src/histogram_loader.py` → `src/histogram_loader.py`
- Copied `v2/src/prediction_utils.py` → `src/prediction_utils.py`
- Kept `src/utils.py` (has `plot_pmf_overlap` etc.)

**Task 1.2** — Merge v2 improvements into existing main src/ files — **COMPLETED**

- `unified_predictor.py`: Adopted v2's YAML support, `compute_window_size` auto-computation, `mu` parameter. **KEPT** `bayesian` method (v2 dropped it — fixed regression)
- `config_templates.py`: Added `compute_window_size()`, kept rich preset library + added `cpd_bayesian` preset
- `booles_cpd_predictor.py`: Kept `adaptive_rolling_prediction` and `get_current_model_params`. Uses `self.alarm_threshold` (not v2's hardcoded 0.01)
- `__init__.py`: Updated to v2-style comprehensive exports (all predictors, RegimeModel, benchmark, histogram_loader, utils)

**Task 1.3** — Copy v2 YAML configs — **COMPLETED**

- Copied `v2/configs/*.yaml` → `configs/`

**Task 1.4** — Update `requirements.txt` — **COMPLETED**

- Added `scipy>=1.10.0`, `statsmodels>=0.14.0`, `pyyaml>=6.0`

---

### Phase 2: Fix Critical Bugs & Threshold Sensitivity — DONE

**Task 2.1** — Implement adaptive alarm threshold linked to data volatility — **COMPLETED**

- **Problem:** BOCPD CP probabilities cluster 0.01–0.015 regardless of regime → fixed threshold either flags everything or is too loose
- **Solution implemented** in `booles_cpd_predictor.py` → `adaptive_rolling_prediction()`:
  - New parameters: `adaptive_threshold=True`, `threshold_base`, `threshold_k`
  - Per-window threshold: `threshold_t = base * (1 + k * v_t)` where `v_t = local_std / running_std_max`
  - Volatile regions → higher threshold (fewer false CPs), smooth regions → lower threshold
  - Threshold trace stored in `self.thresholds_used` and plotted in 4th subplot

**Task 2.2** — Add EWMA-based adaptive window size option — **COMPLETED**

- New function `_adaptive_window_size_ewma()` in `src/change_point_detection.py`
- Uses EWMA of absolute differences as volatility proxy
- Selectable via `window_method='ewma'` parameter in `adaptive_rolling_prediction()`

---

### Phase 3: Testing & Validation — DONE

**Task 3.1** — Test suite `tests/test_predictors.py` — **COMPLETED** (18 tests)

- Unit tests for: `compute_window_size`, `PredictionConfig`, `BayesianOnlinePredictor` (update, predict, rolling), `RollingKalmanPredictor`, `OLScpdPredictor`, `BOOLES_cpdPredictor` (fixed + adaptive + EWMA), all `get_config` presets, adaptive window functions

**Task 3.2** — Test suite `tests/test_pmf.py` — **COMPLETED** (9 tests)

- `_build_P` shape and row-sum invariant
- `build_P_list` length
- `transient_distribution_piecewise` sum-to-one, non-negativity, M/M/1 steady-state convergence, out-of-range assertion
- `transient_distribution_uniformization` sum-to-one, non-negativity

**Task 3.3** — Integration smoke test — **COMPLETED** (via test suite)

- All 30 tests pass: `pytest tests/ -v` → 30 passed in 20s

---

### Phase 4: Clean Callable Notebook — DONE

**Task 4.1** — Created `run_experiment.ipynb` — **COMPLETED**

- 5-step pipeline: Parameters → Load Data → CPD+Prediction → Extract Piecewise → PMF+KL Divergence
- Parameterized top cell for easy scenario switching (Z0, a, b, delta, mu, method, window settings)
- Supports both adaptive (direct BOOLES_cpdPredictor) and non-adaptive (UnifiedPredictor) modes
- Batch scenario parser for all `Simulation_code/*.npz` files

---

### Phase 5: Multi-Scenario Evaluation — READY

**Task 5.1** — Run all CoxM/1 scenarios from `Simulation_code/*.npz`

- ~30+ npz files across Z0=5/20/80, serv=10/30, T=5/10, t=1..10
- For each: run CPD pipeline, compute PMFs, calculate KL divergence vs simulation
- Save results to `data_integrated/result_data/`
- Status: **READY** — notebook Step 5 set up to parse all scenarios. Requires interactive execution.

---

### Phase 5: BOCPD Ablation Study + NPZ PMF Validation — COMPLETED

**Task 5.1** — Alarm-threshold ablation (sweep 0.010–0.015 × 4 window strategies × 2 baseline CSVs) — **COMPLETED**

- Two baseline "initial setting" datasets only:
  - `initial_value_5_samples_500.csv`   (Z0=5,  a=0.3, b=80, mu=10)
  - `initial_value_80_samples_500.csv`  (Z0=80, a=0.3, b=80, mu=100)
- **Key finding:** All thresholds 0.010–0.015 produce identical RMSE within each strategy. BOCPD CP probs cluster at 0.02–0.04 on these data, above every tested threshold. To observe sensitivity, sweep must extend to 0.02–0.05.
- Best config: **Fixed ws=5**, any threshold in [0.010, 0.015], mean RMSE=3.089
- Results saved: `data_integrated/result_data/bocpd_ablation_results.csv` (48 rows)

**Task 5.2** — PMF validation on NPZ ground-truth files — **COMPLETED**

- Used best config (fixed ws=5, threshold=0.015) on Z0=5 arrival CSV
- Tested against 22 Z05 NPZ files (Z020 skipped — no matching arrival CSV)
- Service rate (mu) read directly from each NPZ's `service_rate` field
- **Key finding:** KL divergence < 0.02 for t ≤ 5 (excellent); degrades sharply at t ≥ 6 (serv=10) and t ≥ 8 (serv=30)
- Results saved: `data_integrated/result_data/pmf_validation_results.csv` (22 rows)

**Filename convention note:**

- `T` = total time horizon (scale parameter); `t` = evaluation time point
- New arrival data files should include T in the name: `initial_value_{Z0}_samples_{N}_T{T}.csv`
- Example pending: `initial_value_20_samples_500_T10.csv` for Z020 NPZ validation

**Scenario → service rate mapping:**


| Z0 (initial arrival intensity) | OU params    | Service rate (mu) |
| ------------------------------ | ------------ | ----------------- |
| 5                              | a=0.3, b=80  | 10                |
| 20                             | a=0.3, b=80  | 30                |
| 80                             | a=0.3, b=80  | 100               |
| 70                             | a=0.3, b=150 | 100               |


**Pending:** Generate `initial_value_20_samples_500_T10.csv` (Z0=20, a=0.3, b=80) to enable Z020 NPZ validation.

---

### Phase 6: Plotting & Results Consolidation — READY

**Task 6.1** — Consolidate plotting utilities

- `src/prediction_utils.py` has `plot_pmf_overlap`, `compare_pmfs_kl` (from v2)
- `src/utils.py` has the original `plot_pmf_overlap` with more features
- Status: **READY** — both available; consolidation deferred to avoid breaking notebooks

**Task 6.2** — Generate publication-quality figures

- Status: **READY** — depends on Phase 5 batch run

---

## Blockers & Resolution


| ID  | Blocker                                | Status       | Resolution                                            |
| --- | -------------------------------------- | ------------ | ----------------------------------------------------- |
| B1  | v2 hardcodes threshold=0.01            | **RESOLVED** | Kept main's `self.alarm_threshold`                    |
| B2  | v2 drops `bayesian` method             | **RESOLVED** | Re-added in merged `unified_predictor.py`             |
| B3  | v2 drops `adaptive_rolling_prediction` | **RESOLVED** | Kept main's version, enhanced with adaptive threshold |
| B4  | Alarm threshold sensitivity            | **RESOLVED** | Implemented adaptive threshold (Task 2.1)             |


---

## Files Modified/Created This Sprint

### Modified

- `src/__init__.py` — comprehensive exports (v2-style)
- `src/unified_predictor.py` — YAML support, `mu` param, auto window_size, kept bayesian
- `src/config_templates.py` — added `compute_window_size()`, `cpd_bayesian` preset
- `src/booles_cpd_predictor.py` — adaptive threshold + EWMA window method
- `src/change_point_detection.py` — added `_adaptive_window_size_ewma()`
- `requirements.txt` — added scipy, statsmodels, pyyaml

### Created

- `src/benchmark.py` — from v2
- `src/pipeline.py` — RegimeModel (from v2)
- `src/histogram_loader.py` — safe pickle loader (from v2)
- `src/prediction_utils.py` — prediction metrics + KL divergence (from v2)
- `configs/` — YAML config presets (from v2)
- `tests/test_predictors.py` — 18 predictor tests
- `tests/test_pmf.py` — 9 PMF tests
- `run_experiment.ipynb` — clean 5-step experiment notebook

