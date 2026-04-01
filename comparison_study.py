"""
comparison_study.py — BOCPD alarm-threshold ablation + NPZ PMF validation.

Phase 1 (run_ablation):
  Sweeps alarm_threshold in [0.010 … 0.015] × 4 window strategies on the
  two baseline arrival CSVs (Z0=5 and Z0=80).  Reports RMSE/MAE table and
  returns the single best (strategy, threshold) configuration.

Phase 2 (validate_pmf):
  Uses the best config to run the full pipeline on the Z0=5 arrival CSV,
  computes queue-length PMF at each t, and compares against the 22 Z05
  ground-truth NPZ files via KL divergence.

Usage:
    python comparison_study.py
"""

import os
import glob
import re
import time
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

from src import UnifiedPredictor, PredictionConfig
from src.pmf import transient_distribution_piecewise
from src.utils import calculate_kl_divergence

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR  = 'data_integrated/arrival_data'
NPZ_DIR   = 'Simulation_code'
OUT_DIR   = 'data_integrated/result_data'
os.makedirs(OUT_DIR, exist_ok=True)

# ── Ablation datasets (baseline "initial setting" CSVs only) ──────────────────
ABLATION_DATASETS = [
    dict(csv='initial_value_5_samples_500.csv',  Z0=5,  mu=10,  label='Z0=5,  mu=10'),
    dict(csv='initial_value_80_samples_500.csv', Z0=80, mu=100, label='Z0=80, mu=100'),
]

# ── Window strategies ─────────────────────────────────────────────────────────
WINDOW_STRATEGIES = [
    dict(
        id='fixed_ws5',
        label='Fixed ws=5',
        kwargs=dict(adaptive=False, window_size=5),
    ),
    dict(
        id='fixed_ws10',
        label='Fixed ws=10',
        kwargs=dict(adaptive=False, window_size=10),
    ),
    dict(
        id='adaptive_std',
        label='Adaptive (rolling_std)',
        kwargs=dict(adaptive=True, window_method='rolling_std',
                    adaptive_w_min=3, adaptive_w_max=15,
                    adaptive_threshold=True, threshold_k=2.0),
    ),
    dict(
        id='adaptive_ewma',
        label='Adaptive (EWMA)',
        kwargs=dict(adaptive=True, window_method='ewma',
                    adaptive_w_min=3, adaptive_w_max=15,
                    adaptive_threshold=True, threshold_k=2.0),
    ),
]

ALARM_THRESHOLDS = [0.010, 0.011, 0.012, 0.013, 0.014, 0.015]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_csv(csv_name: str) -> pd.DataFrame:
    path = os.path.join(DATA_DIR, csv_name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found: {path}")
    return pd.read_csv(path)


def _run_bocpd(strategy_kwargs: dict, alarm_threshold: float,
               data: pd.DataFrame) -> tuple:
    """Run BOCPD with given config, return (metrics_dict, predictor)."""
    cfg = PredictionConfig(
        method='cpd_bayesian',
        hazard_lambda=50.0,
        alarm_threshold=alarm_threshold,
        plot=False,
        verbose=False,
        **strategy_kwargs,
    )
    up = UnifiedPredictor(cfg)
    t0 = time.time()
    results = up.predict(data)
    elapsed = time.time() - t0
    metrics = results.get('summary_metrics') or {}
    cp_probs = up.predictor.results.get('changepoint_probs', pd.Series(dtype=float))
    avg_cp = float(cp_probs.mean()) if len(cp_probs) > 0 else 0.0
    row = {
        'rmse':               metrics.get('rmse', np.nan),
        'mae':                metrics.get('mae', np.nan),
        'direction_accuracy': metrics.get('direction_accuracy', np.nan),
        'avg_changepoint_prob': avg_cp,
        'runtime_s':          elapsed,
    }
    return row, up.predictor


def _extract_z_piece(predictor, data: pd.DataFrame):
    """Extract Z_piece and dt_piece from a fitted predictor's results."""
    r = predictor.results
    step_raw = r.get('predicted_step_function', r.get('stepwise_value', []))
    step_vals = pd.Series(step_raw).dropna().values

    if len(step_vals) == 0:
        raise ValueError(
            "_extract_z_piece: predictor produced no step-function values. "
            "Check that rolling_prediction / adaptive_rolling_prediction ran successfully."
        )

    times_raw = r.get('prediction_times', data['time'].values)
    times = times_raw.values if hasattr(times_raw, 'values') else np.asarray(times_raw)

    n = len(step_vals)
    t_slice = times[:n] if len(times) >= n else times
    if len(t_slice) == 0:
        raise ValueError("_extract_z_piece: prediction_times array is empty.")
    dt = np.diff(t_slice, prepend=t_slice[0])
    dt = np.where(dt <= 0, 0.02, dt)
    Z_piece = np.maximum(step_vals, 1e-6)
    return Z_piece, dt


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1: Ablation
# ─────────────────────────────────────────────────────────────────────────────

def run_ablation():
    """
    Sweep alarm_threshold × window strategies on 2 baseline CSVs.
    Returns (df_full, best_config_dict).
    """
    print(f"\n{'='*80}")
    print("PHASE 1: BOCPD Alarm-Threshold Ablation")
    print(f"  Thresholds: {ALARM_THRESHOLDS}")
    print(f"  Strategies: {[s['label'] for s in WINDOW_STRATEGIES]}")
    print(f"  Datasets:   {[d['label'] for d in ABLATION_DATASETS]}")
    print(f"  Total runs: {len(ALARM_THRESHOLDS)*len(WINDOW_STRATEGIES)*len(ABLATION_DATASETS)}")
    print(f"{'='*80}\n")

    rows = []
    for ds in ABLATION_DATASETS:
        data = _load_csv(ds['csv'])
        for thr in ALARM_THRESHOLDS:
            for strat in WINDOW_STRATEGIES:
                try:
                    res, _ = _run_bocpd(strat['kwargs'], thr, data)
                    row = dict(
                        strategy_id=strat['id'],
                        strategy=strat['label'],
                        alarm_threshold=thr,
                        dataset=ds['label'],
                        Z0=ds['Z0'],
                        **res,
                    )
                    rows.append(row)
                    print(f"  {strat['label']:<25} thr={thr:.3f}  {ds['label']:<14} "
                          f"RMSE={res['rmse']:7.3f}  MAE={res['mae']:7.3f}  "
                          f"Dir={res['direction_accuracy']:5.1f}%  "
                          f"avgCP={res['avg_changepoint_prob']:.4f}  "
                          f"t={res['runtime_s']:.1f}s")
                except Exception as e:
                    print(f"  ERROR {strat['id']} thr={thr} {ds['label']}: {e}")
                    rows.append(dict(strategy_id=strat['id'], strategy=strat['label'],
                                     alarm_threshold=thr, dataset=ds['label'], Z0=ds['Z0'],
                                     rmse=np.nan, mae=np.nan, direction_accuracy=np.nan,
                                     avg_changepoint_prob=np.nan, runtime_s=np.nan))

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT_DIR, 'bocpd_ablation_results.csv'), index=False)

    # Aggregate across 2 datasets
    agg = (df.groupby(['strategy_id', 'strategy', 'alarm_threshold'])
             [['rmse', 'mae', 'direction_accuracy', 'avg_changepoint_prob', 'runtime_s']]
             .mean()
             .reset_index()
             .sort_values('rmse'))

    print(f"\n{'='*80}")
    print("AGGREGATE (mean across 2 datasets), sorted by RMSE:")
    print(f"{'='*80}")
    print(f"{'Rank':<5} {'Strategy':<26} {'Thr':>6} {'RMSE':>8} {'MAE':>8} "
          f"{'Dir%':>6} {'avgCP':>7} {'Time':>6}")
    print("-" * 80)
    for rank, (_, r) in enumerate(agg.iterrows(), 1):
        print(f"  {rank:<4} {r['strategy']:<26} {r['alarm_threshold']:.3f} "
              f"{r['rmse']:8.3f} {r['mae']:8.3f} "
              f"{r['direction_accuracy']:6.1f} {r['avg_changepoint_prob']:7.4f} "
              f"{r['runtime_s']:6.2f}s")

    best_row = agg.iloc[0]
    # Reconstruct kwargs for best strategy
    best_strat = next(s for s in WINDOW_STRATEGIES if s['id'] == best_row['strategy_id'])
    best_config = {
        'strategy_id':      best_row['strategy_id'],
        'strategy_label':   best_row['strategy'],
        'alarm_threshold':  best_row['alarm_threshold'],
        'strategy_kwargs':  best_strat['kwargs'],
    }

    print(f"\n>>> BEST: strategy='{best_config['strategy_label']}', "
          f"alarm_threshold={best_config['alarm_threshold']:.3f}, "
          f"RMSE={best_row['rmse']:.3f}, MAE={best_row['mae']:.3f}\n")

    return df, best_config


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2: PMF Validation on NPZ files
# ─────────────────────────────────────────────────────────────────────────────

def validate_pmf(best_strategy_kwargs: dict, best_threshold: float, N: int = 200):
    """
    Run best BOCPD config on Z0=5 arrival CSV, compute PMF at each t,
    compare with NPZ ground-truth PMFs via KL divergence.
    Covers 22 Z05 NPZ files; skips Z020 (no matching arrival CSV).
    """
    print(f"\n{'='*80}")
    print("PHASE 2: PMF Validation on NPZ Ground-Truth Files")
    print(f"  Best config: {best_strategy_kwargs}")
    print(f"  alarm_threshold: {best_threshold:.3f}")
    print(f"{'='*80}\n")

    # Load arrival CSV and run BOCPD once
    csv_z5 = 'initial_value_5_samples_500.csv'
    data_z5 = _load_csv(csv_z5)
    print(f"Running BOCPD on {csv_z5} ...")
    _, predictor_z5 = _run_bocpd(best_strategy_kwargs, best_threshold, data_z5)
    Z_piece, dt_piece = _extract_z_piece(predictor_z5, data_z5)
    T_total = dt_piece.sum()
    print(f"  Z_piece: {len(Z_piece)} segments, "
          f"range [{Z_piece.min():.2f}, {Z_piece.max():.2f}], "
          f"total T={T_total:.2f}\n")

    # Find all Z05 NPZ files
    npz_pattern = os.path.join(NPZ_DIR, 'pmf_CoxM1_Z05_serv*_T*_t*_v2.npz')
    npz_files = sorted(glob.glob(npz_pattern))
    print(f"Found {len(npz_files)} Z05 NPZ files to validate against.\n")

    rows = []
    print(f"{'NPZ file':<45} {'serv':>5} {'T':>4} {'t':>4} {'KL':>10}")
    print("-" * 75)

    for npz_path in npz_files:
        fname = os.path.basename(npz_path)
        # Parse: pmf_CoxM1_Z05_serv{serv}_T{T}_t{t}_v2.npz
        m = re.match(r'pmf_CoxM1_Z0(\d+)_serv(\d+)_T(\d+)_t(\d+)_v2\.npz', fname)
        if not m:
            print(f"  SKIP (unrecognised name): {fname}")
            continue
        z0_npz = int(m.group(1))
        serv   = int(m.group(2))
        T_npz  = int(m.group(3))
        t_npz  = int(m.group(4))

        try:
            npz_data = np.load(npz_path)
            mu_npz   = float(npz_data['service_rate'])   # always use NPZ's own service_rate
            t_val    = float(npz_data['t'])
            npz_pmf  = npz_data['pmf']

            # Clamp t to the predicted horizon; warn if clamping fires
            t_target = min(t_val, T_total * 0.99)
            if t_target < t_val:
                print(f"    WARN: t={t_val} > T_total*0.99={T_total*0.99:.2f}; "
                      f"clamping to {t_target:.2f}")

            pmf_pred = transient_distribution_piecewise(
                Z_piece, dt_piece, mu=mu_npz, m=1, t=t_target, N=N
            )

            # Align lengths
            min_len = min(len(npz_pmf), len(pmf_pred))
            kl = calculate_kl_divergence(npz_pmf[:min_len], pmf_pred[:min_len])

            rows.append(dict(
                npz_file=fname, Z0=z0_npz, service_rate=serv,
                T=T_npz, t=t_npz, kl_divergence=kl,
                strategy=best_strategy_kwargs,
                alarm_threshold=best_threshold,
            ))
            print(f"  {fname:<43} {serv:>5} {T_npz:>4} {t_npz:>4} {kl:>10.6f}")

        except Exception as e:
            print(f"  ERROR {fname}: {e}")
            rows.append(dict(npz_file=fname, Z0=z0_npz, service_rate=serv,
                              T=T_npz, t=t_npz, kl_divergence=np.nan,
                              strategy=best_strategy_kwargs,
                              alarm_threshold=best_threshold))

    df = pd.DataFrame(rows)
    out_path = os.path.join(OUT_DIR, 'pmf_validation_results.csv')
    df.to_csv(out_path, index=False)
    print(f"\nResults saved → {out_path}")

    if len(df) > 0 and not df['kl_divergence'].isna().all():
        print("\n--- Mean KL divergence by (service_rate, t) ---")
        summary = (df.dropna(subset=['kl_divergence'])
                     .groupby(['service_rate', 't'])['kl_divergence']
                     .mean()
                     .reset_index()
                     .sort_values(['service_rate', 't']))
        for _, r in summary.iterrows():
            print(f"  serv={int(r['service_rate']):<5} t={int(r['t']):<4}  mean_KL={r['kl_divergence']:.6f}")

        overall = df['kl_divergence'].mean()
        best_npz = df.loc[df['kl_divergence'].idxmin()]
        print(f"\n  Overall mean KL: {overall:.6f}")
        print(f"  Best  NPZ file:  {best_npz['npz_file']} (KL={best_npz['kl_divergence']:.6f})")

    # Note about skipped Z020 files
    z020_files = sorted(glob.glob(os.path.join(NPZ_DIR, 'pmf_CoxM1_Z020_*.npz')))
    if z020_files:
        print(f"\n  NOTE: {len(z020_files)} Z020 NPZ files skipped "
              f"(no matching arrival CSV for Z0=20). "
              "Generate initial_value_20_samples_500_T10.csv to enable.")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    df_ablation, best_config = run_ablation()
    df_pmf = validate_pmf(best_config['strategy_kwargs'], best_config['alarm_threshold'])
    print("\nDone.")
