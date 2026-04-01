"""
Benchmark runner — evaluate multiple prediction methods on multiple datasets.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Optional


def benchmark(
    methods: List[str],
    datasets: List[pd.DataFrame],
    dataset_names: Optional[List[str]] = None,
    config_overrides: Optional[Dict] = None,
    plot: bool = False,
    pmf_fn=None,
    mu: float = 100.0,
    m: int = 1,
    t: float = 1.0,
    hist_data: Optional[Dict[str, np.ndarray]] = None,
) -> pd.DataFrame:
    """
    Run all methods on all datasets and return a tidy results DataFrame.

    Parameters
    ----------
    methods : list of str
        Prediction method names, e.g. ['bayesian', 'kalman', 'cpd'].
    datasets : list of pd.DataFrame
        Each DataFrame must have 'time' and 'value' columns.
    dataset_names : list of str, optional
        Names for each dataset (used in output table).
    config_overrides : dict, optional
        Extra kwargs forwarded to PredictionConfig for every method.
    plot : bool
        Whether to show prediction plots during the run.
    pmf_fn : callable, optional
        If provided, also compute PMF and KL divergence for each regime.
        Signature: pmf_fn(Z_piece, dt_piece, mu, m, t, ...) -> np.ndarray
    mu, m, t : float/int
        Queue model parameters (only used when pmf_fn is not None).
    hist_data : dict of str -> np.ndarray, optional
        Simulation histograms keyed by dataset name; used for KL divergence
        computation when pmf_fn is provided.

    Returns
    -------
    pd.DataFrame with columns:
        method, dataset, rmse, mae, mape, direction_accuracy,
        confidence_coverage, n_predictions,
        avg_cp_prob, max_cp_prob, n_cp_detected,
        kl_mean, kl_max  (only when pmf_fn is given)
    """
    from .unified_predictor import UnifiedPredictor, PredictionConfig
    from .prediction_utils import calculate_kl_divergence

    config_overrides = config_overrides or {}
    if dataset_names is None:
        dataset_names = [f'dataset_{i}' for i in range(len(datasets))]

    rows = []

    for method in methods:
        for ds_name, data in zip(dataset_names, datasets):
            print(f"  [{method}] on {ds_name} ...", end=' ', flush=True)

            cfg = PredictionConfig(
                method=method,
                plot=plot,
                verbose=False,
                **config_overrides,
            )

            try:
                up = UnifiedPredictor(cfg)
                results = up.predict(data)
                metrics = results.get('summary_metrics') or {}
                inner = results.get('predictor')

                cp_probs = np.array([])
                if inner is not None and hasattr(inner, 'results'):
                    r = inner.results
                    for key in ('changepoint_prob', 'changepoint_probs'):
                        if key in r and r[key] is not None:
                            cp_probs = np.asarray(r[key]).flatten()
                            cp_probs = cp_probs[~np.isnan(cp_probs)]
                            break

                avg_cp = float(np.mean(cp_probs)) if len(cp_probs) else 0.0
                max_cp = float(np.max(cp_probs)) if len(cp_probs) else 0.0
                n_cp = int(np.sum(cp_probs > 0.05)) if len(cp_probs) else 0

                row = {
                    'method': method,
                    'dataset': ds_name,
                    'rmse': metrics.get('rmse', np.nan),
                    'mae': metrics.get('mae', np.nan),
                    'mape': metrics.get('mape', np.nan),
                    'direction_accuracy': metrics.get('direction_accuracy', np.nan),
                    'confidence_coverage': metrics.get('confidence_coverage', np.nan),
                    'n_predictions': metrics.get('n_predictions', 0),
                    'avg_cp_prob': avg_cp,
                    'max_cp_prob': max_cp,
                    'n_cp_detected': n_cp,
                }

                if pmf_fn is not None:
                    try:
                        cp_indices = list(np.where(cp_probs > cfg.alarm_threshold)[0]) if len(cp_probs) else []
                        boundaries = [0] + cp_indices + [len(data)]
                        segments = [data.iloc[s:e] for s, e in zip(boundaries[:-1], boundaries[1:]) if e > s]

                        kl_vals = []
                        for seg in segments:
                            vals = seg['value'].values
                            times = seg['time'].values if 'time' in seg.columns else np.arange(len(seg))
                            dt_piece = np.diff(times, prepend=times[0])
                            dt_piece[0] = times[1] - times[0] if len(times) > 1 else 1.0
                            Z_piece = np.maximum(vals, 1e-6)
                            t_seg = min(t, dt_piece.sum())
                            pmf_pred = pmf_fn(Z_piece, dt_piece, mu, m, t=t_seg)

                            hist = hist_data.get(ds_name) if hist_data else None
                            if hist is not None and pmf_pred is not None and len(pmf_pred):
                                kl = calculate_kl_divergence(hist, pmf_pred)
                                kl_vals.append(kl)

                        row['kl_mean'] = float(np.mean(kl_vals)) if kl_vals else np.nan
                        row['kl_max'] = float(np.max(kl_vals)) if kl_vals else np.nan
                    except Exception as e:
                        print(f"(PMF/KL failed: {e})", end=' ')
                        row['kl_mean'] = np.nan
                        row['kl_max'] = np.nan

                rows.append(row)
                print("ok")

            except Exception as e:
                print(f"FAILED: {e}")
                rows.append({'method': method, 'dataset': ds_name, 'error': str(e)})

    return pd.DataFrame(rows)


def plot_benchmark_results(df: pd.DataFrame, metrics: Optional[List[str]] = None):
    """
    Bar chart comparing methods across datasets for each metric.

    Parameters
    ----------
    df : pd.DataFrame  — output of benchmark()
    metrics : list of str, optional — which columns to plot
    """
    if metrics is None:
        metrics = ['rmse', 'mae', 'direction_accuracy']

    n = len(metrics)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes = [axes]

    methods = df['method'].unique()
    datasets = df['dataset'].unique()
    x = np.arange(len(datasets))
    width = 0.8 / len(methods)

    for ax, metric in zip(axes, metrics):
        if metric not in df.columns:
            continue
        for i, method in enumerate(methods):
            sub = df[df['method'] == method]
            vals = [sub[sub['dataset'] == ds][metric].values[0]
                    if len(sub[sub['dataset'] == ds]) else np.nan
                    for ds in datasets]
            ax.bar(x + i * width, vals, width, label=method)

        ax.set_xticks(x + width * (len(methods) - 1) / 2)
        ax.set_xticklabels(datasets, rotation=30, ha='right')
        ax.set_title(metric)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.show()
