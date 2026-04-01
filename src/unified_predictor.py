"""
Unified Prediction Interface — v2

Provides PredictionConfig (with YAML support) and UnifiedPredictor,
a single entry point for bayesian / kalman / cpd / cpd_bayesian methods.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Union

from .config_templates import compute_window_size
from .bayesian_online_predictor import BayesianOnlinePredictor


class PredictionConfig:
    """Configuration for all prediction methods."""

    _SENTINEL = object()

    def __init__(self,
                 method: str = 'bayesian',
                 min_history_length: int = 5,
                 window_size: int = _SENTINEL,
                 verbose: bool = True,
                 plot: bool = True,
                 # Bayesian parameters
                 hazard_lambda: float = 50.0,
                 mu0: float = 80.0,
                 kappa0: float = 0.1,
                 alpha0: float = 1.0,
                 beta0: float = 1.0,
                 alarm_threshold: float = 0.1,
                 alarm_min_consecutive: int = 1,
                 # Kalman parameters
                 kalman_a: float = 0.3,
                 kalman_b: float = 80.0,
                 # CPD parameters
                 cpd_gamma: Optional[float] = None,
                 # Service rate (used to auto-compute window_size)
                 mu: float = 100.0,
                 # Adaptive window / threshold (cpd_bayesian only)
                 adaptive: bool = False,
                 adaptive_w_min: int = 3,
                 adaptive_w_max: int = 15,
                 adaptive_threshold: bool = True,
                 threshold_k: float = 2.0,
                 window_method: str = 'rolling_std'):
        self.method = method
        self.min_history_length = min_history_length
        self.verbose = verbose
        self.plot = plot
        self.hazard_lambda = hazard_lambda
        self.mu0 = mu0
        self.kappa0 = kappa0
        self.alpha0 = alpha0
        self.beta0 = beta0
        self.alarm_threshold = alarm_threshold
        self.alarm_min_consecutive = alarm_min_consecutive
        self.kalman_a = kalman_a
        self.kalman_b = kalman_b
        self.cpd_gamma = cpd_gamma
        self.mu = mu
        self.adaptive = adaptive
        self.adaptive_w_min = adaptive_w_min
        self.adaptive_w_max = adaptive_w_max
        self.adaptive_threshold = adaptive_threshold
        self.threshold_k = threshold_k
        self.window_method = window_method

        if window_size is PredictionConfig._SENTINEL:
            self.window_size = compute_window_size(mu)
        else:
            self.window_size = window_size

    def to_dict(self) -> Dict[str, Any]:
        return {
            'method': self.method,
            'min_history_length': self.min_history_length,
            'window_size': self.window_size,
            'verbose': self.verbose,
            'plot': self.plot,
            'hazard_lambda': self.hazard_lambda,
            'mu0': self.mu0,
            'kappa0': self.kappa0,
            'alpha0': self.alpha0,
            'beta0': self.beta0,
            'alarm_threshold': self.alarm_threshold,
            'alarm_min_consecutive': self.alarm_min_consecutive,
            'kalman_a': self.kalman_a,
            'kalman_b': self.kalman_b,
            'cpd_gamma': self.cpd_gamma,
            'mu': self.mu,
            'adaptive': self.adaptive,
            'adaptive_w_min': self.adaptive_w_min,
            'adaptive_w_max': self.adaptive_w_max,
            'adaptive_threshold': self.adaptive_threshold,
            'threshold_k': self.threshold_k,
            'window_method': self.window_method,
        }

    def to_yaml(self, path: str) -> None:
        import yaml
        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)

    @classmethod
    def from_yaml(cls, path: str) -> 'PredictionConfig':
        import yaml
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)


class UnifiedPredictor:
    """Single entry point for all prediction methods."""

    def __init__(self, config: Optional[PredictionConfig] = None):
        self.config = config or PredictionConfig()
        self.predictor = None
        self.results = None

    def _create_predictor(self):
        cfg = self.config
        if cfg.method == 'bayesian':
            return BayesianOnlinePredictor(
                hazard_lambda=cfg.hazard_lambda,
                mu0=cfg.mu0,
                kappa0=cfg.kappa0,
                alpha0=cfg.alpha0,
                beta0=cfg.beta0,
                alarm_threshold=cfg.alarm_threshold,
            )
        elif cfg.method == 'kalman':
            from .kalman_predictor import RollingKalmanPredictor
            return RollingKalmanPredictor(
                min_history_length=cfg.min_history_length,
                kalman_a=cfg.kalman_a,
                kalman_b=cfg.kalman_b,
            )
        elif cfg.method == 'cpd':
            from .cpd_predictor import OLScpdPredictor
            return OLScpdPredictor(min_history_length=cfg.min_history_length)
        elif cfg.method == 'cpd_bayesian':
            from .booles_cpd_predictor import BOOLES_cpdPredictor
            return BOOLES_cpdPredictor(
                min_history_length=cfg.min_history_length,
                hazard_lambda=cfg.hazard_lambda,
                mu0=cfg.mu0,
                kappa0=cfg.kappa0,
                alpha0=cfg.alpha0,
                beta0=cfg.beta0,
                alarm_threshold=cfg.alarm_threshold,
                alarm_min_consecutive=cfg.alarm_min_consecutive,
            )
        else:
            raise ValueError(f"Unknown prediction method: {cfg.method}")

    def predict(self, data: Union[pd.DataFrame, np.ndarray],
                config: Optional[PredictionConfig] = None) -> Dict[str, Any]:
        if config is not None:
            self.config = config

        self.predictor = self._create_predictor()

        if isinstance(data, np.ndarray):
            data = pd.DataFrame({'value': data, 'time': np.arange(len(data))})
        elif isinstance(data, pd.DataFrame) and 'time' not in data.columns:
            data = data.copy()
            data['time'] = np.arange(len(data))

        cfg = self.config
        if cfg.method == 'bayesian':
            metrics = self.predictor.rolling_prediction(
                data,
                window_size=cfg.window_size,
                plot=cfg.plot,
                verbose=cfg.verbose,
            )
        elif cfg.method == 'kalman':
            metrics = self.predictor.rolling_prediction(
                data['time'].values,
                data['value'].values,
                verbose=cfg.verbose,
            )
        elif cfg.method == 'cpd':
            metrics = self.predictor.rolling_prediction(
                data,
                window_size=cfg.window_size,
                gamma=cfg.cpd_gamma,
                plot=cfg.plot,
                method='ols',
                file_name=None,
            )
        elif cfg.method == 'cpd_bayesian':
            if cfg.adaptive:
                metrics = self.predictor.adaptive_rolling_prediction(
                    data,
                    w_min=cfg.adaptive_w_min,
                    w_max=cfg.adaptive_w_max,
                    adaptive_threshold=cfg.adaptive_threshold,
                    threshold_base=cfg.alarm_threshold,
                    threshold_k=cfg.threshold_k,
                    window_method=cfg.window_method,
                    plot=cfg.plot,
                    method='ols',
                )
            else:
                metrics = self.predictor.rolling_prediction(
                    data,
                    window_size=cfg.window_size,
                    plot=cfg.plot,
                    method='ols',
                    file_name=None,
                    cpd_method='bayesian',
                )

        self.results = {
            'predictor': self.predictor,
            'summary_metrics': metrics,
            'detailed_results': (self.predictor.results
                                 if hasattr(self.predictor, 'results') else {}),
            'config': cfg.to_dict(),
        }

        if cfg.verbose and metrics:
            self._print_summary(metrics)

        return self.results

    def _print_summary(self, metrics: Dict[str, float]):
        print("\n" + "=" * 60)
        print(f"{self.config.method.upper()} Prediction Performance Summary")
        print("=" * 60)
        print(f"RMSE:               {metrics.get('rmse', 0):.4f}")
        print(f"MAE:                {metrics.get('mae', 0):.4f}")
        print(f"MAPE:               {metrics.get('mape', 0):.2f}%")
        print(f"Direction accuracy: {metrics.get('direction_accuracy', 0):.1f}%")
        cc = metrics.get('confidence_coverage', None)
        if cc is not None:
            print(f"CI coverage:        {cc:.1f}%")
        print(f"Predictions:        {metrics.get('n_predictions', 0)}")
        avg_cp = metrics.get('avg_changepoint_prob', None)
        if avg_cp is not None:
            print(f"Avg CP prob:        {avg_cp:.4f}")
            print(f"Max CP prob:        {metrics.get('max_changepoint_prob', 0):.4f}")
        print("=" * 60)

    def get_summary_metrics(self) -> Dict[str, float]:
        return self.results.get('summary_metrics', {}) if self.results else {}

    def plot_results(self, **kwargs):
        if self.predictor is None:
            print("No prediction results. Run predict() first.")
            return
        if hasattr(self.predictor, 'plot_rolling_predictions'):
            self.predictor.plot_rolling_predictions(**kwargs)

    def save_results(self, filepath: str):
        if self.results is None:
            print("No results to save. Run predict() first.")
            return
        dr = self.results.get('detailed_results', {})
        if dr:
            pd.DataFrame(dr).to_csv(filepath, index=False)
            print(f"Results saved to {filepath}")


def rolling_predictor(times, values, method='bayesian', min_history_length=5,
                      window_size=3, verbose=True, **kwargs) -> Dict[str, Any]:
    """Convenience function for rolling prediction."""
    config = PredictionConfig(
        method=method,
        min_history_length=min_history_length,
        window_size=window_size,
        verbose=verbose,
        **kwargs,
    )
    data = pd.DataFrame({'time': np.asarray(times), 'value': np.asarray(values)})
    return UnifiedPredictor(config).predict(data)
