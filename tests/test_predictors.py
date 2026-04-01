"""
Unit tests for all prediction methods.
"""

import numpy as np
import pandas as pd
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import (
    UnifiedPredictor,
    PredictionConfig,
    BayesianOnlinePredictor,
    RollingKalmanPredictor,
    OLScpdPredictor,
    BOOLES_cpdPredictor,
    get_config,
    compute_window_size,
)


# ── Fixtures ──────────────────────────────────────────────────────────

def _make_synthetic_data(n=200, seed=42):
    """Synthetic arrival-rate time series with one change point at midpoint."""
    rng = np.random.RandomState(seed)
    times = np.linspace(0, 10, n)
    mid = n // 2
    values = np.empty(n)
    values[:mid] = 80 + rng.normal(0, 2, mid)
    values[mid:] = 120 + rng.normal(0, 3, n - mid)
    return pd.DataFrame({'time': times, 'value': values})


# ── compute_window_size ───────────────────────────────────────────────

def test_compute_window_size():
    assert compute_window_size(100) == 5
    assert compute_window_size(200) == 10
    assert compute_window_size(10) == 5   # min_ws clamp


# ── PredictionConfig ─────────────────────────────────────────────────

def test_config_defaults():
    cfg = PredictionConfig()
    assert cfg.method == 'bayesian'
    assert cfg.window_size == compute_window_size(cfg.mu)


def test_config_explicit_window_size():
    cfg = PredictionConfig(window_size=7)
    assert cfg.window_size == 7


def test_config_to_dict_roundtrip():
    cfg = PredictionConfig(method='cpd', mu=50)
    d = cfg.to_dict()
    assert d['method'] == 'cpd'
    assert d['mu'] == 50


# ── BayesianOnlinePredictor ──────────────────────────────────────────

def test_bayesian_update():
    bp = BayesianOnlinePredictor(hazard_lambda=50, mu0=80.0)
    probs = [bp.update(80 + np.random.randn()) for _ in range(20)]
    assert len(probs) == 20
    assert all(0.0 <= p <= 1.0 for p in probs)


def test_bayesian_predict():
    bp = BayesianOnlinePredictor(mu0=80.0)
    for v in [80, 81, 79, 80, 82]:
        bp.update(v)
    pred = bp.predict()
    assert 'mean' in pred
    assert 'variance' in pred
    assert pred['variance'] > 0


def test_bayesian_rolling():
    data = _make_synthetic_data(100)
    cfg = PredictionConfig(method='bayesian', hazard_lambda=30,
                           alarm_threshold=0.1, plot=False, verbose=False)
    up = UnifiedPredictor(cfg)
    results = up.predict(data)
    metrics = results['summary_metrics']
    assert metrics is not None
    assert 'rmse' in metrics
    assert metrics['n_predictions'] > 0


# ── RollingKalmanPredictor ───────────────────────────────────────────

def test_kalman_rolling():
    data = _make_synthetic_data(100)
    cfg = PredictionConfig(method='kalman', kalman_a=0.3, kalman_b=80.0,
                           plot=False, verbose=False)
    up = UnifiedPredictor(cfg)
    results = up.predict(data)
    metrics = results['summary_metrics']
    assert metrics is not None
    assert metrics['rmse'] > 0
    assert metrics['n_predictions'] > 0


# ── OLScpdPredictor ─────────────────────────────────────────────────

def test_cpd_rolling():
    data = _make_synthetic_data(100)
    cfg = PredictionConfig(method='cpd', window_size=5, plot=False, verbose=False)
    up = UnifiedPredictor(cfg)
    results = up.predict(data)
    metrics = results['summary_metrics']
    assert metrics is not None
    assert metrics['n_predictions'] > 0


# ── BOOLES_cpdPredictor ─────────────────────────────────────────────

def test_cpd_bayesian_rolling():
    data = _make_synthetic_data(100)
    cfg = PredictionConfig(method='cpd_bayesian', window_size=5,
                           hazard_lambda=30, alarm_threshold=0.1,
                           plot=False, verbose=False)
    up = UnifiedPredictor(cfg)
    results = up.predict(data)
    metrics = results['summary_metrics']
    assert metrics is not None
    assert metrics['n_predictions'] > 0


def test_cpd_bayesian_adaptive():
    data = _make_synthetic_data(200)
    predictor = BOOLES_cpdPredictor(
        hazard_lambda=30, mu0=80.0, alarm_threshold=0.01,
    )
    metrics = predictor.adaptive_rolling_prediction(
        data, w_min=3, w_max=10, plot=False,
        adaptive_threshold=True, threshold_base=0.01, threshold_k=2.0,
    )
    assert metrics is not None
    assert len(predictor.window_sizes_used) > 0
    assert len(predictor.thresholds_used) > 0


def test_cpd_bayesian_adaptive_ewma():
    data = _make_synthetic_data(200)
    predictor = BOOLES_cpdPredictor(
        hazard_lambda=30, mu0=80.0, alarm_threshold=0.01,
    )
    metrics = predictor.adaptive_rolling_prediction(
        data, w_min=3, w_max=10, plot=False,
        window_method='ewma',
    )
    assert metrics is not None
    assert len(predictor.window_sizes_used) > 0


# ── get_config presets ───────────────────────────────────────────────

@pytest.mark.parametrize("scenario", [
    'default', 'sensitive', 'robust', 'high_freq', 'queue', 'arrival', 'cpd_bayesian'
])
def test_get_config(scenario):
    cfg = get_config(scenario)
    assert isinstance(cfg, PredictionConfig)


def test_get_config_invalid():
    with pytest.raises(ValueError):
        get_config('nonexistent')


# ── Adaptive window size functions ───────────────────────────────────

def test_adaptive_window_functions():
    from src.change_point_detection import _adaptive_window_size, _adaptive_window_size_ewma
    v = np.random.randn(50).cumsum() + 80
    ws1 = _adaptive_window_size(v, w_min=3, w_max=15)
    ws2 = _adaptive_window_size_ewma(v, w_min=3, w_max=15)
    assert 3 <= ws1 <= 15
    assert 3 <= ws2 <= 15


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
