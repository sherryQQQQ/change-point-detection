"""
Change Point Detection Package (v2)

Prediction methods for Cox/M/1 arrival intensity with PMF computation.

Main Components:
- BayesianOnlinePredictor: Online BOCPD with Student-t (standalone)
- BOOLES_cpdPredictor: BOCPD + OLS change point detection predictor
- OLScpdPredictor: MMD-based OLS change point detection predictor
- RollingKalmanPredictor: Kalman filter predictor (OU state-space model)
- UnifiedPredictor: Unified interface for all prediction methods
- PredictionConfig: Configuration management
- RegimeModel: CP detection → segment fitting → PMF pipeline
"""

__version__ = "2.0.0"

# Core predictors
from .bayesian_online_predictor import BayesianOnlinePredictor
from .kalman_predictor import RollingKalmanPredictor
from .cpd_predictor import OLScpdPredictor
from .booles_cpd_predictor import BOOLES_cpdPredictor

# Unified interface
from .unified_predictor import UnifiedPredictor, PredictionConfig, rolling_predictor
from .config_templates import get_config, compute_window_size, ConfigTemplates

# PMF computation
from .pmf import (
    _build_P,
    build_P_list,
    transient_distribution_piecewise,
    transient_distribution_uniformization,
)

# Pipeline & benchmark
from .pipeline import RegimeModel
from .benchmark import benchmark, plot_benchmark_results

# Histogram loader
from .histogram_loader import load_histogram

# MMD-based CPD (legacy/standalone)
from .change_point_detection import mmd_statistic, prediction_deviation_analysis

# Utilities
from .utils import (
    calculate_prediction_metrics,
    calculate_kl_divergence,
    compare_pmfs_kl,
    plot_pmf_overlap,
    create_prediction_report,
    load_prediction_data,
    save_prediction_data,
)

__all__ = [
    # Predictors
    'BayesianOnlinePredictor',
    'RollingKalmanPredictor',
    'OLScpdPredictor',
    'BOOLES_cpdPredictor',
    # Unified
    'UnifiedPredictor',
    'PredictionConfig',
    'rolling_predictor',
    'get_config',
    'compute_window_size',
    'ConfigTemplates',
    # PMF
    '_build_P',
    'build_P_list',
    'transient_distribution_piecewise',
    'transient_distribution_uniformization',
    # Pipeline
    'RegimeModel',
    'benchmark',
    'plot_benchmark_results',
    # Histogram
    'load_histogram',
    # Legacy CPD
    'mmd_statistic',
    'prediction_deviation_analysis',
    # Utils
    'calculate_prediction_metrics',
    'calculate_kl_divergence',
    'compare_pmfs_kl',
    'plot_pmf_overlap',
    'create_prediction_report',
    'load_prediction_data',
    'save_prediction_data',
]
