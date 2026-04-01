"""
Configuration Templates for Prediction Methods

Provides predefined configuration presets and the compute_window_size helper.
"""

from __future__ import annotations


def compute_window_size(mu: float, c: float = 0.05, min_ws: int = 5) -> int:
    """Compute window size from service rate: ws = max(min_ws, int(c * mu))."""
    return max(min_ws, int(c * mu))


# Lazy import to avoid circular dependency at module level
def _get_prediction_config():
    from .unified_predictor import PredictionConfig
    return PredictionConfig


class ConfigTemplates:
    """Predefined configuration templates for different scenarios."""

    @staticmethod
    def default_bayesian():
        PC = _get_prediction_config()
        return PC(method='bayesian', hazard_lambda=100.0, mu0=0.0, kappa0=1.0,
                  alpha0=1.0, beta0=1.0, alarm_threshold=0.7, verbose=True, plot=True)

    @staticmethod
    def sensitive_change_detection():
        PC = _get_prediction_config()
        return PC(method='bayesian', hazard_lambda=50.0, mu0=0.0, kappa0=0.5,
                  alpha0=0.5, beta0=0.5, alarm_threshold=0.1, verbose=True, plot=True)

    @staticmethod
    def robust_prediction():
        PC = _get_prediction_config()
        return PC(method='bayesian', hazard_lambda=200.0, mu0=0.0, kappa0=2.0,
                  alpha0=2.0, beta0=2.0, alarm_threshold=0.3, alarm_min_consecutive=2,
                  verbose=True, plot=True)

    @staticmethod
    def high_frequency_data():
        PC = _get_prediction_config()
        return PC(method='bayesian', hazard_lambda=20.0, mu0=0.0, kappa0=0.1,
                  alpha0=0.1, beta0=0.1, alarm_threshold=0.05, verbose=True, plot=True)

    @staticmethod
    def queue_length_prediction():
        PC = _get_prediction_config()
        return PC(method='bayesian', hazard_lambda=80.0, mu0=10.0, kappa0=1.0,
                  alpha0=1.0, beta0=1.0, alarm_threshold=0.15, verbose=True, plot=True)

    @staticmethod
    def arrival_rate_prediction():
        PC = _get_prediction_config()
        return PC(method='bayesian', hazard_lambda=60.0, mu0=5.0, kappa0=0.5,
                  alpha0=0.5, beta0=0.5, alarm_threshold=0.2, verbose=True, plot=True)

    @staticmethod
    def cpd_bayesian_default():
        """BOCPD + OLS — the recommended method for Cox/M/1 arrival rate prediction."""
        PC = _get_prediction_config()
        return PC(method='cpd_bayesian', hazard_lambda=50.0, mu0=80.0, kappa0=0.1,
                  alpha0=1.0, beta0=1.0, alarm_threshold=0.1, mu=100.0,
                  verbose=True, plot=False)


def get_config(scenario: str = 'default'):
    """
    Get configuration for a specific scenario.

    Available: 'default', 'sensitive', 'robust', 'high_freq', 'queue',
               'arrival', 'cpd_bayesian'
    """
    templates = {
        'default': ConfigTemplates.default_bayesian,
        'sensitive': ConfigTemplates.sensitive_change_detection,
        'robust': ConfigTemplates.robust_prediction,
        'high_freq': ConfigTemplates.high_frequency_data,
        'queue': ConfigTemplates.queue_length_prediction,
        'arrival': ConfigTemplates.arrival_rate_prediction,
        'cpd_bayesian': ConfigTemplates.cpd_bayesian_default,
    }

    if scenario not in templates:
        raise ValueError(f"Unknown scenario: {scenario}. Available: {list(templates.keys())}")

    return templates[scenario]()


def create_custom_config(**kwargs):
    """Create custom configuration with overrides on top of default."""
    config = ConfigTemplates.default_bayesian()
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            print(f"Warning: Unknown parameter '{key}' ignored")
    return config
