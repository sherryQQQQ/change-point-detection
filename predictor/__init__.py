

from .karman_filter import RollingKalmanPredictor
from .bayesian_online import BOOLES_cpdPredictor
from .kernel import OLScpdPredictor
from .base import RollingPredictor

def get_predictor(name, **kwargs):
    if name == "rolling_kalman":
        return RollingKalmanPredictor(**kwargs)
    elif name == "cpd_bayesian":
        return BOOLES_cpdPredictor(**kwargs)
    elif name == "kernel":
        return OLScpdPredictor(**kwargs)
    else:
        raise ValueError(f"Unknown predictor: {name}")
    