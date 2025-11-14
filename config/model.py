def get_model_config(method):
    """
    Get model-specific configuration parameters.
    
    Args:
        method: 'cpd_bayesian', 'kalman', or 'cpd'
    """
    if method == 'cpd_bayesian':
        model_config = {
            'method': 'cpd_bayesian',
            'min_history_length': 5,
            'hazard_lambda': 100,
            'mu0': 0.0,
            'kappa0': 1.0,
            'alpha0': 1.0,
            'beta0': 1.0,
            'alarm_threshold': 0.4,
            'alarm_min_consecutive': 1
        }
    elif method == 'kalman':
        model_config = {
            'method': 'kalman',
            'min_history_length': 10,
            'a': 0.3,
            'b': 10,
        }
    elif method == 'cpd':
        model_config = {
            'method': 'cpd',
            'min_history_length': 5,
            'gamma': None,
        }
    else:
        raise ValueError(f"Invalid method: {method}. Use 'cpd_bayesian', 'kalman', or 'cpd'")
    return model_config