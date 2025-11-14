"""Config loader for different scenarios and methods"""


def load_config(scenario, method):
    """
    Load configuration for a specific scenario and method.
    
    Args:
        scenario: Scenario name, e.g., 'z5t1', 'z5t5', 'z80t1', 'z80t5'
        method: Method name, e.g., 'bayesian', 'rolling_kalman', 'kernel'
    
    Returns:
        dict: Configuration dictionary
    
    Example:
        >>> config = load_config('z5t1', 'rolling_kalman')
        >>> print(config['z_initial'])
        5
    """
    if scenario == 'z5t1':
        from config.transient.z5t1 import get_config
    elif scenario == 'z5t5':
        from config.transient.z5t5 import get_config
    elif scenario == 'z80t1':
        from config.transient.z80t1 import get_config
    elif scenario == 'z80t5':
        from config.transient.z80t5 import get_config
    elif scenario == 'trial':
        from config.trial import get_config
    else:
        raise ValueError(f"Unknown scenario: {scenario}. Available: z5t1, z5t5, z80t1, z80t5")
    
    return get_config(method)

