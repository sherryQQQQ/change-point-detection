from config.model import get_model_config


def get_config(method):
    config = {
        'z_initial': 80,
        'min_history_length': 10,
        'ws': 45,
        'method': method,
        't': 5,
        'model_config': get_model_config(method)
    }
    return config

