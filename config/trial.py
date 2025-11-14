


from config.model import get_model_config

def get_config(method):
    config = {
        'z_initial': 5,
        'min_history_length': 10,
        'ws': 20,
        'method': method,
        't': 7,
        'model_config': get_model_config(method)
    }
    return config