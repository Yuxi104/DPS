import yaml
from types import SimpleNamespace


def dict_to_namespace(d):
    if isinstance(d, dict):
        return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
    elif isinstance(d, list):
        return [dict_to_namespace(x) for x in d]
    else:
        return d

def setup(path):
    with open(path, 'r') as f:
        config_dict = yaml.safe_load(f)
        config = dict_to_namespace(config_dict)
    return config