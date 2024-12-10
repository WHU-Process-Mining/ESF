import yaml

def load_config_data(path: str) -> dict:
    """Load a config data from a given path

    :param path: the path as a string
    :return: the config as a dict
    """
    with open(path) as f:
        cfg: dict = yaml.load(f, Loader=yaml.FullLoader)
    return cfg


def save_config_data(data: dict, path: str) -> None:
    """Save the config on the disk

    :param data: the config as a dict
    :param path: the path as a string
    """
    with open(path, "w") as f:
        yaml.dump(data, f)