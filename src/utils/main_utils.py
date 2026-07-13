import yaml


def load_yaml(filepath: str) -> dict:
    with open(filepath, "rb") as f:
        return yaml.safe_load(f)
