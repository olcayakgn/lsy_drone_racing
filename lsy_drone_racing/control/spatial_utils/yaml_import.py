"""Small YAML loader for controller constants."""

import yaml


def load_yaml(file_path: str) -> dict | None:
    """Load and return the ``constants`` section from a YAML file, or None on error."""
    with open(file_path, "r") as file:
        try:
            config = yaml.safe_load(file)
            return config["constants"]
        except yaml.YAMLError as exc:
            print(f"Error reading YAML: {exc}")
            return None
