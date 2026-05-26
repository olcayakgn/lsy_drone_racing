import yaml


def load_yaml(file_path):
    with open(file_path, "r") as file:
        try:
            config = yaml.safe_load(file)
            return config["constants"]
        except yaml.YAMLError as exc:
            print(f"Error reading YAML: {exc}")
            return None
