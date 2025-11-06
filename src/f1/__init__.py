import os
import yaml
from typing import Dict, Any, Iterator

CONFIG_ENV_VAR = "F1_CONFIG"
DEFAULT_CONFIG_PATH = "./conf/config.yaml"
__all__ = ['config', 'load_config', 'CONFIG_ENV_VAR', 'DEFAULT_CONFIG_PATH']


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from a YAML file.

    :param config_path: Path to the YAML configuration file
    :return: Dictionary containing configuration data
    :raises FileNotFoundError: If the configuration file doesn't exist
    :raises ValueError: If there are issues parsing the YAML file
    """
    # Expand user home directory if using ~
    config_path = os.path.expanduser(config_path)

    # Check if file exists
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    # Read YAML file
    with open(config_path, 'r') as file:
        try:
            config_data = yaml.safe_load(file)
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML file: {e}")

    # Ensure the loaded data is a dictionary
    if not isinstance(config_data, dict):
        raise ValueError("YAML file must contain a dictionary")

    # Override configuration with environment variables
    config_data = override_config_from_env(config_data)

    return config_data


def iterate_over_leaf(config_data: Dict[str, Any], parent_path: str = "") -> Iterator[tuple[str, Any]]:
    """
    Iterate over the configuration dictionary and yield key-value pairs.

    :param config_data: Dictionary containing configuration data
    :param parent_path: Current path in the configuration hierarchy
    :return: Generator yielding path and value
    """
    for key, value in config_data.items():
        if isinstance(value, dict):
            if parent_path:
                new_parent_path = f"{parent_path}.{key}"
            else:
                new_parent_path = key
            yield from iterate_over_leaf(config_data=value, parent_path=new_parent_path)
        else:
            full_path = f"{parent_path}.{key}" if parent_path else key
            yield f"{full_path}", value


def set_nested_value(d: Dict[str, Any], path: str, value: Any) -> None:
    """
    Set a value in a nested dictionary using a dotted path string.

    :param d: Dictionary to update
    :param path: Dotted path string (e.g., "patientSummary-llm.base_url")
    :param value: Value to set
    """
    keys = path.split('.')
    for key in keys[:-1]:
        d = d.setdefault(key, {})
    d[keys[-1]] = value


def override_config_from_env(config_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Override configuration values using environment variables.
    Environment variables take precedence over YAML configuration.

    :param config_data: Dictionary containing configuration data from YAML
    :return: Dictionary with values overridden by environment variables
    """
    for full_path, value in iterate_over_leaf(config_data=config_data, parent_path=""):
        env_var_name = f"f1_cfg.{full_path}"
        if os.environ.get(env_var_name):
            # As long as environment variable exists, override the value
            env_value = os.environ[env_var_name]
            try:
                # Convert to appropriate type if necessary
                if isinstance(value, bool):
                    converted_value = env_value.lower() in ['true', '1']
                elif isinstance(value, int):
                    converted_value = int(env_value)
                elif isinstance(value, float):
                    converted_value = float(env_value)
                elif isinstance(value, list):  # 仅支持list of str
                    converted_value = [x.strip() for x in env_value.split(',')]
                else:
                    converted_value = env_value

                # Update the nested value in the config dictionary
                set_nested_value(config_data, full_path, converted_value)

            except ValueError as e:
                raise RuntimeError(
                    f"Error parsing environment variable {env_var_name}, env value: {env_value}, target_type: {type(value)}, error: {e}")

    return config_data


def _resolve_config_path() -> str:
    """Resolve config path: prefer F1_CONFIG, else ./conf/config.yaml. Error if missing."""
    env_path = os.environ.get(CONFIG_ENV_VAR)
    if env_path:
        expanded = os.path.expanduser(env_path)
        if os.path.exists(expanded):
            return expanded
        raise FileNotFoundError(
            f"Environment variable {CONFIG_ENV_VAR} is set to '{env_path}', but the file does not exist.")

    # Fall back to default relative path
    if os.path.exists(DEFAULT_CONFIG_PATH):
        return DEFAULT_CONFIG_PATH

    raise FileNotFoundError(
        f"No configuration file found. Set {CONFIG_ENV_VAR} to your YAML path, "
        f"or create one at {DEFAULT_CONFIG_PATH}.")


# Load config at import time (required by design)
config: Dict[str, Any] = load_config(_resolve_config_path())
