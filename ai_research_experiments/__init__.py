import logging
import logging.config
from pathlib import Path
from typing import Optional

import yaml


def get_yaml_config(file_path: Path) -> Optional[dict]:
    """Fetch yaml config and return as dict if it exists."""
    if file_path.exists():
        with open(file_path, "rt") as f:
            return yaml.load(f.read(), Loader=yaml.FullLoader)


# Define project base directory
PROJECT_DIR = Path(__file__).resolve().parents[1]


logging.basicConfig(
    format="%(asctime)s %(levelname)-2s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
# # base/global config
# _base_config_path = Path(__file__).parent.resolve() / "config/base.yaml"
# config = get_yaml_config(_base_config_path)
