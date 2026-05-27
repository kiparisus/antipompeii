"""Configuration management."""
import yaml
from pathlib import Path
from typing import Dict, Any
from src.antipompeii.utils.logger import get_logger

ROOT_DIR = Path(__file__).parent.parent.parent.parent
DATA_DIR = ROOT_DIR / "src/data"

class ConfigManager:

    DEFAULT_CONFIG = {
        'data': {
            'input_directory': str(DATA_DIR / "input"),
            'output_directory': str(DATA_DIR / "output")
        },
        'extent': {
            'min_latitude': 0.0,
            'max_latitude': 0.0,
            'min_longitude': 0.0,
            'max_longitude': 0.0,
            'crs': 'EPSG:4326'
        },
        'processing': {
            'max_workers': 4,
            'chunk_size': 1000,
            'cache_enabled': True
        },
        'logging': {
            'level': 'INFO',
            'file': './logs/app.log',
            'console_output': True
        }
    }

    def __init__(self):
        self.logger = get_logger(__name__)
        self.config = self.DEFAULT_CONFIG.copy()
        self._ensure_directories()

    def load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as f:
                loaded_config = yaml.safe_load(f)
                self.config.update(loaded_config)
            self.logger.info(f"Configuration loaded from {config_path}")
        except FileNotFoundError:
            self.logger.warning(f"Config file not found: {config_path}, using defaults")
        except Exception as e:
            self.logger.error(f"Error loading config: {e}")

        self._ensure_directories()
        return self.config


    def save_config(self, config_path: str):
        """Save current configuration to file."""
        Path(config_path).parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
        self.logger.info(f"Configuration saved to {config_path}")

    def _ensure_directories(self):
        """Create necessary directories."""
        dirs = [
            self.config['data']['input_directory'],
            self.config['data']['output_directory'],
            Path(self.config['logging']['file']).parent
        ]
        for dir_path in dirs:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
