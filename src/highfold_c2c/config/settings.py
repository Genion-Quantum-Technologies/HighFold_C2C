"""
HighFold-C2C Settings

Central configuration settings loaded from environment variables.
"""

import os
from pathlib import Path
from functools import lru_cache

# Load .env file if available
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent.parent.parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    pass


class Settings:
    """Application settings loaded from environment variables."""

    def __init__(self):
        # Application
        self.APP_NAME: str = os.getenv("APP_NAME", "HighFold-C2C")
        self.APP_VERSION: str = os.getenv("APP_VERSION", "1.0.0")
        self.DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"

        # HighFold-specific
        self.C2C_CHECKPOINT_PATH: str = os.getenv(
            "C2C_CHECKPOINT_PATH", "checkpoints/c2c_model.pt"
        )
        self.COLABFOLD_BIN: str = os.getenv("COLABFOLD_BIN", "colabfold_batch")
        self.DEFAULT_MODEL_TYPE: str = os.getenv("DEFAULT_MODEL_TYPE", "alphafold2")
        self.DEFAULT_MSA_MODE: str = os.getenv("DEFAULT_MSA_MODE", "single_sequence")
        self.DEFAULT_NUM_MODELS: int = int(os.getenv("DEFAULT_NUM_MODELS", "5"))
        self.DEFAULT_SPAN_LEN: int = int(os.getenv("DEFAULT_SPAN_LEN", "5"))
        self.DEFAULT_NUM_SAMPLE: int = int(os.getenv("DEFAULT_NUM_SAMPLE", "20"))
        self.DEFAULT_TEMPERATURE: float = float(os.getenv("DEFAULT_TEMPERATURE", "1.0"))
        self.DEFAULT_TOP_P: float = float(os.getenv("DEFAULT_TOP_P", "0.9"))
        self.DEFAULT_SEED: int = int(os.getenv("DEFAULT_SEED", "42"))

        # Paths
        self.RESOURCE_DIR: str = os.getenv("RESOURCE_DIR", "/tmp/highfold_c2c/resource")
        self.TEMP_DIR: str = os.getenv("TEMP_DIR", "/tmp/highfold_c2c")
        self.LOG_DIR: str = os.getenv("LOG_DIR", "/tmp/highfold_c2c/logs")



@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
