"""Project configuration loaded from .env."""

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCHEMA_PATH = PROJECT_ROOT / "src" / "schema.sql"
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"

load_dotenv(PROJECT_ROOT / ".env")

DATABASE_URL = os.getenv(
    "DATABASE_URL", "postgresql+psycopg2://nfl:nfl@localhost:5433/nfl"
)

# Five completed seasons. 2026 is in progress as of this project's creation.
DEFAULT_SEASONS = [2021, 2022, 2023, 2024, 2025, 2026]

# Seasons with complete results, used for training/backtesting. 2026 is in progress
# and is loaded for live prediction only — never include it in a training set.
COMPLETED_SEASONS = [2021, 2022, 2023, 2024, 2025]
CURRENT_SEASON = 2026
