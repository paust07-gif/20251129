"""Configuration helpers for the FRED Streamlit dashboard."""
from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


FRED_BASE_URL = "https://api.stlouisfed.org/fred"
DEFAULT_CPI_SERIES_ID = "CPIAUCSL"
DEFAULT_PMI_SERIES_ID = "NAPM"
DEFAULT_OBSERVATION_START = "2010-01-01"


@dataclass(frozen=True)
class Settings:
    """Runtime settings loaded from environment variables."""

    fred_api_key: str
    fred_base_url: str = FRED_BASE_URL


def get_settings() -> Settings:
    """Load settings from a local .env file and process environment."""
    load_dotenv()
    return Settings(fred_api_key=os.getenv("FRED_API_KEY", "").strip())
