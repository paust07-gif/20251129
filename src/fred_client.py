"""Small requests-based FRED API client."""
from __future__ import annotations

from typing import Any

import pandas as pd
import requests

from src.config import FRED_BASE_URL


class FredApiError(RuntimeError):
    """Raised when a FRED API request fails or returns invalid data."""


class FredClient:
    """Fetch observations and search metadata from the FRED API."""

    def __init__(self, api_key: str, base_url: str = FRED_BASE_URL, timeout: int = 20) -> None:
        if not api_key:
            raise FredApiError("FRED_API_KEY가 설정되어 있지 않습니다. .env 파일에 FRED_API_KEY를 추가하세요.")
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _get(self, endpoint: str, params: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        request_params = {
            "api_key": self.api_key,
            "file_type": "json",
            **params,
        }
        try:
            response = requests.get(url, params=request_params, timeout=self.timeout)
            response.raise_for_status()
        except requests.RequestException as exc:
            raise FredApiError(f"FRED 요청 실패: {exc}") from exc

        payload = response.json()
        if "error_message" in payload:
            raise FredApiError(str(payload["error_message"]))
        return payload

    def get_series_observations(
        self,
        series_id: str,
        observation_start: str | None = None,
        observation_end: str | None = None,
    ) -> pd.DataFrame:
        """Return observation rows for a FRED series as a DataFrame."""
        params: dict[str, Any] = {"series_id": series_id.strip()}
        if observation_start:
            params["observation_start"] = observation_start
        if observation_end:
            params["observation_end"] = observation_end

        payload = self._get("series/observations", params)
        observations = payload.get("observations", [])
        if not observations:
            raise FredApiError(f"{series_id} 시리즈의 관측값을 찾을 수 없습니다.")
        return pd.DataFrame(observations)

    def search_series(self, search_text: str, limit: int = 10) -> pd.DataFrame:
        """Search FRED series metadata by keyword."""
        payload = self._get(
            "series/search",
            {
                "search_text": search_text.strip(),
                "limit": limit,
                "order_by": "popularity",
                "sort_order": "desc",
            },
        )
        return pd.DataFrame(payload.get("seriess", []))
