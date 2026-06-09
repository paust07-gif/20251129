"""Data transformation helpers for FRED time-series observations."""
from __future__ import annotations

import pandas as pd


REQUIRED_OBSERVATION_COLUMNS = {"date", "value"}


def prepare_monthly_series(observations: pd.DataFrame, label: str) -> pd.DataFrame:
    """Normalize FRED observation rows into monthly numeric series data."""
    missing_columns = REQUIRED_OBSERVATION_COLUMNS.difference(observations.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"관측 데이터에 필수 컬럼이 없습니다: {missing}")

    series = observations.loc[:, ["date", "value"]].copy()
    series["date"] = pd.to_datetime(series["date"], errors="coerce")
    series[label] = pd.to_numeric(series["value"].replace(".", pd.NA), errors="coerce")
    series = series.drop(columns="value").dropna(subset=["date", label])
    series = series.sort_values("date")
    series["month"] = series["date"].dt.to_period("M").dt.to_timestamp()
    return series.groupby("month", as_index=False)[label].last()


def combine_indicators(cpi: pd.DataFrame, pmi: pd.DataFrame) -> pd.DataFrame:
    """Outer-join CPI and PMI monthly data for plotting and tabular display."""
    return pd.merge(cpi, pmi, on="month", how="outer").sort_values("month")
