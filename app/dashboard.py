"""Streamlit dashboard for monthly CPI and PMI data from FRED."""
from __future__ import annotations

from datetime import date

import pandas as pd
import plotly.express as px
import streamlit as st

from src.config import DEFAULT_CPI_SERIES_ID, DEFAULT_OBSERVATION_START, DEFAULT_PMI_SERIES_ID, get_settings
from src.fred_client import FredApiError, FredClient
from src.transform import combine_indicators, prepare_monthly_series


st.set_page_config(page_title="FRED CPI & PMI Dashboard", layout="wide")


@st.cache_data(ttl=3600, show_spinner=False)
def load_series(series_id: str, observation_start: str) -> pd.DataFrame:
    settings = get_settings()
    client = FredClient(settings.fred_api_key, settings.fred_base_url)
    return client.get_series_observations(series_id=series_id, observation_start=observation_start)


@st.cache_data(ttl=3600, show_spinner=False)
def search_fred_series(query: str, limit: int) -> pd.DataFrame:
    settings = get_settings()
    client = FredClient(settings.fred_api_key, settings.fred_base_url)
    return client.search_series(query, limit=limit)


def show_search_tool(default_query: str) -> None:
    """Render a FRED series search helper."""
    st.subheader("FRED Series Search")
    search_text = st.text_input("검색어", value=default_query, help="예: PMI, ISM manufacturing, purchasing managers")
    limit = st.slider("검색 결과 개수", min_value=5, max_value=25, value=10, step=5)
    if st.button("시리즈 검색", type="secondary") and search_text.strip():
        try:
            results = search_fred_series(search_text.strip(), limit)
            if results.empty:
                st.info("검색 결과가 없습니다.")
                return
            columns = [column for column in ["id", "title", "frequency", "units", "last_updated"] if column in results]
            st.dataframe(results.loc[:, columns], use_container_width=True)
        except FredApiError as exc:
            st.error(f"FRED 검색 실패: {exc}")


def main() -> None:
    st.title("FRED 월간 CPI & PMI 대시보드")
    st.caption("CPI 기본 시리즈는 CPIAUCSL이며, PMI 시리즈는 사이드바에서 변경할 수 있습니다.")

    st.sidebar.header("설정")
    default_start_date = date.fromisoformat(DEFAULT_OBSERVATION_START)
    observation_start = st.sidebar.date_input("시작일", value=default_start_date)
    observation_start_text = observation_start.isoformat()
    pmi_series_id = st.sidebar.text_input("PMI series_id", value=DEFAULT_PMI_SERIES_ID)

    try:
        cpi_raw = load_series(DEFAULT_CPI_SERIES_ID, observation_start_text)
        cpi = prepare_monthly_series(cpi_raw, "CPI")
    except (FredApiError, ValueError) as exc:
        st.error(f"CPI 데이터를 불러오지 못했습니다: {exc}")
        st.stop()

    pmi = None
    pmi_error = None
    try:
        pmi_raw = load_series(pmi_series_id, observation_start_text)
        pmi = prepare_monthly_series(pmi_raw, "PMI")
    except (FredApiError, ValueError) as exc:
        pmi_error = exc
        st.error(f"PMI 데이터를 불러오지 못했습니다: {exc}")
        show_search_tool(pmi_series_id or "PMI")

    if pmi is not None:
        combined = combine_indicators(cpi, pmi)
        chart_data = combined.melt(id_vars="month", value_vars=["CPI", "PMI"], var_name="indicator", value_name="value")
        chart = px.line(chart_data, x="month", y="value", color="indicator", markers=True, title="Monthly CPI and PMI")
        st.plotly_chart(chart, use_container_width=True)
        st.dataframe(combined.sort_values("month", ascending=False), use_container_width=True)
    elif pmi_error is not None:
        chart = px.line(cpi, x="month", y="CPI", markers=True, title="Monthly CPI")
        st.plotly_chart(chart, use_container_width=True)
        st.dataframe(cpi.sort_values("month", ascending=False), use_container_width=True)


if __name__ == "__main__":
    main()
