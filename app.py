from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from importlib import import_module
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


st.set_page_config(
    page_title="POSCO International Corp - LNG Market Insight",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


class LinearRegression:
    """Small linear-regression helper compatible with the required fit/coefficient API."""

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearRegression":
        x = np.asarray(X, dtype=float).reshape(-1)
        y_values = np.asarray(y, dtype=float).reshape(-1)
        if len(x) < 2 or len(y_values) < 2:
            self.coef_ = np.array([0.0])
            self.intercept_ = float(y_values[0]) if len(y_values) else 0.0
            return self
        slope, intercept = np.polyfit(x, y_values, 1)
        self.coef_ = np.array([float(slope)])
        self.intercept_ = float(intercept)
        return self


POSCO_BLUE = "#005BAC"
NAVY = "#12355B"
SKY_BLUE = "#EAF6FF"
CYAN = "#00AEEF"
BORDER = "#CFE8FF"
TEXT_MUTED = "#5B6B7F"


st.markdown(
    f"""
    <style>
        :root {{
            --posco-blue: {POSCO_BLUE};
            --navy: {NAVY};
            --sky-blue: {SKY_BLUE};
            --cyan: {CYAN};
            --border: {BORDER};
            --muted: {TEXT_MUTED};
        }}
        .stApp {{
            background: linear-gradient(180deg, #F7FBFF 0%, #FFFFFF 34%, #F8FCFF 100%);
            color: #10233F;
        }}
        section[data-testid="stSidebar"] {{
            background: linear-gradient(180deg, #FFFFFF 0%, #EEF8FF 100%);
            border-right: 1px solid var(--border);
        }}
        section[data-testid="stSidebar"] h1,
        section[data-testid="stSidebar"] h2,
        section[data-testid="stSidebar"] h3,
        section[data-testid="stSidebar"] label {{
            color: var(--navy) !important;
        }}
        div[data-testid="stMetric"], .metric-card, .chart-card, .status-card, .summary-card {{
            background: rgba(255, 255, 255, 0.96);
            border: 1px solid var(--border);
            border-radius: 18px;
            box-shadow: 0 14px 34px rgba(0, 91, 172, 0.10);
        }}
        div[data-testid="stMetric"] {{
            padding: 18px 18px 14px 18px;
            min-height: 118px;
        }}
        div[data-testid="stMetric"] label {{
            color: var(--muted) !important;
            font-weight: 700;
        }}
        div[data-testid="stMetric"] [data-testid="stMetricValue"] {{
            color: var(--navy) !important;
            font-weight: 800;
        }}
        div[data-testid="stMetric"] [data-testid="stMetricDelta"] {{
            color: var(--posco-blue) !important;
        }}
        .main-header {{
            padding: 26px 30px;
            margin: 0 0 22px 0;
            background: linear-gradient(135deg, #FFFFFF 0%, #F2FAFF 54%, #E6F5FF 100%);
            border: 1px solid var(--border);
            border-left: 7px solid var(--posco-blue);
            border-radius: 22px;
            box-shadow: 0 18px 44px rgba(0, 91, 172, 0.12);
        }}
        .main-title {{
            margin: 0;
            color: var(--navy);
            font-size: 2.18rem;
            font-weight: 850;
            letter-spacing: -0.025em;
        }}
        .main-subtitle {{
            margin: 8px 0 0 0;
            color: #325E8D;
            font-size: 1.02rem;
            font-weight: 650;
        }}
        .last-updated {{
            margin: 10px 0 0 0;
            color: var(--muted);
            font-size: 0.92rem;
            font-weight: 600;
        }}
        .chart-card {{
            padding: 18px 18px 8px 18px;
            margin-bottom: 18px;
        }}
        .section-title {{
            color: var(--navy);
            font-weight: 800;
            font-size: 1.08rem;
            margin-bottom: 10px;
        }}
        .sample-badge {{
            display: inline-flex;
            align-items: center;
            gap: 8px;
            background: #FFF7E6;
            color: #8A5A00;
            border: 1px solid #FFD58A;
            border-radius: 999px;
            padding: 7px 12px;
            font-weight: 800;
            font-size: 0.86rem;
            margin: 4px 0 14px 0;
        }}
        .status-line {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 9px 0;
            border-bottom: 1px solid #E4F1FC;
            color: var(--navy);
            font-size: 0.92rem;
        }}
        .status-line:last-child {{ border-bottom: none; }}
        .interpretation-box {{
            background: linear-gradient(135deg, #FFFFFF 0%, #F3FAFF 100%);
            border: 1px solid var(--border);
            border-radius: 18px;
            padding: 18px 20px;
            box-shadow: 0 12px 28px rgba(0, 91, 172, 0.08);
            color: #173D68;
            line-height: 1.55;
        }}
        .stTabs [data-baseweb="tab-list"] {{
            gap: 8px;
            border-bottom: 1px solid var(--border);
        }}
        .stTabs [data-baseweb="tab"] {{
            background: #FFFFFF;
            border: 1px solid var(--border);
            border-bottom: none;
            border-radius: 13px 13px 0 0;
            color: var(--navy);
            font-weight: 750;
        }}
        .stTabs [aria-selected="true"] {{
            background: linear-gradient(180deg, #E9F7FF 0%, #FFFFFF 100%);
            color: var(--posco-blue) !important;
        }}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def sample_spot_data(analysis_date: date, lookback_days: int) -> pd.DataFrame:
    end = pd.Timestamp(analysis_date)
    dates = pd.date_range(end=end, periods=lookback_days, freq="D")
    t = np.arange(len(dates))
    rng = np.random.default_rng(42)
    jkm = 12.2 + 0.9 * np.sin(t / 34) + 0.35 * np.cos(t / 11) + rng.normal(0, 0.12, len(t))
    ttf = 10.9 + 0.75 * np.sin(t / 38 + 0.6) + 0.28 * np.cos(t / 15) + rng.normal(0, 0.10, len(t))
    hh = 3.0 + 0.22 * np.sin(t / 27) + rng.normal(0, 0.04, len(t))
    gcm = 11.6 + 0.62 * np.sin(t / 41 + 1.1) + rng.normal(0, 0.11, len(t))
    brent = 78 + 5.1 * np.sin(t / 50) + 1.8 * np.cos(t / 18) + rng.normal(0, 0.65, len(t))
    wti = brent - 4.2 + 0.7 * np.sin(t / 23) + rng.normal(0, 0.35, len(t))
    return pd.DataFrame(
        {
            "date": dates,
            "JKM": np.maximum(jkm, 1),
            "TTF": np.maximum(ttf, 1),
            "HH": np.maximum(hh, 0.5),
            "GCM": np.maximum(gcm, 1),
            "Brent": np.maximum(brent, 10),
            "WTI": np.maximum(wti, 10),
        }
    )


@st.cache_data(show_spinner=False)
def sample_forward_data(analysis_date: date) -> pd.DataFrame:
    months = pd.date_range(
        pd.Timestamp(analysis_date).replace(day=1) + pd.DateOffset(months=1), periods=24, freq="MS"
    )
    positions = np.arange(1, 25)
    seasonal = 0.24 * np.sin((positions - 2) / 12 * 2 * np.pi)
    curve = 12.85 - 0.34 * (positions - 1) + seasonal + 0.05 * np.cos(positions / 2)
    return pd.DataFrame(
        {
            "derivative_position": positions,
            "contract_label": months.strftime("%b %Y"),
            "date": months,
            "jkm_forward": curve.round(3),
            "contract": [f"M+{i}" for i in positions],
        }
    )


@st.cache_data(show_spinner=False)
def sample_forecast_data(analysis_date: date) -> pd.DataFrame:
    months = pd.date_range(
        pd.Timestamp(analysis_date).replace(day=1) + pd.DateOffset(months=1), periods=24, freq="MS"
    )
    positions = np.arange(1, 25)
    forecast = 12.3 - 0.24 * (positions - 1) + 0.15 * np.sin(positions / 2.5)
    return pd.DataFrame({"date": months, "forecast_value": forecast.round(3)})


@st.cache_data(show_spinner=False)
def sample_ttf_forward_data(analysis_date: date) -> pd.DataFrame:
    months = pd.date_range(
        pd.Timestamp(analysis_date).replace(day=1) + pd.DateOffset(months=1), periods=24, freq="MS"
    )
    positions = np.arange(1, 25)
    curve = 10.7 - 0.18 * (positions - 1) + 0.18 * np.sin(positions / 2.4)
    return pd.DataFrame({"derivative_position": positions, "date": months, "ttf_forward": curve, "contract": [f"M+{i}" for i in positions]})


@st.cache_data(show_spinner=False)
def sample_hh_forward_data(analysis_date: date) -> pd.DataFrame:
    months = pd.date_range(
        pd.Timestamp(analysis_date).replace(day=1) + pd.DateOffset(months=1), periods=24, freq="MS"
    )
    positions = np.arange(1, 25)
    curve = 3.05 + 0.035 * (positions - 1) + 0.08 * np.sin(positions / 3)
    return pd.DataFrame({"derivative_position": positions, "date": months, "hh_forward": curve, "contract": [f"M+{i}" for i in positions]})


def find_numeric_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for col in candidates:
        if col in df.columns and pd.api.types.is_numeric_dtype(pd.to_numeric(df[col], errors="coerce")):
            return col
    return None


def build_spgci_clients() -> tuple[Any | None, Any | None, str]:
    try:
        spgci = import_module("spgci")
        client = getattr(spgci, "Client", None)
        if client is not None:
            api_client = client()
            fc = getattr(api_client, "forward_curves", None) or getattr(api_client, "forward_curve", None) or api_client
            lng = getattr(api_client, "lng", None) or getattr(api_client, "lng_analytics", None) or api_client
            return fc, lng, "Connected"
        return spgci, spgci, "Partial"
    except Exception as exc:
        return None, None, f"Disconnected: {exc}"


@st.cache_data(show_spinner=False, ttl=1800)
def load_jkm_forward_curve(analysis_date: date) -> tuple[pd.DataFrame, bool, str]:
    fc, _, status = build_spgci_clients()
    if fc is None:
        return sample_forward_data(analysis_date), True, status
    try:
        jkm_raw = fc.get_assessments(
            curve_code="CN06J",
            derivative_maturity_frequency="Month"
        )

        jkm = (
            jkm_raw[jkm_raw["bate"] == "c"]
            .sort_values("assessDate")
            .groupby("derivative_position", as_index=False)
            .last()
            .sort_values("derivative_position")
            .reset_index(drop=True)
        )

        jkm = jkm[
            (jkm["derivative_position"] >= 1) &
            (jkm["derivative_position"] <= 24)
        ].reset_index(drop=True)

        jkm["date"] = pd.to_datetime(
            "1 " + jkm["contract_label"], format="%d %b %Y"
        ).dt.to_period("M").dt.to_timestamp()

        candidate_cols = ["value", "price", "assessment", "close"]
        price_col = find_numeric_column(jkm, candidate_cols)
        if price_col is None:
            raise ValueError(f"No JKM price column found. Available columns: {list(jkm.columns)}")
        jkm_forward = jkm[["derivative_position", "contract_label", "date", price_col]].copy()
        jkm_forward["jkm_forward"] = pd.to_numeric(jkm_forward[price_col], errors="coerce")
        jkm_forward = jkm_forward.dropna(subset=["jkm_forward"])
        jkm_forward["contract"] = "M+" + jkm_forward["derivative_position"].astype(str)
        if jkm_forward.empty:
            raise ValueError("JKM forward curve returned no valid observations.")
        return jkm_forward[["derivative_position", "contract_label", "date", "jkm_forward", "contract"]], False, "Connected"
    except Exception as exc:
        return sample_forward_data(analysis_date), True, f"Partial: {exc}"


@st.cache_data(show_spinner=False, ttl=1800)
def load_sp_forecast_curve(analysis_date: date) -> tuple[pd.DataFrame, bool, str]:
    _, lng, status = build_spgci_clients()
    if lng is None:
        return sample_forecast_data(analysis_date), True, status
    try:
        fcast_raw = lng.get_price_monthly_forecast(
            price_marker_name="Asia Spot",
            date_gte=date.today(),
            date_lte=date(2028, 6, 30)
        )

        fcast = fcast_raw.copy()
        fcast["date"] = pd.to_datetime(fcast["date"]).dt.to_period("M").dt.to_timestamp()

        forecast_candidate_cols = [
            "forecast_value",
            "price",
            "value",
            "forecast",
            "forecastValue",
            "priceValue",
            "assessment"
        ]
        forecast_col = find_numeric_column(fcast, forecast_candidate_cols)
        if forecast_col is None:
            raise ValueError(f"No forecast value column found. Available columns: {list(fcast.columns)}")
        fcast_curve = fcast[["date", forecast_col]].copy()
        fcast_curve["forecast_value"] = pd.to_numeric(fcast_curve[forecast_col], errors="coerce")
        fcast_curve = fcast_curve.dropna(subset=["forecast_value"]).groupby("date", as_index=False).last()
        if fcast_curve.empty:
            raise ValueError("S&P forecast curve returned no valid observations.")
        return fcast_curve[["date", "forecast_value"]], False, "Connected"
    except Exception as exc:
        return sample_forecast_data(analysis_date), True, f"Partial: {exc}"


def fetch_yahoo_series(tickers: dict[str, str], analysis_date: date, lookback_days: int) -> tuple[pd.DataFrame | None, str]:
    try:
        yf = import_module("yfinance")
        start = pd.Timestamp(analysis_date) - pd.Timedelta(days=lookback_days + 14)
        end = pd.Timestamp(analysis_date) + pd.Timedelta(days=1)
        frames = []
        for label, ticker in tickers.items():
            hist = yf.download(ticker, start=start.date(), end=end.date(), progress=False, auto_adjust=True)
            if hist.empty:
                continue
            price = hist["Close"].rename(label).reset_index().rename(columns={"Date": "date"})
            frames.append(price)
        if not frames:
            return None, "Disconnected: no Yahoo Finance observations"
        merged = frames[0]
        for frame in frames[1:]:
            merged = pd.merge(merged, frame, on="date", how="outer")
        merged["date"] = pd.to_datetime(merged["date"]).dt.tz_localize(None)
        merged = merged.sort_values("date").tail(lookback_days)
        return merged, "Connected"
    except Exception as exc:
        return None, f"Disconnected: {exc}"


def fetch_fred_probe() -> str:
    try:
        fredapi = import_module("fredapi")
        api_key = st.secrets.get("FRED_API_KEY", None) if hasattr(st, "secrets") else None
        if not api_key:
            return "Partial: FRED_API_KEY not configured"
        fred = fredapi.Fred(api_key=api_key)
        _ = fred.get_series("DCOILBRENTEU", observation_start="2024-01-01").tail(1)
        return "Connected"
    except Exception as exc:
        return f"Disconnected: {exc}"


def display_status(status: str) -> str:
    lowered = status.lower()
    if "connected" in lowered and "disconnected" not in lowered:
        return "🟢 Connected"
    if "partial" in lowered:
        return "🟡 Partial"
    return "🔴 Disconnected"


def apply_plotly_layout(fig: go.Figure, title: str, y_title: str | None = None) -> go.Figure:
    fig.update_layout(
        title={"text": title, "font": {"color": NAVY, "size": 18}},
        hovermode="x unified",
        template="plotly_white",
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=24, r=24, t=58, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        font=dict(color="#1E3552"),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#E6F2FF", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="#E6F2FF", zeroline=False, title=y_title)
    return fig


def render_chart(fig: go.Figure) -> None:
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    st.markdown('</div>', unsafe_allow_html=True)


def line_chart(df: pd.DataFrame, columns: list[str], title: str, unit: str, colors: list[str]) -> go.Figure:
    fig = go.Figure()
    for col, color in zip(columns, colors):
        fig.add_trace(
            go.Scatter(
                x=df["date"],
                y=df[col],
                mode="lines+markers",
                name=col,
                line=dict(color=color, width=2.7),
                marker=dict(size=4),
                hovertemplate=(
                    f"<b>{col}</b><br>"
                    "Date: %{x|%Y-%m-%d}<br>"
                    f"Price: %{{y:.2f}} {unit}"
                    "<extra></extra>"
                ),
            )
        )
    return apply_plotly_layout(fig, title, unit)


def forward_curve_chart(df: pd.DataFrame, value_col: str, name: str, title: str, color: str) -> go.Figure:
    customdata = np.stack([df["contract"], df["date"].dt.strftime("%b %Y")], axis=-1)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["date"],
            y=df[value_col],
            customdata=customdata,
            mode="lines+markers",
            name=name,
            line=dict(color=color, width=3),
            marker=dict(size=7, color=color),
            hovertemplate=(
                "Contract: %{customdata[0]}<br>"
                "Month: %{customdata[1]}<br>"
                f"{name}: %{{y:.2f}} $/MMBtu"
                "<extra></extra>"
            ),
        )
    )
    return apply_plotly_layout(fig, title, "$/MMBtu")


def calculate_structure(jkm_forward: pd.DataFrame) -> tuple[float, str, float]:
    first_12 = jkm_forward.sort_values("derivative_position").head(12)
    if len(first_12) < 12:
        return 0.0, "FLAT", 0.0
    jkm_m1 = float(first_12.loc[first_12["derivative_position"] == 1, "jkm_forward"].iloc[0])
    jkm_m12 = float(first_12.loc[first_12["derivative_position"] == 12, "jkm_forward"].iloc[0])
    structure_spread = jkm_m12 - jkm_m1
    if structure_spread > 0.10:
        structure = "CONTANGO"
    elif structure_spread < -0.10:
        structure = "BACKWARDATION"
    else:
        structure = "FLAT"

    X = np.array(first_12["derivative_position"]).reshape(-1, 1)
    y = np.array(first_12["jkm_forward"])
    model = LinearRegression().fit(X, y)
    slope = model.coef_[0]
    return float(structure_spread), structure, float(slope)


def correlation_summary(spot_df: pd.DataFrame, analysis_date: date) -> pd.DataFrame:
    periods = [("1 Year", 365), ("6 Months", 183), ("3 Months", 92), ("1 Month", 31)]
    rows = []
    end = pd.Timestamp(analysis_date)
    for label, days in periods:
        window = spot_df[spot_df["date"] >= end - pd.Timedelta(days=days)]
        corr = window["JKM"].corr(window["TTF"]) if len(window) >= 3 else np.nan
        if pd.isna(corr):
            interpretation = "Insufficient Data"
        elif corr >= 0.70:
            interpretation = "Strong Coupling"
        elif corr >= 0.40:
            interpretation = "Moderate Coupling"
        elif corr >= 0.10:
            interpretation = "Decoupling"
        else:
            interpretation = "Dislocated"
        rows.append({"Period": label, "Pearson Correlation": corr, "Interpretation": interpretation})
    return pd.DataFrame(rows)


def correlation_heatmap(corr_df: pd.DataFrame) -> go.Figure:
    z = [corr_df["Pearson Correlation"].fillna(0).tolist()]
    customdata = np.array([[ [row["Period"], row["Interpretation"]] for _, row in corr_df.iterrows() ]], dtype=object)
    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=corr_df["Period"],
            y=["JKM-TTF"],
            customdata=customdata,
            colorscale=[[0, "#EAF6FF"], [0.5, "#70C7FF"], [1, POSCO_BLUE]],
            zmin=-1,
            zmax=1,
            hovertemplate=(
                "Period: %{customdata[0]}<br>"
                "Pearson Correlation: %{z:.2f}<br>"
                "Interpretation: %{customdata[1]}"
                "<extra></extra>"
            ),
            colorbar=dict(title="ρ"),
        )
    )
    return apply_plotly_layout(fig, "JKM-TTF Pearson Correlation", None)


def comparison_chart(comparison: pd.DataFrame) -> go.Figure:
    df = comparison.copy()
    df["month_label"] = df["date"].dt.strftime("%b %Y")
    custom = np.stack(
        [df["month_label"], df["jkm_forward"], df["forecast_value"], df["spread"], df["spread_pct"]], axis=-1
    )
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["date"],
            y=df["jkm_forward"],
            customdata=custom,
            mode="lines+markers",
            name="JKM Forward",
            line=dict(color=POSCO_BLUE, width=3),
            hovertemplate=(
                "Month: %{customdata[0]}<br>"
                "JKM Forward: %{customdata[1]:.2f} $/MMBtu<br>"
                "S&P Forecast: %{customdata[2]:.2f} $/MMBtu<br>"
                "Spread: %{customdata[3]:+.2f} $/MMBtu<br>"
                "Spread %: %{customdata[4]:+.2f}%"
                "<extra></extra>"
            ),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df["date"],
            y=df["forecast_value"],
            customdata=custom,
            mode="lines+markers",
            name="S&P Asia Spot Forecast",
            line=dict(color=CYAN, width=3, dash="dash"),
            hovertemplate=(
                "Month: %{customdata[0]}<br>"
                "JKM Forward: %{customdata[1]:.2f} $/MMBtu<br>"
                "S&P Forecast: %{customdata[2]:.2f} $/MMBtu<br>"
                "Spread: %{customdata[3]:+.2f} $/MMBtu<br>"
                "Spread %: %{customdata[4]:+.2f}%"
                "<extra></extra>"
            ),
        )
    )
    return apply_plotly_layout(fig, "JKM Forward Curve vs S&P Asia Spot Forecast", "$/MMBtu")


def spread_bar_chart(comparison: pd.DataFrame) -> go.Figure:
    df = comparison.copy()
    df["month_label"] = df["date"].dt.strftime("%b %Y")
    colors = np.where(df["spread"] >= 0, POSCO_BLUE, "#FF8A65")
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=df["date"],
            y=df["spread"],
            customdata=df["month_label"],
            name="Forward - Forecast Spread",
            marker_color=colors,
            hovertemplate=(
                "Month: %{customdata}<br>"
                "Forward - Forecast Spread: %{y:+.2f} $/MMBtu"
                "<extra></extra>"
            ),
        )
    )
    fig.add_hline(y=0, line_width=1, line_dash="dash", line_color="#6B7C93")
    return apply_plotly_layout(fig, "Forward - Forecast Spread", "$/MMBtu")


def netback_matrix(jkm_forward: pd.DataFrame, hh_forward: pd.DataFrame) -> pd.DataFrame:
    base = pd.merge(jkm_forward[["date", "contract", "jkm_forward"]], hh_forward[["date", "hh_forward"]], on="date", how="inner")
    liquefaction = 2.35
    shipping_cases = {"Low Freight": 1.05, "Base Freight": 1.45, "High Freight": 1.90}
    feedgas_mult = 1.15
    matrix = pd.DataFrame({"Contract": base["contract"], "Month": base["date"].dt.strftime("%b %Y")})
    for case, freight in shipping_cases.items():
        matrix[case] = base["jkm_forward"] - (base["hh_forward"] * feedgas_mult + liquefaction + freight)
    return matrix.head(12)


def netback_heatmap(matrix: pd.DataFrame) -> go.Figure:
    value_cols = ["Low Freight", "Base Freight", "High Freight"]
    z = matrix[value_cols].T.values
    x_labels = matrix["Contract"] + " | " + matrix["Month"]
    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=x_labels,
            y=value_cols,
            colorscale=[[0, "#FFECE7"], [0.5, "#FFFFFF"], [1, "#A7E8FF"]],
            zmid=0,
            hovertemplate=(
                "Contract: %{x}<br>"
                "Scenario: %{y}<br>"
                "USGC Netback Margin: %{z:+.2f} $/MMBtu"
                "<extra></extra>"
            ),
            colorbar=dict(title="$/MMBtu"),
        )
    )
    return apply_plotly_layout(fig, "USGC-to-Asia Netback Matrix", "")


def curve_structure_chart(jkm_forward: pd.DataFrame) -> go.Figure:
    df = jkm_forward.sort_values("derivative_position").head(12).copy()
    X = np.array(df["derivative_position"]).reshape(-1, 1)
    y = np.array(df["jkm_forward"])
    model = LinearRegression().fit(X, y)
    df["trend"] = model.intercept_ + model.coef_[0] * df["derivative_position"]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["derivative_position"],
            y=df["jkm_forward"],
            mode="lines+markers",
            name="JKM Forward",
            line=dict(color=POSCO_BLUE, width=3),
            hovertemplate=(
                "Contract: M+%{x}<br>"
                "JKM Forward: %{y:.2f} $/MMBtu"
                "<extra></extra>"
            ),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df["derivative_position"],
            y=df["trend"],
            mode="lines",
            name="Linear Regression Trend",
            line=dict(color="#00AEEF", width=2.5, dash="dash"),
            hovertemplate=(
                "Contract: M+%{x}<br>"
                "Regression Value: %{y:.2f} $/MMBtu"
                "<extra></extra>"
            ),
        )
    )
    return apply_plotly_layout(fig, "JKM Curve Structure: M+1 to M+12", "$/MMBtu")


def spread_chart(spot_df: pd.DataFrame) -> go.Figure:
    df = spot_df.copy()
    df["JKM - TTF Spread"] = df["JKM"] - df["TTF"]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["date"],
            y=df["JKM - TTF Spread"],
            mode="lines+markers",
            name="JKM - TTF Spread",
            line=dict(color=POSCO_BLUE, width=2.7),
            marker=dict(size=4),
            hovertemplate=(
                "<b>JKM - TTF Spread</b><br>"
                "Date: %{x|%Y-%m-%d}<br>"
                "Spread: %{y:+.2f} $/MMBtu"
                "<extra></extra>"
            ),
        )
    )
    fig.add_hline(y=0, line_width=1, line_dash="dash", line_color="#6B7C93")
    return apply_plotly_layout(fig, "JKM - TTF Spread", "$/MMBtu")


with st.sidebar:
    st.markdown("### Market Controls")
    analysis_date = st.date_input("Analysis Date", value=date.today(), max_value=date.today() + timedelta(days=365))
    lookback_label = st.selectbox("Lookback Period", ["1 Year", "6 Months", "3 Months", "1 Month"], index=0)
    lookback_days = {"1 Year": 365, "6 Months": 183, "3 Months": 92, "1 Month": 31}[lookback_label]


jkm_forward, jkm_is_sample, jkm_status = load_jkm_forward_curve(analysis_date)
fcast_curve, fcast_is_sample, fcast_status = load_sp_forecast_curve(analysis_date)
spot_df = sample_spot_data(analysis_date, max(lookback_days, 365))
ttf_forward = sample_ttf_forward_data(analysis_date)
hh_forward = sample_hh_forward_data(analysis_date)
yahoo_df, yahoo_status = fetch_yahoo_series({"Brent": "BZ=F", "WTI": "CL=F"}, analysis_date, lookback_days)
if yahoo_df is not None and {"Brent", "WTI"}.issubset(yahoo_df.columns):
    oil_overlay = yahoo_df[["date", "Brent", "WTI"]].dropna()
    if not oil_overlay.empty:
        spot_df = spot_df.drop(columns=["Brent", "WTI"]).merge(oil_overlay, on="date", how="left")
        spot_df[["Brent", "WTI"]] = spot_df[["Brent", "WTI"]].ffill().bfill()
fred_status = fetch_fred_probe()

comparison = pd.merge(
    jkm_forward,
    fcast_curve,
    on="date",
    how="inner"
)

if not comparison.empty:
    comparison["spread"] = comparison["jkm_forward"] - comparison["forecast_value"]
    comparison["spread_pct"] = comparison["spread"] / comparison["forecast_value"] * 100
else:
    st.warning("Forward와 Forecast의 월물이 일치하지 않아 Sample / Estimated Data 비교 곡선을 사용합니다.")
    sample_jkm = sample_forward_data(analysis_date)
    sample_fcast = sample_forecast_data(analysis_date)
    comparison = pd.merge(sample_jkm, sample_fcast, on="date", how="inner")
    comparison["spread"] = comparison["jkm_forward"] - comparison["forecast_value"]
    comparison["spread_pct"] = comparison["spread"] / comparison["forecast_value"] * 100
    jkm_is_sample = True
    fcast_is_sample = True

structure_spread, market_structure, slope = calculate_structure(jkm_forward)
latest_spot = float(spot_df["JKM"].iloc[-1])
latest_hh = float(spot_df["HH"].iloc[-1])
latest_jkm_forward = float(jkm_forward.sort_values("derivative_position")["jkm_forward"].iloc[0])
usgc_margin = latest_jkm_forward - (latest_hh * 1.15 + 2.35 + 1.45)
arb_signal = "OPEN" if usgc_margin > 0.75 else "WATCH" if usgc_margin > 0.15 else "CLOSED"

with st.sidebar:
    st.markdown("### Data Source Status")
    statuses = {
        "S&P Global Spot": "Partial: sample spot proxy in use",
        "S&P Global Forward": jkm_status,
        "S&P Global Forecast": fcast_status,
        "FRED": fred_status,
        "Yahoo Finance": yahoo_status,
    }
    st.markdown('<div class="status-card" style="padding: 12px 14px;">', unsafe_allow_html=True)
    for source, status in statuses.items():
        st.markdown(
            f'<div class="status-line"><span>{source}</span><strong>{display_status(status)}</strong></div>',
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)
    if jkm_is_sample or fcast_is_sample or yahoo_df is None:
        st.markdown('<div class="sample-badge">Sample / Estimated Data</div>', unsafe_allow_html=True)

kst = timezone(timedelta(hours=9))
st.markdown(
    f"""
    <div class="main-header">
        <h1 class="main-title">POSCO International Corp - LNG Market Insight</h1>
        <p class="main-subtitle">Global LNG Market Intelligence Dashboard | Spot, Forward Curve, Forecast Gap & Arbitrage Signal</p>
        <p class="last-updated">Last Updated: {datetime.now(kst).strftime('%Y-%m-%d %H:%M')} KST</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if jkm_is_sample or fcast_is_sample:
    st.warning("Sample / Estimated Data: 일부 S&P Global 데이터가 연결되지 않아 샘플/추정 데이터를 사용 중입니다.")
if yahoo_df is None:
    st.info("Sample / Estimated Data: Yahoo Finance 데이터가 없거나 누락되어 샘플 유가 데이터를 사용 중입니다.")
if "disconnected" in fred_status.lower():
    st.info("FRED API 실패 또는 미설정 상태입니다. 대시보드는 사용 가능한 데이터와 샘플 데이터로 계속 표시됩니다.")

kpi_cols = st.columns(4)
kpi_cols[0].metric("JKM Spot", f"{latest_spot:.2f} $/MMBtu", delta=f"M+1 {latest_jkm_forward:.2f}")
kpi_cols[1].metric("USGC Margin", f"{usgc_margin:.2f} $/MMBtu", delta="Netback to Asia")
kpi_cols[2].metric("Arbitrage Signal", arb_signal, delta="USGC → Asia")
kpi_cols[3].metric("JKM Market Structure", market_structure, delta=f"Slope: {slope:+.2f} $/mo")

tab1, tab2, tab3 = st.tabs([
    "Global Spot & Coupling Analysis",
    "Forward Curve & Netback Signal",
    "JKM Forward vs S&P Forecast",
])

with tab1:
    spot_window = spot_df.tail(lookback_days).copy()
    left, right = st.columns(2)
    with left:
        render_chart(line_chart(spot_window, ["JKM", "TTF", "HH", "GCM"], "Global LNG & Gas Spot Price Trend", "$/MMBtu", [POSCO_BLUE, CYAN, "#56B870", "#7B61FF"]))
        render_chart(spread_chart(spot_window))
    with right:
        render_chart(line_chart(spot_window, ["Brent", "WTI"], "Crude Oil Benchmark Price Trend", "$/bbl", [NAVY, "#4AA3FF"]))
        corr_df = correlation_summary(spot_df, analysis_date)
        render_chart(correlation_heatmap(corr_df))
        st.dataframe(corr_df, width="stretch", hide_index=True)

with tab2:
    left, right = st.columns(2)
    with left:
        render_chart(forward_curve_chart(jkm_forward, "jkm_forward", "JKM Forward", "JKM Forward Curve", POSCO_BLUE))
        render_chart(forward_curve_chart(ttf_forward, "ttf_forward", "TTF Forward", "TTF Forward Curve / Implied Curve", CYAN))
    with right:
        render_chart(forward_curve_chart(hh_forward, "hh_forward", "HH Forward", "Henry Hub Forward Curve", "#56B870"))
        render_chart(curve_structure_chart(jkm_forward))

    structure_cols = st.columns(3)
    structure_cols[0].metric("M+12 - M+1", f"{structure_spread:+.2f} $/MMBtu")
    structure_cols[1].metric("Contango / Backwardation", market_structure)
    structure_cols[2].metric("Linear Regression Slope", f"{slope:+.2f} $/mo")

    matrix = netback_matrix(jkm_forward, hh_forward)
    render_chart(netback_heatmap(matrix))
    st.dataframe(matrix.style.format({"Low Freight": "{:+.2f}", "Base Freight": "{:+.2f}", "High Freight": "{:+.2f}"}), width="stretch", hide_index=True)

with tab3:
    if comparison.empty:
        st.warning("Forward와 Forecast merge 결과가 비어 있습니다. S&P API의 월물 및 날짜 컬럼을 확인해 주세요.")
    else:
        left, right = st.columns(2)
        with left:
            render_chart(comparison_chart(comparison))
        with right:
            render_chart(spread_bar_chart(comparison))

        table = comparison[["date", "contract", "jkm_forward", "forecast_value", "spread", "spread_pct"]].copy()
        table["Month"] = table["date"].dt.strftime("%b %Y")
        table = table[["contract", "Month", "jkm_forward", "forecast_value", "spread", "spread_pct"]]
        table.columns = ["Contract", "Month", "JKM Forward", "S&P Forecast", "Spread", "Spread %"]
        st.markdown('<div class="section-title">Spread Summary Table</div>', unsafe_allow_html=True)
        st.dataframe(
            table.style.format(
                {
                    "JKM Forward": "{:.2f}",
                    "S&P Forecast": "{:.2f}",
                    "Spread": "{:+.2f}",
                    "Spread %": "{:+.2f}%",
                }
            ),
            width="stretch",
            hide_index=True,
        )

        avg_spread = float(comparison["spread"].mean())
        max_abs = comparison.iloc[comparison["spread"].abs().argmax()]
        if avg_spread > 0.25:
            read = "Forward curve is pricing a premium versus S&P Asia Spot Forecast, suggesting near-term risk premium, procurement urgency, or tighter prompt balances."
        elif avg_spread < -0.25:
            read = "Forward curve is discounting S&P Asia Spot Forecast, implying softer market expectations or potential downside to forecast assumptions."
        else:
            read = "Forward curve is broadly aligned with S&P Asia Spot Forecast, indicating limited directional dislocation at current levels."
        st.markdown(
            f"""
            <div class="interpretation-box">
                <strong>Market Interpretation</strong><br>
                Average Forward - Forecast spread is <strong>{avg_spread:+.2f} $/MMBtu</strong>. {read}<br>
                Largest absolute gap appears in <strong>{max_abs['date'].strftime('%b %Y')}</strong> at
                <strong>{max_abs['spread']:+.2f} $/MMBtu</strong> ({max_abs['spread_pct']:+.2f}%).
                Current curve structure is <strong>{market_structure}</strong> with M+12 - M+1 at
                <strong>{structure_spread:+.2f} $/MMBtu</strong> and regression slope of
                <strong>{slope:+.2f} $/mo</strong>.
            </div>
            """,
            unsafe_allow_html=True,
        )
