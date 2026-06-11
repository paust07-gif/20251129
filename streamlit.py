# app.py
import streamlit as st
import spgci as ci
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import date, timedelta

# ─── 페이지 설정 ─────────────────────────────────────────────
st.set_page_config(
    page_title="LNG Market Intelligence",
    page_icon="",
    layout="wide"
)

# ─── 사이드바: 인증 ───────────────────────────────────────────
with st.sidebar:
    st.title("설정")
    username = st.text_input("SPGCI Username", type="default")
    password = st.text_input("SPGCI Password", type="password")
    run_btn  = st.button("데이터 로드", type="primary", use_container_width=True)
    st.divider()
    st.caption("LNG Market Intelligence Dashboard")
    st.caption("Data: S&P Global (spgci)")

if not run_btn:
    st.info("사이드바에서 SPGCI 인증 후 '데이터 로드'를 눌러주세요.")
    st.stop()

# ─── 데이터 로드 ─────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def load_all_data(username, password):
    ci.set_credentials(username, password)
    fc  = ci.ForwardCurves()
    md  = ci.MarketData()
    lng = ci.LNGGlobalAnalytics()

    # 1. JKM Forward Curve
    jkm_raw = fc.get_assessments(
        curve_code="CN06J",
        derivative_maturity_frequency="Month"
    )
    jkm = (
        jkm_raw[jkm_raw["bate"] == "c"]
        .sort_values("assessDate")
        .groupby("derivative_position", as_index=False).last()
        .sort_values("derivative_position")
        .reset_index(drop=True)
    )
    jkm = jkm[(jkm["derivative_position"] >= 1) &
              (jkm["derivative_position"] <= 24)].reset_index(drop=True)
    jkm["date"] = pd.to_datetime(
        "1 " + jkm["contract_label"], format="%d %b %Y"
    ).dt.to_period("M").dt.to_timestamp()

    # 2. TTF Mo01/Mo02
    ttf_raw = md.get_assessments_by_symbol_historical(
        symbol=["DTMSC01", "DTMSC02"],
        assess_date_gte=date.today() - timedelta(days=14)
    )
    ttf = (
        ttf_raw[ttf_raw["bate"] == "c"]
        .sort_values("assessDate")
        .groupby("symbol", as_index=False).last()
    )
    ttf_m1 = ttf[ttf["symbol"] == "DTMSC01"]["value"].iloc[0]
    ttf_m2 = ttf[ttf["symbol"] == "DTMSC02"]["value"].iloc[0]
    ttf_asof = ttf["assessDate"].max()

    # 3. Asia Spot Forecast
    fcast_raw = lng.get_price_monthly_forecast(
        price_marker_name="Asia Spot",
        date_gte=date.today(),
        date_lte=date(2028, 6, 30)
    )
    fcast = fcast_raw.copy()
    fcast["date"] = pd.to_datetime(fcast["date"]).dt.to_period("M").dt.to_timestamp()
    fcast = fcast.rename(columns={"priceMarker": "forecast_value"})

    # 4. Merge forward + forecast
    merged = pd.merge(
        jkm[["date", "contract_label", "value", "derivative_position"]],
        fcast[["date", "forecast_value"]],
        on="date", how="inner"
    ).sort_values("date").reset_index(drop=True)
    merged["gap"] = merged["value"] - merged["forecast_value"]

    return jkm, ttf_m1, ttf_m2, ttf_asof, merged

# ─── 로딩 ────────────────────────────────────────────────────
with st.spinner("데이터 로딩 중..."):
    try:
        jkm, ttf_m1, ttf_m2, ttf_asof, merged = load_all_data(username, password)
    except Exception as e:
        st.error(f"데이터 로드 실패: {e}")
        st.stop()

as_of = pd.to_datetime(jkm["assessDate"]).max().strftime("%Y-%m-%d")

# ─── 헤더 ────────────────────────────────────────────────────
st.title("LNG Market Intelligence Dashboard")
st.caption(f"JKM Forward as of {as_of}  |  TTF as of {pd.to_datetime(ttf_asof).strftime('%Y-%m-%d')}")

# ─── KPI 카드 ─────────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)
jkm_m1   = jkm["value"].iloc[0]
jkm_m24  = jkm["value"].iloc[-1]
spd_m1   = jkm_m1 - ttf_m1
gap_m1   = merged["gap"].iloc[0]
structure = "Backwardation" if jkm_m24 < jkm_m1 else "Contango"

c1.metric("JKM M+1", f"${jkm_m1:.3f}", help="$/MMBtu")
c2.metric("JKM M+24", f"${jkm_m24:.3f}",
          delta=f"{jkm_m24 - jkm_m1:.2f}", delta_color="inverse")
c3.metric("Curve Structure", structure)
c4.metric("JKM-TTF Spread M+1", f"${spd_m1:.3f}",
          help="JKM Mo01 − TTF Mo01")
c5.metric("Fwd vs Forecast Gap M+1", f"${gap_m1:+.3f}",
          delta_color="inverse" if gap_m1 > 0 else "normal",
          help="양수=시장 과열, 음수=저평가")

st.divider()

# ─── 탭 구성 ─────────────────────────────────────────────────
tab1, tab2 = st.tabs([
    "Forward Curve  +  TTF",
    "Forward vs Forecast Gap"
])

# ══ Tab 1: Forward + TTF ══════════════════════════════════════
with tab1:
    fig1 = make_subplots(
        rows=2, cols=1,
        row_heights=[0.75, 0.25],
        shared_xaxes=True,
        vertical_spacing=0.06,
        subplot_titles=("JKM Forward Curve (M+1 ~ M+24)", "JKM − TTF Spread")
    )

    x_labels = jkm["contract_label"].tolist()
    jkm_vals = jkm["value"].tolist()

    # Backwardation 음영
    fig1.add_trace(go.Scatter(
        x=x_labels, y=jkm_vals,
        fill="tozeroy", fillcolor="rgba(59,139,212,0.08)",
        line=dict(color="rgba(0,0,0,0)"),
        showlegend=False, hoverinfo="skip"
    ), row=1, col=1)

    # JKM Forward line
    fig1.add_trace(go.Scatter(
        x=x_labels, y=jkm_vals,
        mode="lines+markers",
        line=dict(color="#E8593C", width=2.5),
        marker=dict(size=6),
        name="JKM Forward (CN06J)",
        hovertemplate="%{x}<br>JKM: $%{y:.3f}/MMBtu<extra></extra>"
    ), row=1, col=1)

    # TTF Mo01/Mo02 점
    fig1.add_trace(go.Scatter(
        x=[x_labels[0], x_labels[1]],
        y=[ttf_m1, ttf_m2],
        mode="markers+text",
        marker=dict(symbol="diamond", size=12, color="#3B8BD4"),
        text=[f"TTF {ttf_m1:.3f}", f"TTF {ttf_m2:.3f}"],
        textposition="bottom center",
        textfont=dict(color="#3B8BD4", size=10),
        name="TTF Mo01/Mo02",
        hovertemplate="%{x}<br>TTF: $%{y:.3f}/MMBtu<extra></extra>"
    ), row=1, col=1)

    # Spread 바
    spd_vals   = [jkm_vals[0] - ttf_m1, jkm_vals[1] - ttf_m2]
    spd_colors = ["#E8593C" if v >= 0 else "#3B8BD4" for v in spd_vals]
    fig1.add_trace(go.Bar(
        x=[x_labels[0], x_labels[1]],
        y=spd_vals,
        marker_color=spd_colors,
        name="JKM−TTF Spread",
        text=[f"{v:+.3f}" for v in spd_vals],
        textposition="outside",
        hovertemplate="%{x}<br>Spread: $%{y:+.3f}/MMBtu<extra></extra>"
    ), row=2, col=1)

    fig1.update_layout(
        height=600, hovermode="x unified",
        legend=dict(orientation="h", y=1.05),
        plot_bgcolor="#FAFAF8", paper_bgcolor="#FAFAF8",
        margin=dict(l=60, r=40, t=60, b=40)
    )
    fig1.update_yaxes(title_text="$/MMBtu", row=1, col=1, tickformat=".2f")
    fig1.update_yaxes(title_text="Spread $/MMBtu", row=2, col=1, tickformat=".2f")
    fig1.update_xaxes(tickangle=-35)

    st.plotly_chart(fig1, use_container_width=True)

# ══ Tab 2: Forward vs Forecast Gap ════════════════════════════
with tab2:
    fig2 = make_subplots(
        rows=2, cols=1,
        row_heights=[0.7, 0.3],
        shared_xaxes=True,
        vertical_spacing=0.06,
        subplot_titles=(
            "JKM Forward vs Asia Spot Forecast",
            "Gap (Forward − Forecast)"
        )
    )

    x2      = merged["contract_label"].tolist()
    fwd_v   = merged["value"].tolist()
    fct_v   = merged["forecast_value"].tolist()
    gap_v   = merged["gap"].tolist()

    # 과열 음영 (Forward > Forecast)
    fig2.add_trace(go.Scatter(
        x=x2, y=fwd_v,
        fill=None, line=dict(color="rgba(0,0,0,0)"),
        showlegend=False, hoverinfo="skip"
    ), row=1, col=1)
    fig2.add_trace(go.Scatter(
        x=x2, y=fct_v,
        fill="tonexty",
        fillcolor="rgba(232,89,60,0.12)",
        line=dict(color="rgba(0,0,0,0)"),
        showlegend=False, hoverinfo="skip"
    ), row=1, col=1)

    # Forward
    fig2.add_trace(go.Scatter(
        x=x2, y=fwd_v,
        mode="lines+markers",
        line=dict(color="#E8593C", width=2.5),
        marker=dict(size=5),
        name="JKM Forward (CN06J)",
        hovertemplate="%{x}<br>Forward: $%{y:.3f}<extra></extra>"
    ), row=1, col=1)

    # Forecast
    fig2.add_trace(go.Scatter(
        x=x2, y=fct_v,
        mode="lines+markers",
        line=dict(color="#1D9E75", width=2.0, dash="dash"),
        marker=dict(size=4, symbol="square"),
        name="Asia Spot Forecast (S&P)",
        hovertemplate="%{x}<br>Forecast: $%{y:.3f}<extra></extra>"
    ), row=1, col=1)

    # Gap 바
    gap_colors = ["#E8593C" if v >= 0 else "#3B8BD4" for v in gap_v]
    fig2.add_trace(go.Bar(
        x=x2, y=gap_v,
        marker_color=gap_colors,
        name="Gap (Fwd − Fcast)",
        hovertemplate="%{x}<br>Gap: $%{y:+.3f}<extra></extra>"
    ), row=2, col=1)
    fig2.add_hline(y=0, line_dash="dash",
                   line_color="#888780", row=2, col=1)

    fig2.update_layout(
        height=620, hovermode="x unified",
        legend=dict(orientation="h", y=1.05),
        plot_bgcolor="#FAFAF8", paper_bgcolor="#FAFAF8",
        margin=dict(l=60, r=40, t=60, b=40)
    )
    fig2.update_yaxes(title_text="$/MMBtu", row=1, col=1, tickformat=".2f")
    fig2.update_yaxes(title_text="Gap $/MMBtu", row=2, col=1, tickformat=".2f")
    fig2.update_xaxes(tickangle=-35)

    # Gap 신호 테이블
    st.divider()
    st.subheader("구간별 매매 시그널")
    signal_df = merged[["contract_label", "value", "forecast_value", "gap"]].copy()
    signal_df.columns = ["만기", "Forward", "Forecast", "Gap"]
    signal_df["시그널"] = signal_df["Gap"].apply(
        lambda v: "매도 (과열)" if v > 0.5 else ("매수 (저평가)" if v < -0.5 else "중립")
    )
    signal_df["Forward"]  = signal_df["Forward"].map("${:.3f}".format)
    signal_df["Forecast"] = signal_df["Forecast"].map("${:.3f}".format)
    signal_df["Gap"]      = signal_df["Gap"].map("{:+.3f}".format)

    st.dataframe(
        signal_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "시그널": st.column_config.TextColumn(width="medium")
        }
    )

    st.plotly_chart(fig2, use_container_width=True)