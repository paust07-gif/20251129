from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from importlib import import_module
import os
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="POSCO International Corp - LNG Market Insight", page_icon="📊", layout="wide", initial_sidebar_state="expanded")
POSCO_BLUE, NAVY, CYAN, BORDER = "#005BAC", "#12355B", "#00AEEF", "#CFE8FF"

st.markdown(f"""
<style>
.stApp {{background:linear-gradient(180deg,#F7FBFF 0%,#FFFFFF 35%,#F8FCFF 100%);color:#10233F;}}
section[data-testid="stSidebar"] {{background:linear-gradient(180deg,#FFFFFF 0%,#EEF8FF 100%);border-right:1px solid {BORDER};}}
div[data-testid="stMetric"],.chart-card,.status-card {{background:rgba(255,255,255,.96);border:1px solid {BORDER};border-radius:18px;box-shadow:0 14px 34px rgba(0,91,172,.10);}}
div[data-testid="stMetric"] {{padding:18px;min-height:118px;}}
.main-header {{padding:26px 30px;margin:0 0 22px 0;background:linear-gradient(135deg,#FFFFFF 0%,#F2FAFF 54%,#E6F5FF 100%);border:1px solid {BORDER};border-left:7px solid {POSCO_BLUE};border-radius:22px;box-shadow:0 18px 44px rgba(0,91,172,.12);}}
.main-title {{margin:0;color:{NAVY};font-size:2.18rem;font-weight:850;letter-spacing:-.025em;}}
.main-subtitle {{margin:8px 0 0 0;color:#325E8D;font-size:1.02rem;font-weight:650;}}
.last-updated {{margin:10px 0 0 0;color:#5B6B7F;font-size:.92rem;font-weight:600;}}
.chart-card {{padding:18px 18px 8px 18px;margin-bottom:18px;}}
.status-line {{display:flex;justify-content:space-between;align-items:center;padding:9px 0;border-bottom:1px solid #E4F1FC;color:{NAVY};font-size:.92rem;}}
.status-line:last-child {{border-bottom:none;}}
.sample-badge {{display:inline-flex;background:#FFF7E6;color:#8A5A00;border:1px solid #FFD58A;border-radius:999px;padding:7px 12px;font-weight:800;font-size:.86rem;margin-top:8px;}}
.interpretation-box {{background:linear-gradient(135deg,#FFFFFF 0%,#F3FAFF 100%);border:1px solid {BORDER};border-radius:18px;padding:18px 20px;box-shadow:0 12px 28px rgba(0,91,172,.08);color:#173D68;line-height:1.55;}}
</style>
""", unsafe_allow_html=True)

SPGCI_UN_NAME = "SPGCI_" + "UN"
SPGCI_PW_NAME = "SPGCI_" + "PW"
FRED_NAME = "FRED_" + "API_" + "KEY"
MMBTU_PER_MWH = 3.412141633


def get_conf(*names: str) -> str:
    for n in names:
        try:
            v = st.secrets.get(n, "")
        except Exception:
            v = ""
        v = v or os.environ.get(n, "")
        if v:
            return str(v)
    return ""


def has_spgci() -> bool:
    return bool(get_conf(SPGCI_UN_NAME) and get_conf(SPGCI_PW_NAME))


def num_col(df: pd.DataFrame, names: list[str]) -> str | None:
    for c in names:
        if c in df.columns and pd.to_numeric(df[c], errors="coerce").notna().any():
            return c
    for c in df.columns:
        if pd.to_numeric(df[c], errors="coerce").notna().sum() >= max(1, len(df) // 5):
            return c
    return None


def date_col(df: pd.DataFrame) -> str | None:
    for c in ["date", "assessDate", "assess_date", "forecastDate", "forecast_date", "month", "period"]:
        if c in df.columns and pd.to_datetime(df[c], errors="coerce").notna().any():
            return c
    return None


def build_spgci() -> tuple[Any | None, Any | None, Any | None, Any | None, str]:
    u, p = get_conf(SPGCI_UN_NAME), get_conf(SPGCI_PW_NAME)
    if not u or not p:
        return None, None, None, None, "Demo"
    try:
        spgci = import_module("spgci")
        if hasattr(spgci, "set_credentials"):
            spgci.set_credentials(u, p)
        fc = spgci.ForwardCurves() if hasattr(spgci, "ForwardCurves") else None
        lng = spgci.LNGGlobalAnalytics() if hasattr(spgci, "LNGGlobalAnalytics") else None
        md = spgci.MarketData() if hasattr(spgci, "MarketData") else None
        return fc, lng, md, spgci, "Connected"
    except Exception:
        return None, None, None, None, "Fallback"


@st.cache_data(show_spinner=False)
def sample_spot(ad: date, days: int) -> pd.DataFrame:
    d = pd.date_range(end=pd.Timestamp(ad), periods=days, freq="D")
    t = np.arange(len(d)); rng = np.random.default_rng(42)
    brent = 78 + 5.1*np.sin(t/50) + 1.8*np.cos(t/18) + rng.normal(0,.65,len(t))
    return pd.DataFrame({"date":d,"JKM":np.maximum(12.2+.9*np.sin(t/34)+.35*np.cos(t/11)+rng.normal(0,.12,len(t)),1),"TTF":np.maximum(10.9+.75*np.sin(t/38+.6)+.28*np.cos(t/15)+rng.normal(0,.10,len(t)),1),"HH":np.maximum(3.0+.22*np.sin(t/27)+rng.normal(0,.04,len(t)),.5),"GCM":np.maximum(11.6+.62*np.sin(t/41+1.1)+rng.normal(0,.11,len(t)),1),"Brent":np.maximum(brent,10),"WTI":np.maximum(brent-4.2+.7*np.sin(t/23)+rng.normal(0,.35,len(t)),10)})


@st.cache_data(show_spinner=False)
def sample_forward(ad: date) -> pd.DataFrame:
    m = pd.date_range(pd.Timestamp(ad).replace(day=1)+pd.DateOffset(months=1), periods=24, freq="MS")
    p = np.arange(1,25); c = 12.85-.34*(p-1)+.24*np.sin((p-2)/12*2*np.pi)+.05*np.cos(p/2)
    return pd.DataFrame({"derivative_position":p,"contract_label":m.strftime("%b %Y"),"date":m,"jkm_forward":c.round(3),"contract":[f"M+{i}" for i in p]})


@st.cache_data(show_spinner=False)
def sample_forecast(ad: date) -> pd.DataFrame:
    m = pd.date_range(pd.Timestamp(ad).replace(day=1)+pd.DateOffset(months=1), periods=24, freq="MS")
    p = np.arange(1,25)
    return pd.DataFrame({"date":m,"forecast_value":(12.3-.24*(p-1)+.15*np.sin(p/2.5)).round(3)})


def sample_ttf_forward(ad: date) -> pd.DataFrame:
    m = pd.date_range(pd.Timestamp(ad).replace(day=1)+pd.DateOffset(months=1), periods=24, freq="MS")
    p = np.arange(1,25)
    return pd.DataFrame({"derivative_position":p,"date":m,"ttf_forward":10.7-.18*(p-1)+.18*np.sin(p/2.4),"contract":[f"M+{i}" for i in p]})


def sample_hh_forward(ad: date) -> pd.DataFrame:
    m = pd.date_range(pd.Timestamp(ad).replace(day=1)+pd.DateOffset(months=1), periods=24, freq="MS")
    p = np.arange(1,25)
    return pd.DataFrame({"derivative_position":p,"date":m,"hh_forward":3.05+.035*(p-1)+.08*np.sin(p/3),"contract":[f"M+{i}" for i in p]})


@st.cache_data(show_spinner=False, ttl=1800)
def load_forward(ad: date, refresh: int=0) -> tuple[pd.DataFrame,bool,str]:
    fc,_,_,_,_ = build_spgci()
    if fc is None: return sample_forward(ad), True, "Demo"
    try:
        raw = pd.DataFrame(fc.get_assessments(curve_code="CN06J", derivative_maturity_frequency="Month"))
        if "bate" in raw.columns: raw = raw[raw["bate"].astype(str).str.lower().eq("c")]
        if "derivative_position" not in raw.columns: raw["derivative_position"] = np.arange(1,len(raw)+1)
        raw = raw.sort_values([c for c in ["derivative_position","assessDate"] if c in raw.columns]).groupby("derivative_position",as_index=False).last()
        raw = raw[(raw["derivative_position"]>=1)&(raw["derivative_position"]<=24)].copy()
        if "contract_label" in raw.columns: raw["date"] = pd.to_datetime("1 "+raw["contract_label"].astype(str), errors="coerce").dt.to_period("M").dt.to_timestamp()
        if "date" not in raw.columns or raw["date"].isna().all(): raw["date"] = pd.date_range(pd.Timestamp(ad).replace(day=1)+pd.DateOffset(months=1), periods=len(raw), freq="MS")
        v = num_col(raw,["value","price","assessment","close","settlement","mid"])
        out = raw[["derivative_position","date",v]].copy(); out["jkm_forward"] = pd.to_numeric(out[v], errors="coerce")
        out = out.dropna(subset=["jkm_forward","date"]).sort_values("derivative_position")
        out["contract_label"] = out["date"].dt.strftime("%b %Y"); out["contract"] = "M+"+out["derivative_position"].astype(int).astype(str)
        return out[["derivative_position","contract_label","date","jkm_forward","contract"]], False, "Connected"
    except Exception:
        return sample_forward(ad), True, "Fallback"


@st.cache_data(show_spinner=False, ttl=1800)
def load_spot(ad: date, days: int, refresh: int=0) -> tuple[pd.DataFrame|None,bool,str]:
    _,_,md,spgci,_ = build_spgci(); client = md if md is not None else spgci
    if client is None or not hasattr(client,"get_assessments_by_symbol_historical"): return None, True, "Demo"
    symbol="AAOVQ00"; start = pd.Timestamp(ad).date()-timedelta(days=days+30)
    variants=[{"symbol":[symbol],"paginate":True},{"symbol":symbol,"paginate":True},{"symbol":[symbol],"bate":"c","paginate":True},{"symbol":[symbol],"assess_date_gte":start,"assess_date_lte":ad,"paginate":True},{"symbols":[symbol],"paginate":True}]
    for kw in variants:
        try:
            raw = pd.DataFrame(client.get_assessments_by_symbol_historical(**kw))
            if raw.empty: continue
            if "bate" in raw.columns:
                c = raw[raw["bate"].astype(str).str.lower().eq("c")]
                if not c.empty: raw = c
            dc = date_col(raw); vc = num_col(raw,["value","price","assessment","close","settlement","mid"])
            if dc is None or vc is None: continue
            raw["date"] = pd.to_datetime(raw[dc], errors="coerce").dt.tz_localize(None); raw["JKM"] = pd.to_numeric(raw[vc], errors="coerce")
            out = raw.dropna(subset=["date","JKM"])[["date","JKM"]]
            out = out[(out["date"]>=pd.Timestamp(start))&(out["date"]<=pd.Timestamp(ad))].sort_values("date").groupby("date",as_index=False).last().tail(days)
            if not out.empty: return out, False, "Connected"
        except Exception: pass
    return None, True, "Fallback"


@st.cache_data(show_spinner=False, ttl=1800)
def load_forecast(ad: date, refresh: int=0) -> tuple[pd.DataFrame,bool,str]:
    _,lng,_,spgci,_ = build_spgci(); client = lng if lng is not None else spgci
    if client is None or not hasattr(client,"get_price_monthly_forecast"): return sample_forecast(ad), True, "Demo"
    variants=[{"price_marker_name":"Asia Spot"},{"price_marker_name":"Asia Spot","date_gte":date.today(),"date_lte":date(2028,6,30)},{"price_marker_name":"JKM"},{"price_marker_name":"Japan Korea Marker"},{"priceMarkerName":"Asia Spot"}]
    for kw in variants:
        try:
            raw = pd.DataFrame(client.get_price_monthly_forecast(**kw))
            if raw.empty: continue
            dc = date_col(raw); vc = num_col(raw,["forecast_value","price","value","forecast","forecastValue","priceValue","assessment","mean"])
            if vc is None: continue
            if dc is not None: raw["date"] = pd.to_datetime(raw[dc], errors="coerce").dt.to_period("M").dt.to_timestamp()
            elif "derivative_position" in raw.columns:
                raw = raw.sort_values("derivative_position"); raw["date"] = pd.date_range(pd.Timestamp(ad).replace(day=1)+pd.DateOffset(months=1), periods=len(raw), freq="MS")
            else: raw["date"] = pd.date_range(pd.Timestamp(ad).replace(day=1)+pd.DateOffset(months=1), periods=len(raw), freq="MS")
            raw["forecast_value"] = pd.to_numeric(raw[vc], errors="coerce")
            out = raw.dropna(subset=["date","forecast_value"]).groupby("date",as_index=False).last().sort_values("date")
            out = out[out["date"]>=pd.Timestamp(ad).replace(day=1)].head(24)
            if not out.empty: return out[["date","forecast_value"]], False, "Connected"
        except Exception: pass
    return sample_forecast(ad), True, "Fallback"


@st.cache_data(show_spinner=False, ttl=1800)
def load_fred(ad: date, days: int) -> tuple[pd.DataFrame|None,str]:
    key = get_conf(FRED_NAME)
    if not key: return None, "Demo"
    try:
        fredapi = import_module("fredapi"); fred = fredapi.Fred(api_key=key)
        start = pd.Timestamp(ad)-pd.Timedelta(days=days+21); end = pd.Timestamp(ad)
        b = fred.get_series("DCOILBRENTEU", observation_start=start.date(), observation_end=end.date()); w = fred.get_series("DCOILWTICO", observation_start=start.date(), observation_end=end.date())
        df = pd.DataFrame({"date":b.index,"Brent":b.values}).merge(pd.DataFrame({"date":w.index,"WTI":w.values}), on="date", how="outer")
        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None); df[["Brent","WTI"]] = df[["Brent","WTI"]].apply(pd.to_numeric, errors="coerce")
        df = df.dropna(how="all", subset=["Brent","WTI"]).sort_values("date").tail(days)
        return (df,"Connected") if not df.empty else (None,"Fallback")
    except Exception: return None, "Fallback"


@st.cache_data(show_spinner=False, ttl=1800)
def load_yahoo(tickers: dict[str,str], ad: date, days: int) -> tuple[pd.DataFrame|None,str]:
    try:
        yf = import_module("yfinance"); start = pd.Timestamp(ad)-pd.Timedelta(days=days+14); end = pd.Timestamp(ad)+pd.Timedelta(days=1); frames=[]
        for label,ticker in tickers.items():
            hist = yf.download(ticker,start=start.date(),end=end.date(),progress=False,auto_adjust=True)
            if hist.empty: continue
            close = hist["Close"]
            if isinstance(close,pd.DataFrame): close = close.iloc[:,0]
            frames.append(close.rename(label).reset_index().rename(columns={"Date":"date","index":"date"}))
        if not frames: return None,"Fallback"
        merged=frames[0]
        for f in frames[1:]: merged=pd.merge(merged,f,on="date",how="outer")
        merged["date"]=pd.to_datetime(merged["date"]).dt.tz_localize(None)
        return merged.sort_values("date").tail(days),"Connected"
    except Exception: return None,"Fallback"


@st.cache_data(show_spinner=False, ttl=1800)
def load_ttf_hh(ad: date, days: int) -> tuple[pd.DataFrame|None,str]:
    try:
        raw, status = load_yahoo({"TTF_EUR_MWH":"TTF=F", "HH":"NG=F", "EURUSD":"EURUSD=X"}, ad, days)
        if raw is None or raw.empty or "TTF_EUR_MWH" not in raw.columns:
            return None, "Fallback"
        raw = raw.sort_values("date")
        if "EURUSD" not in raw.columns:
            raw["EURUSD"] = 1.08
        raw[["TTF_EUR_MWH", "HH", "EURUSD"]] = raw[["TTF_EUR_MWH", "HH", "EURUSD"]].ffill().bfill()
        raw["TTF"] = raw["TTF_EUR_MWH"] * raw["EURUSD"] / MMBTU_PER_MWH
        out = raw[["date", "TTF", "HH"]].dropna(how="all", subset=["TTF", "HH"])
        if out.empty:
            return None, "Fallback"
        return out.tail(days), "Connected"
    except Exception:
        return None, "Fallback"


def status_label(s: str) -> str:
    return "🟢 Connected" if s == "Connected" else "🟡 Fallback"


def layout(fig: go.Figure, title: str, y_title: str|None=None) -> go.Figure:
    fig.update_layout(title={"text":title,"font":{"color":NAVY,"size":18}}, hovermode="x unified", template="plotly_white", plot_bgcolor="white", paper_bgcolor="white", margin=dict(l=24,r=24,t=58,b=30), legend=dict(orientation="h",yanchor="bottom",y=1.02,xanchor="right",x=1), font=dict(color="#1E3552"))
    fig.update_xaxes(showgrid=True, gridcolor="#E6F2FF", zeroline=False); fig.update_yaxes(showgrid=True, gridcolor="#E6F2FF", zeroline=False, title=y_title)
    return fig


def render(fig: go.Figure) -> None:
    st.markdown('<div class="chart-card">', unsafe_allow_html=True); st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False}); st.markdown('</div>', unsafe_allow_html=True)


def line_chart(df: pd.DataFrame, cols: list[str], title: str, unit: str, colors: list[str]) -> go.Figure:
    fig=go.Figure()
    for col,color in zip(cols,colors):
        if col in df.columns: fig.add_trace(go.Scatter(x=df["date"], y=df[col], mode="lines+markers", name=col, line=dict(color=color,width=2.7), marker=dict(size=4), hovertemplate=f"<b>{col}</b><br>Date: %{{x|%Y-%m-%d}}<br>Price: %{{y:.2f}} {unit}<extra></extra>"))
    return layout(fig,title,unit)


def forward_chart(df: pd.DataFrame, val: str, name: str, title: str, color: str) -> go.Figure:
    custom=np.stack([df["contract"],df["date"].dt.strftime("%b %Y")],axis=-1); fig=go.Figure(); fig.add_trace(go.Scatter(x=df["date"], y=df[val], customdata=custom, mode="lines+markers", name=name, line=dict(color=color,width=3), marker=dict(size=7,color=color), hovertemplate="Contract: %{customdata[0]}<br>Month: %{customdata[1]}<br>"+f"{name}: %{{y:.2f}} $/MMBtu<extra></extra>")); return layout(fig,title,"$/MMBtu")


def structure(jkm: pd.DataFrame) -> tuple[float,str,float]:
    d=jkm.sort_values("derivative_position").head(12)
    if len(d)<12: return 0.0,"FLAT",0.0
    spread=float(d.loc[d["derivative_position"]==12,"jkm_forward"].iloc[0]-d.loc[d["derivative_position"]==1,"jkm_forward"].iloc[0]); name="CONTANGO" if spread>.10 else "BACKWARDATION" if spread<-.10 else "FLAT"; slope=float(np.polyfit(np.asarray(d["derivative_position"],dtype=float),np.asarray(d["jkm_forward"],dtype=float),1)[0]); return spread,name,slope


def spread_chart(df: pd.DataFrame) -> go.Figure:
    d=df.copy(); d["JKM - TTF Spread"]=d["JKM"]-d["TTF"]; fig=go.Figure(); fig.add_trace(go.Scatter(x=d["date"],y=d["JKM - TTF Spread"],mode="lines+markers",name="JKM - TTF Spread",line=dict(color=POSCO_BLUE,width=2.7),marker=dict(size=4),hovertemplate="Date: %{x|%Y-%m-%d}<br>Spread: %{y:+.2f} $/MMBtu<extra></extra>")); fig.add_hline(y=0,line_width=1,line_dash="dash",line_color="#6B7C93"); return layout(fig,"JKM - TTF Spread","$/MMBtu")


def curve_structure_chart(jkm: pd.DataFrame) -> go.Figure:
    d=jkm.sort_values("derivative_position").head(12).copy(); slope,intercept=np.polyfit(np.asarray(d["derivative_position"],dtype=float),np.asarray(d["jkm_forward"],dtype=float),1); d["trend"]=intercept+slope*d["derivative_position"]; fig=go.Figure(); fig.add_trace(go.Scatter(x=d["derivative_position"],y=d["jkm_forward"],mode="lines+markers",name="JKM Forward",line=dict(color=POSCO_BLUE,width=3),hovertemplate="Contract: M+%{x}<br>JKM Forward: %{y:.2f} $/MMBtu<extra></extra>")); fig.add_trace(go.Scatter(x=d["derivative_position"],y=d["trend"],mode="lines",name="Linear Trend",line=dict(color=CYAN,width=2.5,dash="dash"),hovertemplate="Contract: M+%{x}<br>Trend: %{y:.2f} $/MMBtu<extra></extra>")); return layout(fig,"JKM Curve Structure: M+1 to M+12","$/MMBtu")


def netback_matrix(jkm: pd.DataFrame, hh: pd.DataFrame) -> pd.DataFrame:
    base=pd.merge(jkm[["date","contract","jkm_forward"]],hh[["date","hh_forward"]],on="date",how="inner"); m=pd.DataFrame({"Contract":base["contract"],"Month":base["date"].dt.strftime("%b %Y")})
    for case,fr in {"Low Freight":1.05,"Base Freight":1.45,"High Freight":1.90}.items(): m[case]=base["jkm_forward"]-(base["hh_forward"]*1.15+2.35+fr)
    return m.head(12)


def netback_heatmap(m: pd.DataFrame) -> go.Figure:
    cols=["Low Freight","Base Freight","High Freight"]; fig=go.Figure(data=go.Heatmap(z=m[cols].T.values,x=m["Contract"]+" | "+m["Month"],y=cols,colorscale=[[0,"#FFECE7"],[0.5,"#FFFFFF"],[1,"#A7E8FF"]],zmid=0,hovertemplate="Contract: %{x}<br>Scenario: %{y}<br>USGC Netback Margin: %{z:+.2f} $/MMBtu<extra></extra>",colorbar=dict(title="$/MMBtu"))); return layout(fig,"USGC-to-Asia Netback Matrix","")


def comparison_chart(c: pd.DataFrame) -> go.Figure:
    d=c.copy(); d["month_label"]=d["date"].dt.strftime("%b %Y"); custom=np.stack([d["month_label"],d["jkm_forward"],d["forecast_value"],d["spread"],d["spread_pct"]],axis=-1); fig=go.Figure(); fig.add_trace(go.Scatter(x=d["date"],y=d["jkm_forward"],customdata=custom,mode="lines+markers",name="JKM Forward",line=dict(color=POSCO_BLUE,width=3),hovertemplate="Month: %{customdata[0]}<br>JKM Forward: %{customdata[1]:.2f} $/MMBtu<br>S&P Forecast: %{customdata[2]:.2f} $/MMBtu<br>Spread: %{customdata[3]:+.2f} $/MMBtu<br>Spread %: %{customdata[4]:+.2f}%<extra></extra>")); fig.add_trace(go.Scatter(x=d["date"],y=d["forecast_value"],customdata=custom,mode="lines+markers",name="S&P Asia Spot Forecast",line=dict(color=CYAN,width=3,dash="dash"),hovertemplate="Month: %{customdata[0]}<br>Forecast: %{customdata[2]:.2f} $/MMBtu<br>Spread: %{customdata[3]:+.2f} $/MMBtu<br>Spread %: %{customdata[4]:+.2f}%<extra></extra>")); return layout(fig,"JKM Forward Curve vs S&P Asia Spot Forecast","$/MMBtu")


def spread_bar(c: pd.DataFrame) -> go.Figure:
    d=c.copy(); colors=np.where(d["spread"]>=0,POSCO_BLUE,"#FF8A65"); fig=go.Figure(); fig.add_trace(go.Bar(x=d["date"],y=d["spread"],customdata=d["date"].dt.strftime("%b %Y"),name="Forward - Forecast Spread",marker_color=colors,hovertemplate="Month: %{customdata}<br>Forward - Forecast Spread: %{y:+.2f} $/MMBtu<extra></extra>")); fig.add_hline(y=0,line_width=1,line_dash="dash",line_color="#6B7C93"); return layout(fig,"Forward - Forecast Spread","$/MMBtu")


def rolling_corr(df: pd.DataFrame, window: int = 30) -> pd.DataFrame:
    d=df[["date","JKM","TTF"]].dropna().sort_values("date").copy(); d["JKM_return"]=d["JKM"].pct_change(); d["TTF_return"]=d["TTF"].pct_change(); d["Rolling Correlation"]=d["JKM_return"].rolling(window).corr(d["TTF_return"]); d["Change"] = d["Rolling Correlation"].diff(); return d.dropna(subset=["Rolling Correlation"])


def corr_line_chart(corr: pd.DataFrame, corr_label: str) -> go.Figure:
    fig=go.Figure(); fig.add_trace(go.Scatter(x=corr["date"], y=corr["Rolling Correlation"], mode="lines+markers", name=f"{corr_label} Return Correlation", line=dict(color=POSCO_BLUE,width=2.8), marker=dict(size=4), hovertemplate="Date: %{x|%Y-%m-%d}<br>Return correlation: %{y:.2f}<extra></extra>")); fig.add_trace(go.Scatter(x=corr["date"], y=corr["Change"], mode="lines", name="Change", line=dict(color=CYAN,width=2,dash="dot"), hovertemplate="Date: %{x|%Y-%m-%d}<br>Change: %{y:+.3f}<extra></extra>")); fig.add_hline(y=0,line_width=1,line_dash="dash",line_color="#6B7C93"); return layout(fig,"JKM-TTF Rolling Return Correlation & Change","Return correlation / change")


def corr_summary(df: pd.DataFrame, ad: date) -> pd.DataFrame:
    rows=[]; end=pd.Timestamp(ad)
    for label,days in [("1 Year",365),("6 Months",183),("3 Months",92),("1 Month",31)]:
        w=df[df["date"]>=end-pd.Timedelta(days=days)][["JKM","TTF"]].dropna().pct_change().dropna(); corr=w["JKM"].corr(w["TTF"]) if len(w)>=3 else np.nan; interp="Insufficient Data" if pd.isna(corr) else "Strong Coupling" if corr>=.70 else "Moderate Coupling" if corr>=.40 else "Decoupling" if corr>=.10 else "Dislocated"; rows.append({"Period":label,"Return Correlation":corr,"Interpretation":interp})
    return pd.DataFrame(rows)


with st.sidebar:
    st.markdown("### Market Controls")
    analysis_date=st.date_input("Analysis Date",value=date.today(),max_value=date.today()+timedelta(days=365))
    lookback_label=st.selectbox("Lookback Period",["1 Year","6 Months","3 Months","1 Month"],index=0)
    lookback_days={"1 Year":365,"6 Months":183,"3 Months":92,"1 Month":31}[lookback_label]
    corr_window=st.selectbox("Correlation Window",["30D","60D","90D"],index=0)
    corr_days={"30D":30,"60D":60,"90D":90}[corr_window]
    if "refresh_key" not in st.session_state: st.session_state["refresh_key"]=0
    if st.button("Refresh Data",width="stretch"):
        st.session_state["refresh_key"]+=1; load_spot.clear(); load_forward.clear(); load_forecast.clear(); load_fred.clear(); load_yahoo.clear(); load_ttf_hh.clear()
    st.divider(); st.markdown("### Data Source Mode"); st.success("Live mode: S&P configured") if has_spgci() else st.warning("Demo mode: S&P not configured"); st.caption("Inputs are hidden and loaded from Streamlit settings.")

refresh=st.session_state.get("refresh_key",0)
jkm_forward,jkm_sample,jkm_status=load_forward(analysis_date,refresh); fcast_curve,fcast_sample,fcast_status=load_forecast(analysis_date,refresh); spot_df=sample_spot(analysis_date,max(lookback_days,365)); spgci_spot_df,spot_sample,spot_status=load_spot(analysis_date,lookback_days,refresh)
if spgci_spot_df is not None and "JKM" in spgci_spot_df.columns:
    spot_df=spot_df.drop(columns=["JKM"],errors="ignore").merge(spgci_spot_df[["date","JKM"]],on="date",how="left"); spot_df["JKM"]=spot_df["JKM"].ffill().bfill()
ttf_forward=sample_ttf_forward(analysis_date); hh_forward=sample_hh_forward(analysis_date); fred_df,fred_status=load_fred(analysis_date,lookback_days); yahoo_df,yahoo_status=load_yahoo({"Brent":"BZ=F","WTI":"CL=F"},analysis_date,lookback_days); gas_df,gas_status=load_ttf_hh(analysis_date,lookback_days); oil_df=fred_df if fred_df is not None else yahoo_df
if gas_df is not None and {"TTF","HH"}.issubset(gas_df.columns):
    spot_df=spot_df.drop(columns=["TTF","HH"],errors="ignore").merge(gas_df[["date","TTF","HH"]],on="date",how="left"); spot_df[["TTF","HH"]]=spot_df[["TTF","HH"]].ffill().bfill()
if oil_df is not None and {"Brent","WTI"}.issubset(oil_df.columns):
    overlay=oil_df[["date","Brent","WTI"]].dropna(how="all",subset=["Brent","WTI"])
    if not overlay.empty:
        spot_df=spot_df.drop(columns=["Brent","WTI"],errors="ignore").merge(overlay,on="date",how="left"); spot_df[["Brent","WTI"]]=spot_df[["Brent","WTI"]].ffill().bfill()
comparison=pd.merge(jkm_forward,fcast_curve,on="date",how="inner")
if comparison.empty:
    comparison=pd.merge(sample_forward(analysis_date),sample_forecast(analysis_date),on="date",how="inner"); jkm_sample=True; fcast_sample=True
comparison["spread"]=comparison["jkm_forward"]-comparison["forecast_value"]; comparison["spread_pct"]=comparison["spread"]/comparison["forecast_value"]*100
structure_spread,market_structure,slope=structure(jkm_forward); latest_spot=float(spot_df["JKM"].iloc[-1]); latest_hh=float(spot_df["HH"].iloc[-1]); latest_jkm_forward=float(jkm_forward.sort_values("derivative_position")["jkm_forward"].iloc[0]); usgc_margin=latest_jkm_forward-(latest_hh*1.15+2.35+1.45); arb_signal="OPEN" if usgc_margin>.75 else "WATCH" if usgc_margin>.15 else "CLOSED"
corr_data=rolling_corr(spot_df,corr_days); latest_corr=float(corr_data["Rolling Correlation"].iloc[-1]) if not corr_data.empty else float("nan"); corr_change=float(corr_data["Change"].iloc[-1]) if not corr_data.empty else float("nan")

with st.sidebar:
    st.divider(); st.markdown("### Data Source Status"); statuses={"S&P Global Spot":spot_status,"S&P Global Forward":jkm_status,"S&P Global Forecast":fcast_status,"Yahoo TTF/HH":gas_status,"FRED Oil":fred_status,"Yahoo Oil":yahoo_status}; st.markdown('<div class="status-card" style="padding:12px 14px;">',unsafe_allow_html=True)
    for source,status in statuses.items(): st.markdown(f'<div class="status-line"><span>{source}</span><strong>{status_label(status)}</strong></div>',unsafe_allow_html=True)
    st.markdown("</div>",unsafe_allow_html=True)
    if spot_sample or jkm_sample or fcast_sample or gas_df is None or (fred_df is None and yahoo_df is None): st.markdown('<div class="sample-badge">Sample / Estimated Data</div>',unsafe_allow_html=True)

kst=timezone(timedelta(hours=9)); st.markdown(f"""<div class="main-header"><h1 class="main-title">POSCO International Corp - LNG Market Insight</h1><p class="main-subtitle">Global LNG Market Intelligence Dashboard | Spot, Forward Curve, Forecast Gap & Arbitrage Signal</p><p class="last-updated">Last Updated: {datetime.now(kst).strftime('%Y-%m-%d %H:%M')} KST</p></div>""",unsafe_allow_html=True)
if spot_sample or jkm_sample or fcast_sample or gas_df is None: st.warning("Sample / Estimated Data: 일부 가스 벤치마크가 연결되지 않아 샘플/추정 데이터를 사용 중입니다.")
cols=st.columns(4); cols[0].metric("JKM Spot",f"{latest_spot:.2f} $/MMBtu",delta=f"M+1 {latest_jkm_forward:.2f}"); cols[1].metric("USGC Margin",f"{usgc_margin:.2f} $/MMBtu",delta="Netback to Asia"); cols[2].metric("Arbitrage Signal",arb_signal,delta="USGC → Asia"); cols[3].metric("JKM-TTF Return Corr",f"{latest_corr:.2f}",delta=f"{corr_change:+.3f} d/d")

tab1,tab2=st.tabs(["Market Overview & Coupling", "Forward Curve, Forecast & Netback"])
with tab1:
    spot_window=spot_df.tail(lookback_days).copy(); left,right=st.columns(2)
    with left: render(line_chart(spot_window,["JKM","TTF","HH","GCM"],"Global LNG & Gas Benchmarks", "$/MMBtu",[POSCO_BLUE,CYAN,"#56B870","#7B61FF"])); render(spread_chart(spot_window))
    with right: render(line_chart(spot_window,["Brent","WTI"],"Crude Oil Benchmark Price Trend","$/bbl",[NAVY,"#4AA3FF"])); render(corr_line_chart(corr_data,corr_window)); st.markdown("### JKM-TTF Coupling Summary"); st.dataframe(corr_summary(spot_df,analysis_date).style.format({"Return Correlation":"{:.2f}"}),use_container_width=True,hide_index=True)
with tab2:
    left,right=st.columns(2)
    with left: render(forward_chart(jkm_forward,"jkm_forward","JKM Forward","JKM Forward Curve",POSCO_BLUE)); render(comparison_chart(comparison)); render(spread_bar(comparison))
    with right: render(forward_chart(ttf_forward,"ttf_forward","TTF Forward","TTF Forward Curve / Implied Curve",CYAN)); render(forward_chart(hh_forward,"hh_forward","Henry Hub Forward","Henry Hub Forward Curve","#56B870")); render(curve_structure_chart(jkm_forward))
    scols=st.columns(3); scols[0].metric("M+12 - M+1",f"{structure_spread:+.2f} $/MMBtu"); scols[1].metric("Contango / Backwardation",market_structure); scols[2].metric("Linear Regression Slope",f"{slope:+.2f} $/mo")
    matrix=netback_matrix(jkm_forward,hh_forward); render(netback_heatmap(matrix)); st.dataframe(matrix.style.format({"Low Freight":"{:+.2f}","Base Freight":"{:+.2f}","High Freight":"{:+.2f}"}),use_container_width=True,hide_index=True)
    table=comparison[["date","contract","jkm_forward","forecast_value","spread","spread_pct"]].copy(); table["Month"]=table["date"].dt.strftime("%b %Y"); table=table[["contract","Month","jkm_forward","forecast_value","spread","spread_pct"]]; table.columns=["Contract","Month","JKM Forward","S&P Forecast","Spread","Spread %"]; st.markdown("### Spread Summary Table"); st.dataframe(table.style.format({"JKM Forward":"{:.2f}","S&P Forecast":"{:.2f}","Spread":"{:+.2f}","Spread %":"{:+.2f}%"}),use_container_width=True,hide_index=True)
    avg=float(comparison["spread"].mean()); max_abs=comparison.iloc[comparison["spread"].abs().argmax()]; read="Forward curve is pricing a premium versus S&P Asia Spot Forecast, suggesting near-term risk premium." if avg>.25 else "Forward curve is discounting S&P Asia Spot Forecast, implying softer market expectations." if avg<-.25 else "Forward curve is broadly aligned with S&P Asia Spot Forecast."
    st.markdown(f"""<div class="interpretation-box"><strong>Market Interpretation</strong><br>Average Forward - Forecast spread is <strong>{avg:+.2f} $/MMBtu</strong>. {read}<br>Largest absolute gap appears in <strong>{max_abs['date'].strftime('%b %Y')}</strong> at <strong>{max_abs['spread']:+.2f} $/MMBtu</strong> ({max_abs['spread_pct']:+.2f}%).<br>Current curve structure is <strong>{market_structure}</strong> with M+12 - M+1 at <strong>{structure_spread:+.2f} $/MMBtu</strong> and regression slope of <strong>{slope:+.2f} $/mo</strong>.</div>""",unsafe_allow_html=True)
