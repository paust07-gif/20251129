POSCO International Corp — LNG Market Insight Dashboard


Financial Data Science Final Assignment

Yonsei University Graduate School of Economics

Live Dashboard → ml-lngmarketinsight.streamlit.app




Overview

This project is a Streamlit-based LNG market intelligence dashboard built for commodity trading-oriented analysis. It integrates real-time and historical data from multiple sources to support decision-making across LNG spot trading, forward curve positioning, and arbitrage signal generation.

The dashboard is designed around the workflow of a physical LNG trading desk: monitoring benchmark prices, assessing forward curve structure, comparing market consensus forecasts against traded curves, and evaluating USGC-to-Asia netback margins.


Live Demo

Apphttps://ml-lngmarketinsight.streamlit.app/Sourcehttps://github.com/paust07-gif/20251129


Data Sources

SourceSeriesDescriptionS&P Global Commodity InsightsAAOVQ00, CN06JJKM spot price, JKM forward curve (M+1~M+24)S&P Global LNG AnalyticsAsia Spot ForecastMonthly LNG price forecastFRED (St. Louis Fed)DHHNGSPHenry Hub natural gas daily spot priceFREDDCOILBRENTEU, DCOILWTICOBrent / WTI crude oil spot priceYahoo FinanceTTF=F, EURUSD=XTTF natural gas futures, EUR/USD FX rate


When live API credentials are unavailable, the dashboard falls back to historically-calibrated sample data — including the Winter Storm Fern event (Jan 26, 2026: HH peak ~$30.57/MMBtu).




Key Features

Tab 1 — Market Overview & Coupling


Global LNG & Gas Benchmarks — JKM, TTF, Henry Hub, GCM spot price trends
JKM–TTF Spread — Daily spread with zero-line reference
Crude Oil Benchmarks — Brent / WTI price trend
JKM–TTF Spot Price Correlation by Period — 1Y / 6M / 3M / 1M correlation with period-on-period change (△) annotated on chart
JKM–TTF Coupling Summary Table — Spot price correlation with market regime interpretation (Strong Coupling / Moderate Coupling / Decoupling / Dislocated)


Tab 2 — Forward Curve, Forecast & Netback


JKM Curve Structure (M+1~M+12) — Forward curve with linear regression trend line; Contango / Backwardation / Flat regime detection
JKM Forward vs S&P Asia Spot Forecast — Market-implied vs. analyst consensus comparison
Forward – Forecast Spread Bar Chart — Monthly premium/discount visualization
TTF Forward Curve — Implied European gas forward structure
Henry Hub Forward Curve — US gas forward structure
USGC-to-Asia Netback Matrix — Heatmap across Low / Base / High freight scenarios (M+1~M+12)
Spread Summary Table — Numerical forward-forecast gap by contract month
Market Interpretation — Rule-based text summary of curve structure and spread regime



Architecture

app.py
│
├── Data Loading Layer
│   ├── load_fred_hh()       → FRED DHHNGSP (HH spot, daily)
│   ├── load_fred()          → FRED Brent / WTI
│   ├── load_ttf_hh()        → Yahoo Finance TTF=F (TTF only)
│   ├── load_spot()          → S&P Global JKM spot
│   ├── load_forward()       → S&P Global JKM forward curve
│   └── load_forecast()      → S&P Global Asia Spot Forecast
│
├── Sample Fallback Layer
│   ├── sample_spot()        → Realistic HH spike pattern (Winter Storm Fern)
│   ├── sample_forward()     → JKM forward sample
│   └── sample_forecast()    → Asia Spot Forecast sample
│
├── Analysis Layer
│   ├── structure()          → Curve structure (Contango/Backwardation/Flat)
│   ├── corr_period_chart()  → JKM-TTF spot correlation by period
│   ├── corr_summary()       → Coupling summary table
│   └── netback_matrix()     → USGC-to-Asia netback margin scenarios
│
└── Visualization Layer (Plotly)
    ├── line_chart(), spread_chart(), forward_chart()
    ├── curve_structure_chart(), comparison_chart(), spread_bar()
    └── netback_heatmap(), corr_period_chart()


HH Spot Data — Design Note

Henry Hub spot data uses a three-tier priority system:


FRED DHHNGSP (primary) — EIA daily spot, most accurate; captures Winter Storm Fern Jan 26, 2026 peak (~$30.57/MMBtu)
Yahoo Finance NG=F removed from HH pipeline — front-month futures dilute spot spikes
sample_spot() fallback — date-aware realistic pattern encoding the Jan 2026 surge when API unavailable



How to Run Locally

bashgit clone https://github.com/paust07-gif/20251129.git
cd 20251129
pip install -r requirements.txt
streamlit run app.py

API Credentials (Streamlit Secrets)

Create .streamlit/secrets.toml:

tomlSPGCI_UN = "your_sp_username"
SPGCI_PW = "your_sp_password"
FRED_API_KEY = "your_fred_api_key"

FRED API key is free at fred.stlouisfed.org.

Without credentials, the app runs in Demo mode with sample data.


Requirements

streamlit
plotly
pandas
numpy
spgci
yfinance
fredapi

