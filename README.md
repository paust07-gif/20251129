# POSCO International Corp - LNG Market Insight

This project is a Streamlit-based LNG market analysis dashboard developed for a machine learning / financial data science final assignment.

## Project Purpose

The dashboard supports LNG market monitoring and trading-oriented analysis by integrating LNG price indicators, forward curves, macroeconomic data, and market interpretation logic.

## Main Features

- JKM forward curve monitoring
- S&P Global Commodity Insights data loading
- FRED macroeconomic data loading
- Brent / WTI market data loading
- LNG spot, forward, and forecast comparison
- Spread and curve structure analysis
- Interactive Plotly charts with hover tooltips
- Rule-based market interpretation
- Sample-data fallback when live API credentials are unavailable

## How to Run

```bash
pip install -r requirements.txt
python -m streamlit run app.py
