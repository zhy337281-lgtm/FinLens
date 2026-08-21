from __future__ import annotations

import hashlib
import hmac
import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import yfinance as yf
from dotenv import load_dotenv

from analytics import (
    dcf_valuation,
    daily_returns,
    expected_asset_returns,
    historical_stress,
    monte_carlo_terminal_values,
    normalize_weights,
    portfolio_metrics,
    risk_contributions,
    terminal_summary,
)

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - surfaced in the UI
    OpenAI = None


APP_DIR = Path(__file__).resolve().parent
load_dotenv(APP_DIR / ".env.local")
load_dotenv(APP_DIR / ".env")

st.set_page_config(
    page_title="FinLens | Private Equity Research",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)


STYLE = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@500&display=swap');
:root { --ink:#e8edf5; --muted:#8f9bad; --panel:#101722; --line:#223044; --accent:#46d6a0; }
.stApp { background: #080d14; color: var(--ink); }
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
[data-testid="stSidebar"] { background: #0c121b; border-right: 1px solid var(--line); }
[data-testid="stMetric"] { background: var(--panel); border: 1px solid var(--line); padding: 16px; border-radius: 12px; }
[data-testid="stMetricLabel"] { color: var(--muted); }
[data-testid="stMetricValue"] { font-family: 'IBM Plex Mono', monospace; }
.block-container { padding-top: 2rem; padding-bottom: 4rem; max-width: 1500px; }
h1, h2, h3 { letter-spacing: -0.025em; }
.eyebrow { color: var(--accent); font: 500 0.78rem 'IBM Plex Mono', monospace; letter-spacing: .14em; text-transform: uppercase; }
.hero { padding: 26px 0 18px; border-bottom: 1px solid var(--line); margin-bottom: 25px; }
.hero h1 { font-size: 3rem; margin: .25rem 0 .4rem; }
.hero p { color: var(--muted); max-width: 820px; font-size: 1.05rem; }
.panel { background: var(--panel); border: 1px solid var(--line); border-radius: 14px; padding: 20px; height: 100%; }
.panel h3 { margin-top: 0; }
.pill { display:inline-block; border:1px solid #2c8064; color:#72e0b5; border-radius:999px; padding:4px 10px; font-size:.78rem; margin-right:6px; }
.muted { color: var(--muted); }
.status { display:flex; align-items:center; gap:8px; color:#9ba8ba; font-size:.88rem; }
.status-dot { width:7px; height:7px; background:var(--accent); border-radius:50%; box-shadow:0 0 12px var(--accent); }
.stButton > button { border-radius: 9px; font-weight: 600; }
div[data-testid="stDataFrame"] { border: 1px solid var(--line); border-radius: 10px; overflow: hidden; }
.disclaimer { border-left: 3px solid #805f2c; background:#18140d; color:#b9ad96; padding:12px 14px; border-radius:4px; font-size:.82rem; }
</style>
"""
st.markdown(STYLE, unsafe_allow_html=True)


def secret_value(name: str) -> str | None:
    value = os.getenv(name)
    if value:
        return str(value)
    try:
        value = st.secrets.get(name)
        return str(value) if value else None
    except Exception:
        return None


def require_authentication() -> None:
    configured_password = secret_value("FINLENS_PASSWORD")
    if not configured_password:
        st.error("Private access is not configured. Set `FINLENS_PASSWORD` in the deployment secrets.")
        st.info("The application is fail-closed: research data is unavailable until a password is configured.")
        st.stop()

    expected = hashlib.sha256(configured_password.encode("utf-8")).digest()
    if st.session_state.get("authenticated"):
        return

    st.markdown(
        """
        <div class="hero">
          <div class="eyebrow">Private research system</div>
          <h1>FinLens</h1>
          <p>Personal equity research, valuation and portfolio intelligence.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    candidate = st.text_input("Access password", type="password", placeholder="Enter your private password")
    if st.button("Unlock workspace", type="primary", width="stretch"):
        supplied = hashlib.sha256(candidate.encode("utf-8")).digest()
        if hmac.compare_digest(supplied, expected):
            st.session_state["authenticated"] = True
            st.rerun()
        else:
            st.error("Incorrect password.")
    st.stop()


def valid_number(value) -> bool:
    try:
        return value is not None and bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def money(value, decimals: int = 2) -> str:
    if not valid_number(value):
        return "—"
    value = float(value)
    sign = "-" if value < 0 else ""
    absolute = abs(value)
    for threshold, suffix in ((1e12, "T"), (1e9, "B"), (1e6, "M")):
        if absolute >= threshold:
            return f"{sign}${absolute / threshold:.{decimals}f}{suffix}"
    return f"{sign}${absolute:,.{decimals}f}"


def percent(value, decimals: int = 1) -> str:
    return f"{float(value) * 100:.{decimals}f}%" if valid_number(value) else "—"


def multiple(value) -> str:
    return f"{float(value):.1f}×" if valid_number(value) else "—"


def safe(info: dict, key: str, default=None):
    value = info.get(key, default)
    if value is None or value == "":
        return default
    try:
        return default if pd.isna(value) else value
    except (TypeError, ValueError):
        return value


@st.cache_data(ttl=900, show_spinner=False)
def company_bundle(ticker: str):
    stock = yf.Ticker(ticker)
    try:
        info = stock.info or {}
    except Exception:
        info = {}
    frames = []
    for attribute in ("financials", "balance_sheet", "cashflow"):
        try:
            frame = getattr(stock, attribute)
            frames.append(frame if frame is not None else pd.DataFrame())
        except Exception:
            frames.append(pd.DataFrame())
    return info, frames[0], frames[1], frames[2]


@st.cache_data(ttl=600, show_spinner=False)
def price_history(tickers: tuple[str, ...], period: str = "5y") -> pd.DataFrame:
    symbols = list(dict.fromkeys(ticker.upper() for ticker in tickers if ticker))
    if not symbols:
        return pd.DataFrame()
    raw = yf.download(symbols, period=period, auto_adjust=True, progress=False, group_by="column", threads=True)
    if raw.empty:
        return pd.DataFrame()
    if isinstance(raw.columns, pd.MultiIndex):
        if "Close" in raw.columns.get_level_values(0):
            close = raw["Close"]
        elif "Close" in raw.columns.get_level_values(1):
            close = raw.xs("Close", axis=1, level=1)
        else:
            return pd.DataFrame()
    else:
        close = raw[["Close"]].rename(columns={"Close": symbols[0]})
    if isinstance(close, pd.Series):
        close = close.to_frame(symbols[0])
    close.columns = [str(column).upper() for column in close.columns]
    return close.sort_index().ffill().dropna(how="all")


def statement_value(frame: pd.DataFrame, rows: list[str]):
    if frame is None or frame.empty:
        return None
    for row in rows:
        if row in frame.index:
            values = pd.to_numeric(frame.loc[row], errors="coerce").dropna()
            if not values.empty:
                return float(values.iloc[0])
    return None


@st.cache_data(ttl=900, show_spinner=False)
def peer_snapshot(tickers: tuple[str, ...]) -> pd.DataFrame:
    rows = []
    for ticker in tickers:
        try:
            info = yf.Ticker(ticker).info or {}
            rows.append(
                {
                    "Ticker": ticker,
                    "Company": safe(info, "shortName", ticker),
                    "Market cap": safe(info, "marketCap"),
                    "Forward P/E": safe(info, "forwardPE"),
                    "EV/EBITDA": safe(info, "enterpriseToEbitda"),
                    "Revenue growth": safe(info, "revenueGrowth"),
                    "EBITDA margin": safe(info, "ebitdaMargins"),
                    "ROE": safe(info, "returnOnEquity"),
                }
            )
        except Exception:
            continue
    return pd.DataFrame(rows)


def display_peer_table(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    if result.empty:
        return result
    result["Market cap"] = result["Market cap"].map(money)
    for column in ("Revenue growth", "EBITDA margin", "ROE"):
        result[column] = result[column].map(percent)
    for column in ("Forward P/E", "EV/EBITDA"):
        result[column] = result[column].map(multiple)
    return result


def page_header(kicker: str, title: str, description: str) -> None:
    st.markdown(
        f"""
        <div class="hero">
          <div class="eyebrow">{kicker}</div>
          <h1>{title}</h1>
          <p>{description}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def data_notice() -> None:
    st.markdown(
        """
        <div class="disclaimer">Market and fundamental data are sourced from Yahoo Finance and may be delayed,
        incomplete or restated. Expected returns, VaR and simulations are model estimates—not forecasts or investment advice.</div>
        """,
        unsafe_allow_html=True,
    )


def overview_page() -> None:
    page_header(
        "Personal investment intelligence",
        "Research with a portfolio view.",
        "A private workspace for company fundamentals, valuation, risk decomposition and evidence-led investment memos.",
    )
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("<div class='panel'><h3>01 · Research</h3><p class='muted'>Price, fundamentals, quality, peer valuation and a reusable investment-thesis workflow.</p><span class='pill'>Single name</span><span class='pill'>Peers</span></div>", unsafe_allow_html=True)
    with c2:
        st.markdown("<div class='panel'><h3>02 · Value</h3><p class='muted'>Enterprise-to-equity DCF with net debt, scenarios and discount-rate sensitivity.</p><span class='pill'>DCF</span><span class='pill'>Sensitivity</span></div>", unsafe_allow_html=True)
    with c3:
        st.markdown("<div class='panel'><h3>03 · Portfolio</h3><p class='muted'>Expected return, volatility, drawdown, VaR, beta, correlations and marginal risk.</p><span class='pill'>Risk</span><span class='pill'>Monte Carlo</span></div>", unsafe_allow_html=True)

    st.subheader("Research operating system")
    workflow = pd.DataFrame(
        {
            "Stage": ["Screen", "Underwrite", "Value", "Size", "Monitor"],
            "Decision question": [
                "Is the opportunity worth analyst time?",
                "What must be true for the thesis to work?",
                "What is priced in across scenarios?",
                "How much risk does the position add?",
                "Which evidence would invalidate the thesis?",
            ],
            "FinLens module": ["Company Research", "Company Research + AI Memo", "Valuation", "Portfolio Lab", "AI Memo"],
        }
    )
    st.dataframe(workflow, hide_index=True, width="stretch")
    data_notice()


def company_research_page(ticker: str) -> None:
    with st.spinner(f"Loading {ticker} research data…"):
        info, income, _, cashflow = company_bundle(ticker)
        prices = price_history((ticker,), "5y")
    if not info:
        st.error(f"No company data was returned for {ticker}.")
        return

    name = safe(info, "longName", ticker)
    page_header("Company research", f"{name} · {ticker}", f"{safe(info, 'sector', '—')} / {safe(info, 'industry', '—')}")
    price = safe(info, "currentPrice", safe(info, "regularMarketPrice"))
    metrics = st.columns(6)
    metric_values = [
        ("Price", money(price)),
        ("Market cap", money(safe(info, "marketCap"))),
        ("Forward P/E", multiple(safe(info, "forwardPE"))),
        ("EV / EBITDA", multiple(safe(info, "enterpriseToEbitda"))),
        ("Revenue growth", percent(safe(info, "revenueGrowth"))),
        ("ROIC proxy (ROA)", percent(safe(info, "returnOnAssets"))),
    ]
    for column, (label, value) in zip(metrics, metric_values):
        column.metric(label, value)

    left, right = st.columns([1.7, 1])
    with left:
        st.subheader("Price & regime")
        if prices.empty:
            st.warning("Price history is unavailable.")
        else:
            line = px.line(prices, x=prices.index, y=ticker, labels={ticker: "Adjusted price", "index": "Date"})
            line.update_layout(height=390, margin=dict(l=10, r=10, t=10, b=10), showlegend=False)
            st.plotly_chart(line, width="stretch")
    with right:
        st.subheader("Market profile")
        profile = pd.DataFrame(
            {
                "Metric": ["52-week low", "52-week high", "Beta", "Dividend yield", "Short % float", "Target mean"],
                "Value": [
                    money(safe(info, "fiftyTwoWeekLow")), money(safe(info, "fiftyTwoWeekHigh")),
                    f"{safe(info, 'beta', '—')}", percent(safe(info, "dividendYield")),
                    percent(safe(info, "shortPercentOfFloat")), money(safe(info, "targetMeanPrice")),
                ],
            }
        )
        st.dataframe(profile, hide_index=True, width="stretch")

    tabs = st.tabs(["Fundamentals", "Peer set", "Business profile"])
    with tabs[0]:
        rows = []
        if income is not None and not income.empty:
            for column in income.columns[:4]:
                revenue = float(income.loc["Total Revenue", column]) if "Total Revenue" in income.index else np.nan
                operating = float(income.loc["Operating Income", column]) if "Operating Income" in income.index else np.nan
                net_income = float(income.loc["Net Income", column]) if "Net Income" in income.index else np.nan
                fcf = float(cashflow.loc["Free Cash Flow", column]) if cashflow is not None and "Free Cash Flow" in cashflow.index and column in cashflow.columns else np.nan
                rows.append({"Period": pd.Timestamp(column).strftime("%Y"), "Revenue": money(revenue), "Operating income": money(operating), "Operating margin": percent(operating / revenue) if revenue else "—", "Net income": money(net_income), "Free cash flow": money(fcf)})
        if rows:
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        else:
            st.warning("Financial statements are unavailable.")
    with tabs[1]:
        peer_input = st.text_input("Peer tickers", "MSFT,AMZN,GOOGL,META", help="Use comparable companies from the same economic value chain.")
        peers = tuple(dict.fromkeys([ticker] + [item.strip().upper() for item in peer_input.split(",") if item.strip()]))
        st.dataframe(display_peer_table(peer_snapshot(peers)), hide_index=True, width="stretch")
    with tabs[2]:
        st.write(safe(info, "longBusinessSummary", "Business description is unavailable."))
        st.caption(f"Website: {safe(info, 'website', '—')} · Employees: {safe(info, 'fullTimeEmployees', '—')}")
    data_notice()


def valuation_page(ticker: str) -> None:
    info, _, balance, cashflow = company_bundle(ticker)
    name = safe(info, "longName", ticker)
    page_header("Intrinsic value", f"DCF laboratory · {ticker}", f"Enterprise value bridge for {name} with explicit scenario assumptions.")
    latest_fcf = statement_value(cashflow, ["Free Cash Flow"])
    if valid_number(latest_fcf) and latest_fcf <= 0 and cashflow is not None and "Free Cash Flow" in cashflow.index:
        positive_history = pd.to_numeric(cashflow.loc["Free Cash Flow"], errors="coerce")
        positive_history = positive_history[positive_history > 0]
        latest_fcf = float(positive_history.iloc[0]) if not positive_history.empty else None
    if not valid_number(latest_fcf):
        operating_cf = statement_value(cashflow, ["Operating Cash Flow", "Total Cash From Operating Activities"])
        capex = statement_value(cashflow, ["Capital Expenditure", "Capital Expenditures"])
        latest_fcf = operating_cf + capex if valid_number(operating_cf) and valid_number(capex) else None
    debt = statement_value(balance, ["Total Debt"]) or 0.0
    cash = statement_value(balance, ["Cash Cash Equivalents And Short Term Investments", "Cash And Cash Equivalents"]) or 0.0
    shares = safe(info, "sharesOutstanding")

    assumptions, output = st.columns([1, 1.25])
    with assumptions:
        st.subheader("Assumptions")
        fcf_input = st.number_input("Base free cash flow", min_value=0.0, value=float(latest_fcf or 1_000_000_000.0), step=10_000_000.0, format="%.0f", help="When the latest reported FCF is negative, FinLens uses the most recent positive historical value as a starting point. Normalize it before relying on the output.")
        shares_input = st.number_input("Diluted shares", min_value=1.0, value=float(shares or 1_000_000_000.0), step=1_000_000.0, format="%.0f")
        net_debt_input = st.number_input("Net debt (debt − cash)", value=float(debt - cash), step=10_000_000.0, format="%.0f")
        growth = st.slider("FCF growth", -5.0, 30.0, 8.0, 0.5, format="%.1f%%") / 100
        discount = st.slider("WACC / required return", 5.0, 18.0, 9.0, 0.25, format="%.2f%%") / 100
        terminal = st.slider("Terminal growth", 0.0, 5.0, 2.5, 0.25, format="%.2f%%") / 100
    try:
        result = dcf_valuation(fcf_input, shares_input, net_debt_input, growth, discount, terminal)
    except ValueError as error:
        st.error(str(error))
        return

    current = safe(info, "currentPrice", safe(info, "regularMarketPrice"))
    with output:
        st.subheader("Value bridge")
        a, b, c = st.columns(3)
        a.metric("Fair value / share", money(result["fair_value_per_share"]))
        b.metric("Current price", money(current))
        upside = result["fair_value_per_share"] / current - 1 if valid_number(current) else None
        c.metric("Implied upside", percent(upside))
        bridge = pd.DataFrame({"Component": ["Enterprise value", "Less: net debt", "Equity value", "PV from terminal value"], "Value": [money(result["enterprise_value"]), money(net_debt_input), money(result["equity_value"]), percent(result["terminal_value_share"])]})
        st.dataframe(bridge, hide_index=True, width="stretch")
        projected = pd.DataFrame({"Year": range(1, 6), "Free cash flow": result["projected_fcfs"]})
        chart = px.bar(projected, x="Year", y="Free cash flow")
        chart.update_layout(height=280, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(chart, width="stretch")

    st.subheader("WACC × terminal-growth sensitivity")
    wacc_values = np.linspace(max(terminal + 0.015, discount - 0.02), discount + 0.02, 5)
    terminal_values = np.linspace(max(0.0, terminal - 0.01), min(0.05, terminal + 0.01), 5)
    sensitivity = pd.DataFrame(index=[f"{value:.1%}" for value in wacc_values], columns=[f"{value:.1%}" for value in terminal_values])
    for wacc in wacc_values:
        for growth_terminal in terminal_values:
            try:
                value = dcf_valuation(fcf_input, shares_input, net_debt_input, growth, float(wacc), float(growth_terminal))["fair_value_per_share"]
                sensitivity.loc[f"{wacc:.1%}", f"{growth_terminal:.1%}"] = round(value, 2)
            except ValueError:
                sensitivity.loc[f"{wacc:.1%}", f"{growth_terminal:.1%}"] = np.nan
    st.caption("Rows: WACC · Columns: terminal growth · Values: fair value per share")
    st.dataframe(sensitivity, width="stretch")
    data_notice()


def default_holdings() -> pd.DataFrame:
    return pd.DataFrame({"Ticker": ["MSFT", "NVDA", "GOOGL", "AMZN", "BRK-B"], "Shares": [10.0, 15.0, 20.0, 12.0, 5.0], "Cost basis": [350.0, 120.0, 170.0, 180.0, 450.0]})


def portfolio_page() -> None:
    page_header("Portfolio intelligence", "Risk & return laboratory", "Translate positions into portfolio-level exposures, loss ranges and forward-looking scenarios.")
    st.markdown("<div class='status'><span class='status-dot'></span>Session-only holdings · nothing is saved to the server</div>", unsafe_allow_html=True)
    uploaded = st.file_uploader("Import holdings CSV", type=["csv"], help="Columns: Ticker, Shares, Cost basis (optional)")
    if uploaded is not None:
        try:
            imported = pd.read_csv(uploaded)
            if not {"Ticker", "Shares"}.issubset(imported.columns):
                st.error("CSV must contain Ticker and Shares columns.")
            else:
                if "Cost basis" not in imported.columns:
                    imported["Cost basis"] = np.nan
                st.session_state["holdings"] = imported[["Ticker", "Shares", "Cost basis"]]
        except Exception:
            st.error("The CSV could not be read.")

    if "holdings" not in st.session_state:
        st.session_state["holdings"] = default_holdings()
    editor = st.data_editor(
        st.session_state["holdings"], num_rows="dynamic", hide_index=True, width="stretch",
        column_config={"Ticker": st.column_config.TextColumn(required=True), "Shares": st.column_config.NumberColumn(min_value=0.0, required=True, format="%.4f"), "Cost basis": st.column_config.NumberColumn(min_value=0.0, format="$%.2f")},
        key="portfolio_editor",
    )
    st.session_state["holdings"] = editor
    holdings = editor.copy()
    holdings["Ticker"] = holdings["Ticker"].astype(str).str.strip().str.upper()
    holdings["Shares"] = pd.to_numeric(holdings["Shares"], errors="coerce")
    holdings["Cost basis"] = pd.to_numeric(holdings["Cost basis"], errors="coerce")
    holdings = holdings[(holdings["Ticker"] != "") & (holdings["Shares"] > 0)]
    if holdings.empty:
        st.info("Add at least one position to run the analysis.")
        return

    settings = st.columns(4)
    period = settings[0].selectbox("History", ["1y", "3y", "5y", "10y"], index=2)
    benchmark = settings[1].text_input("Benchmark", "SPY").strip().upper()
    risk_free = settings[2].number_input("Risk-free rate", 0.0, 0.15, 0.04, 0.005, format="%.3f")
    return_method = settings[3].selectbox("Expected return method", ["Historical mean", "Historical CAGR", "Exponentially weighted"])

    tickers = tuple(dict.fromkeys(holdings["Ticker"].tolist()))
    with st.spinner("Building the portfolio risk model…"):
        all_prices = price_history(tuple(dict.fromkeys(tickers + (benchmark,))), period)
    available = [ticker for ticker in tickers if ticker in all_prices.columns]
    missing = sorted(set(tickers) - set(available))
    if missing:
        st.warning(f"No price history for: {', '.join(missing)}")
    if not available:
        st.error("No usable price history was returned.")
        return

    grouped = holdings.groupby("Ticker", as_index=False).agg({"Shares": "sum", "Cost basis": "mean"})
    grouped = grouped[grouped["Ticker"].isin(available)].copy()
    latest = all_prices[available].ffill().iloc[-1]
    grouped["Price"] = grouped["Ticker"].map(latest)
    grouped["Market value"] = grouped["Shares"] * grouped["Price"]
    grouped["Weight"] = grouped["Market value"] / grouped["Market value"].sum()
    grouped["Unrealized P&L"] = (grouped["Price"] - grouped["Cost basis"]) * grouped["Shares"]
    grouped = grouped.sort_values("Market value", ascending=False)
    analysis_tickers = grouped["Ticker"].tolist()
    weights = normalize_weights(grouped["Weight"])
    returns = daily_returns(all_prices[analysis_tickers])
    benchmark_returns = all_prices[benchmark].pct_change(fill_method=None).dropna() if benchmark in all_prices.columns else None
    if returns.empty or len(returns) < 20:
        st.error("At least 20 common trading days are required for portfolio analysis.")
        return

    metrics = portfolio_metrics(returns, weights, benchmark_returns, risk_free, return_method)
    portfolio_value = float(grouped["Market value"].sum())
    metric_columns = st.columns(6)
    headline = [("Portfolio value", money(portfolio_value)), ("Expected return", percent(metrics.expected_return)), ("Annual volatility", percent(metrics.volatility)), ("Sharpe ratio", f"{metrics.sharpe_ratio:.2f}"), ("Max drawdown", percent(metrics.max_drawdown)), ("1-day VaR 95%", money(metrics.var_95_daily * portfolio_value))]
    for column, (label, value) in zip(metric_columns, headline):
        column.metric(label, value)

    tabs = st.tabs(["Positions", "Performance", "Risk decomposition", "Stress & simulation"])
    with tabs[0]:
        display = grouped.copy()
        for column in ("Price", "Market value", "Unrealized P&L"):
            display[column] = display[column].map(money)
        display["Weight"] = display["Weight"].map(percent)
        display["Cost basis"] = display["Cost basis"].map(money)
        st.dataframe(display, hide_index=True, width="stretch")
        allocation = px.bar(grouped, x="Ticker", y="Weight", color="Weight", color_continuous_scale="Teal")
        allocation.update_layout(height=320, coloraxis_showscale=False, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(allocation, width="stretch")
        st.download_button("Download current holdings", grouped[["Ticker", "Shares", "Cost basis"]].to_csv(index=False), "finlens-holdings.csv", "text/csv")
    with tabs[1]:
        portfolio_returns = returns.mul(weights, axis=1).sum(axis=1)
        comparison = pd.DataFrame({"Portfolio": (1.0 + portfolio_returns).cumprod() * 100})
        if benchmark_returns is not None:
            common = comparison.index.intersection(benchmark_returns.index)
            comparison = comparison.loc[common]
            comparison[benchmark] = (1.0 + benchmark_returns.loc[common]).cumprod() * 100
        chart = px.line(comparison, x=comparison.index, y=comparison.columns, labels={"value": "Growth of 100", "index": "Date", "variable": "Series"})
        chart.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(chart, width="stretch")
        ratio_columns = st.columns(5)
        ratios = [("Sortino", metrics.sortino_ratio), ("Beta", metrics.beta), ("Tracking error", percent(metrics.tracking_error)), ("Information ratio", metrics.information_ratio), ("1-day CVaR 95%", money(metrics.cvar_95_daily * portfolio_value))]
        for column, (label, value) in zip(ratio_columns, ratios):
            formatted = f"{value:.2f}" if valid_number(value) and label not in ("Tracking error", "1-day CVaR 95%") else value
            column.metric(label, formatted if formatted is not None else "—")
    with tabs[2]:
        left, right = st.columns(2)
        heatmap = px.imshow(returns.corr(), text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto")
        heatmap.update_layout(height=430, margin=dict(l=10, r=10, t=30, b=10), title="Return correlation")
        left.plotly_chart(heatmap, width="stretch")
        contributions = risk_contributions(returns, weights).sort_values(ascending=False).reset_index()
        contributions.columns = ["Ticker", "Risk contribution"]
        risk_chart = px.bar(contributions, x="Ticker", y="Risk contribution", color="Risk contribution", color_continuous_scale="Oranges")
        risk_chart.update_layout(height=430, coloraxis_showscale=False, margin=dict(l=10, r=10, t=30, b=10), title="Contribution to portfolio variance")
        right.plotly_chart(risk_chart, width="stretch")
        asset_expected = expected_asset_returns(returns, return_method)
        scatter_data = pd.DataFrame({"Ticker": returns.columns, "Expected return": asset_expected, "Volatility": returns.std() * np.sqrt(252)})
        scatter = px.scatter(scatter_data, x="Volatility", y="Expected return", text="Ticker", size=[16] * len(scatter_data))
        scatter.update_traces(textposition="top center")
        scatter.update_layout(height=380, margin=dict(l=10, r=10, t=30, b=10), title="Asset risk / return map")
        st.plotly_chart(scatter, width="stretch")
    with tabs[3]:
        left, right = st.columns([0.9, 1.1])
        with left:
            st.subheader("Historical stress")
            stress = historical_stress(returns, weights)
            stress["Portfolio return"] = stress["Portfolio return"].map(percent)
            st.dataframe(stress, hide_index=True, width="stretch")
            years = st.slider("Simulation horizon (years)", 1, 15, 5)
            simulations = st.select_slider("Simulation paths", options=[1_000, 2_500, 5_000, 10_000], value=5_000)
        terminal_values = monte_carlo_terminal_values(returns, weights, portfolio_value, years, simulations)
        summary = terminal_summary(terminal_values, portfolio_value)
        with right:
            st.subheader("Monte Carlo terminal value")
            histogram = px.histogram(x=terminal_values, nbins=55, labels={"x": "Terminal portfolio value", "count": "Paths"})
            histogram.add_vline(x=portfolio_value, line_dash="dash", line_color="#46d6a0")
            histogram.update_layout(height=360, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(histogram, width="stretch")
        summary_columns = st.columns(4)
        for column, (label, value) in zip(summary_columns, summary.items()):
            column.metric(label, percent(value) if label == "Probability of loss" else money(value))
    data_notice()


def ai_memo_page(ticker: str) -> None:
    page_header("Research copilot", "AI investment memo", "Use the model to challenge a thesis and structure evidence—not to replace primary research.")
    api_key = secret_value("OPENAI_API_KEY")
    if not api_key or OpenAI is None:
        st.error("OpenAI is not configured for this environment.")
        return
    info, _, balance, cashflow = company_bundle(ticker)
    memo_type = st.selectbox("Memo type", ["Initiation of coverage", "Thesis stress test", "Earnings preparation", "Risk register"])
    notes = st.text_area("Your thesis, questions and known evidence", height=220, placeholder="Example: Revenue growth is accelerating, but capex intensity and customer concentration may be underappreciated…")
    include_financials = st.checkbox("Include summarized company financial data", value=True)
    st.caption("Only the ticker, selected public company data and the notes above are sent when you click Generate.")
    if st.button("Generate analyst memo", type="primary"):
        financial_context = ""
        if include_financials:
            financial_context = f"""
Public company context:
- Company: {safe(info, 'longName', ticker)} ({ticker})
- Sector / industry: {safe(info, 'sector', 'N/A')} / {safe(info, 'industry', 'N/A')}
- Market cap: {safe(info, 'marketCap')}
- Forward P/E: {safe(info, 'forwardPE')}
- Revenue growth: {safe(info, 'revenueGrowth')}
- EBITDA margin: {safe(info, 'ebitdaMargins')}
- Latest FCF: {statement_value(cashflow, ['Free Cash Flow'])}
- Total debt: {statement_value(balance, ['Total Debt'])}
"""
        prompt = f"""
Memo type: {memo_type}
Ticker: {ticker}
Analyst notes:
{notes or 'No analyst notes supplied; identify the minimum evidence needed before forming a view.'}
{financial_context}

Produce a concise professional equity-research memo with:
1. Executive view and confidence level
2. What the market may be pricing in
3. Three thesis pillars with evidence required
4. Bull, base and bear cases (clearly labeled as scenarios, not forecasts)
5. Variant perception
6. Catalysts and timeline
7. Risk register with leading indicators
8. Disconfirming evidence and next research actions
Do not provide personalized investment advice. Flag missing or stale data explicitly.
"""
        try:
            with st.spinner("Drafting the memo…"):
                client = OpenAI(api_key=api_key, timeout=30.0, max_retries=1)
                response = client.responses.create(
                    model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
                    instructions="You are a skeptical institutional equity research analyst. Separate facts, assumptions and inferences. Never invent financial data or citations.",
                    input=prompt,
                    max_output_tokens=2200,
                    store=False,
                )
            st.markdown(response.output_text)
            st.download_button("Download memo", response.output_text, f"{ticker.lower()}-{memo_type.lower().replace(' ', '-')}.md", "text/markdown")
        except Exception as error:
            st.error("The memo could not be generated. Check API access, model availability and project billing.")
            st.caption(type(error).__name__)
    data_notice()


def methodology_page() -> None:
    page_header("Model governance", "Methodology & limitations", "A professional tool is explicit about assumptions, data lineage and where its outputs can fail.")
    sections = {
        "Expected return": "Historical mean annualizes average daily returns; CAGR compounds the observed sample; exponentially weighted estimates emphasize recent returns. None is a reliable standalone forecast.",
        "Risk": "Volatility and covariance use daily adjusted-price returns with 252 trading days. Risk contribution allocates total portfolio variance using marginal covariance.",
        "VaR / CVaR": "Historical 95% VaR is the fifth-percentile one-day loss. CVaR is the average loss in that tail. Both can understate regime shifts and illiquidity.",
        "Monte Carlo": "Terminal values use a lognormal process calibrated to historical portfolio mean and volatility. Paths are scenarios, not price targets.",
        "DCF": "The model discounts explicit FCF, adds a Gordon-growth terminal value, then subtracts net debt to bridge enterprise to equity value.",
        "Privacy": "Holdings remain in the Streamlit session and are not stored by FinLens. AI requests send only the information shown on the AI Memo page after an explicit click.",
    }
    for title, body in sections.items():
        st.markdown(f"### {title}\n{body}")
    data_notice()


require_authentication()
st.sidebar.markdown("## ◈ FINLENS")
st.sidebar.caption("PRIVATE RESEARCH TERMINAL")
page = st.sidebar.radio("Workspace", ["Overview", "Company Research", "Valuation", "Portfolio Lab", "AI Memo", "Methodology"], label_visibility="collapsed")
st.sidebar.divider()
ticker = st.sidebar.text_input("Active ticker", st.session_state.get("ticker", "ORCL")).strip().upper()
st.session_state["ticker"] = ticker or "ORCL"
st.sidebar.markdown("<div class='status'><span class='status-dot'></span>Private session active</div>", unsafe_allow_html=True)
if st.sidebar.button("Lock workspace", width="stretch"):
    st.session_state["authenticated"] = False
    st.rerun()

if page == "Overview":
    overview_page()
elif page == "Company Research":
    company_research_page(st.session_state["ticker"])
elif page == "Valuation":
    valuation_page(st.session_state["ticker"])
elif page == "Portfolio Lab":
    portfolio_page()
elif page == "AI Memo":
    ai_memo_page(st.session_state["ticker"])
else:
    methodology_page()
