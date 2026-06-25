import os
import requests
import pandas as pd
import yfinance as yf
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from dotenv import load_dotenv
from openai import OpenAI
from streamlit_autorefresh import st_autorefresh

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

st.set_page_config(page_title="AI Equity Research Terminal", layout="wide")


# -----------------------
# Helper Functions
# -----------------------

def fmt_money(x):
    try:
        if x is None or pd.isna(x):
            return "N/A"
        x = float(x)
        if abs(x) >= 1_000_000_000:
            return f"${x / 1_000_000_000:.2f}B"
        if abs(x) >= 1_000_000:
            return f"${x / 1_000_000:.2f}M"
        return f"${x:,.2f}"
    except:
        return "N/A"


def fmt_pct(x):
    try:
        if x is None or pd.isna(x):
            return "N/A"
        return f"{float(x) * 100:.1f}%"
    except:
        return "N/A"


def safe_get(info, key, default=None):
    value = info.get(key, default)
    return value if value not in ["", None] else default


def search_ticker(query):
    query = query.strip()

    known_map = {
        "oracle": "ORCL",
        "microsoft": "MSFT",
        "nvidia": "NVDA",
        "apple": "AAPL",
        "amazon": "AMZN",
        "tesla": "TSLA",
        "google": "GOOGL",
        "alphabet": "GOOGL",
        "meta": "META",
        "facebook": "META",
        "tsmc": "TSM",
        "sap": "SAP",
        "salesforce": "CRM"
    }

    if query.lower() in known_map:
        return known_map[query.lower()]

    if query.isupper() and len(query) <= 6:
        return query

    try:
        url = "https://query2.finance.yahoo.com/v1/finance/search"
        params = {"q": query, "quotes_count": 5, "news_count": 0}
        headers = {"User-Agent": "Mozilla/5.0"}

        r = requests.get(url, params=params, headers=headers, timeout=8)
        data = r.json()

        quotes = data.get("quotes", [])
        for q in quotes:
            if q.get("quoteType") == "EQUITY" and q.get("symbol"):
                return q.get("symbol")

    except:
        pass

    return query.upper()


def get_logo_url(info):
    website = info.get("website")
    if not website:
        return None
    try:
        domain = website.replace("https://", "").replace("http://", "").split("/")[0]
        return f"https://logo.clearbit.com/{domain}"
    except:
        return None


def get_company(ticker):
    stock = yf.Ticker(ticker)
    info = stock.info
    return stock, info


def get_financial_summary(stock):
    income = stock.financials
    bs = stock.balance_sheet
    cf = stock.cashflow

    rows = []

    if income.empty:
        return pd.DataFrame(), income, bs, cf

    for year in income.columns:
        revenue = income.loc["Total Revenue", year] if "Total Revenue" in income.index else None
        gross_profit = income.loc["Gross Profit", year] if "Gross Profit" in income.index else None
        ebitda = income.loc["EBITDA", year] if "EBITDA" in income.index else None
        operating_income = income.loc["Operating Income", year] if "Operating Income" in income.index else None
        net_income = income.loc["Net Income", year] if "Net Income" in income.index else None

        operating_cf = cf.loc["Operating Cash Flow", year] if "Operating Cash Flow" in cf.index and year in cf.columns else None
        capex = cf.loc["Capital Expenditure", year] if "Capital Expenditure" in cf.index and year in cf.columns else None
        fcf = cf.loc["Free Cash Flow", year] if "Free Cash Flow" in cf.index and year in cf.columns else None

        total_debt = bs.loc["Total Debt", year] if "Total Debt" in bs.index and year in bs.columns else None
        cash = bs.loc["Cash And Cash Equivalents", year] if "Cash And Cash Equivalents" in bs.index and year in bs.columns else None

        rows.append({
            "Year": year.strftime("%Y-%m-%d"),
            "Revenue": revenue,
            "Gross Profit": gross_profit,
            "EBITDA": ebitda,
            "Operating Income": operating_income,
            "Net Income": net_income,
            "Operating Cash Flow": operating_cf,
            "CapEx": capex,
            "Free Cash Flow": fcf,
            "Total Debt": total_debt,
            "Cash": cash,
            "Gross Margin": gross_profit / revenue if revenue and gross_profit else None,
            "EBITDA Margin": ebitda / revenue if revenue and ebitda else None,
            "Operating Margin": operating_income / revenue if revenue and operating_income else None,
            "Net Margin": net_income / revenue if revenue and net_income else None,
            "FCF Margin": fcf / revenue if revenue and fcf else None,
        })

    return pd.DataFrame(rows), income, bs, cf


def simple_dcf(latest_fcf, growth, discount_rate, terminal_growth, shares):
    try:
        if not latest_fcf or not shares or discount_rate <= terminal_growth:
            return None

        fcfs = [latest_fcf * ((1 + growth) ** i) for i in range(1, 6)]
        pv_fcfs = [fcf / ((1 + discount_rate) ** (i + 1)) for i, fcf in enumerate(fcfs)]

        terminal_value = fcfs[-1] * (1 + terminal_growth) / (discount_rate - terminal_growth)
        pv_terminal = terminal_value / ((1 + discount_rate) ** 5)

        equity_value = sum(pv_fcfs) + pv_terminal
        return equity_value / shares
    except:
        return None


def ai_report(prompt):
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content


def get_competitor_table(tickers):
    rows = []

    for t in tickers:
        try:
            s = yf.Ticker(t)
            i = s.info

            rows.append({
                "Ticker": t,
                "Name": i.get("shortName", "N/A"),
                "Price": i.get("currentPrice", None),
                "Market Cap": i.get("marketCap", None),
                "P/E": i.get("trailingPE", None),
                "Forward P/E": i.get("forwardPE", None),
                "Revenue Growth": i.get("revenueGrowth", None),
                "Profit Margin": i.get("profitMargins", None),
                "ROE": i.get("returnOnEquity", None),
            })
        except:
            pass

    return pd.DataFrame(rows)


def get_news(stock):
    try:
        return stock.news[:10]
    except:
        return []


# -----------------------
# Sidebar
# -----------------------

st.sidebar.title("AI Equity Research Terminal")

query = st.sidebar.text_input("Search company or ticker", "Oracle")
ticker = search_ticker(query)

page = st.sidebar.radio(
    "Pages",
    [
        "Dashboard",
        "Financial Statements",
        "Market Data",
        "Charts & K-Line",
        "DCF Valuation",
        "P/E EPS Valuation",
        "Analyst Price Targets",
        "Business & Moat",
        "Competitors & Risk",
        "News & AI Analysis",
        "AI Report",
        "Watchlist"
    ]
)


# -----------------------
# Load Data
# -----------------------

stock, info = get_company(ticker)
summary_df, income, bs, cf = get_financial_summary(stock)

company_name = safe_get(info, "longName", ticker)
price = safe_get(info, "currentPrice", safe_get(info, "regularMarketPrice", None))
regular_price = safe_get(info, "regularMarketPrice", price)
pre_price = safe_get(info, "preMarketPrice", None)
post_price = safe_get(info, "postMarketPrice", None)

market_cap = safe_get(info, "marketCap", None)
pe = safe_get(info, "trailingPE", None)
forward_pe = safe_get(info, "forwardPE", None)
eps = safe_get(info, "trailingEps", None)
forward_eps = safe_get(info, "forwardEps", None)
shares = safe_get(info, "sharesOutstanding", None)
sector = safe_get(info, "sector", "N/A")
industry = safe_get(info, "industry", "N/A")

target_mean = safe_get(info, "targetMeanPrice", None)
target_high = safe_get(info, "targetHighPrice", None)
target_low = safe_get(info, "targetLowPrice", None)


# -----------------------
# Header
# -----------------------

logo_url = get_logo_url(info)

col_title, col_logo = st.columns([8, 1])

with col_title:
    st.title(f"{company_name} ({ticker})")
    st.caption(f"{sector} | {industry}")

with col_logo:
    if logo_url:
        st.image(logo_url, width=70)


# -----------------------
# Pages
# -----------------------

if page == "Dashboard":
    st_autorefresh(interval=60_000, key="dashboard_refresh")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Current Price", fmt_money(price))
    col2.metric("Market Cap", fmt_money(market_cap))
    col3.metric("P/E", pe if pe else "N/A")
    col4.metric("Forward P/E", forward_pe if forward_pe else "N/A")

    st.subheader("Financial Snapshot")

    display_df = summary_df.copy()

    money_cols = [
        "Revenue", "Gross Profit", "EBITDA", "Operating Income", "Net Income",
        "Operating Cash Flow", "CapEx", "Free Cash Flow", "Total Debt", "Cash"
    ]

    pct_cols = [
        "Gross Margin", "EBITDA Margin", "Operating Margin", "Net Margin", "FCF Margin"
    ]

    for col in money_cols:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(fmt_money)

    for col in pct_cols:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(fmt_pct)

    st.dataframe(display_df, use_container_width=True)


elif page == "Financial Statements":

    def format_statement(df):
        formatted = df.T.copy()

        for col in formatted.columns:
            formatted[col] = formatted[col].apply(
                lambda x: fmt_money(x) if isinstance(x, (int, float)) else x
            )

        return formatted

    st.subheader("Income Statement")
    st.dataframe(format_statement(income), use_container_width=True)

    st.subheader("Balance Sheet")
    st.dataframe(format_statement(bs), use_container_width=True)

    st.subheader("Cash Flow Statement")
    st.dataframe(format_statement(cf), use_container_width=True)


elif page == "Market Data":
    st_autorefresh(interval=60_000, key="market_refresh")

    pre_change = (pre_price - regular_price) / regular_price if pre_price and regular_price else None
    post_change = (post_price - regular_price) / regular_price if post_price and regular_price else None

    col1, col2, col3 = st.columns(3)
    col1.metric("Regular Market Price", fmt_money(regular_price))
    col2.metric("Pre-Market Price", fmt_money(pre_price), fmt_pct(pre_change) if pre_change else None)
    col3.metric("After-Hours Price", fmt_money(post_price), fmt_pct(post_change) if post_change else None)

    col4, col5, col6 = st.columns(3)
    col4.metric("Market Cap", fmt_money(market_cap))
    col5.metric("Beta", info.get("beta", "N/A"))
    col6.metric("Volume", info.get("volume", "N/A"))

    col7, col8, col9 = st.columns(3)
    col7.metric("52W High", fmt_money(info.get("fiftyTwoWeekHigh")))
    col8.metric("52W Low", fmt_money(info.get("fiftyTwoWeekLow")))
    col9.metric("Dividend Yield", fmt_pct(info.get("dividendYield")))

    st.caption("Pre-market and after-hours data may be unavailable depending on market session and Yahoo Finance coverage.")


elif page == "Charts & K-Line":
    period = st.selectbox("Period", ["1mo", "3mo", "6mo", "1y", "5y"], index=3)
    hist = stock.history(period=period)

    st.subheader("Line Chart")
    fig = px.line(hist, x=hist.index, y="Close", title=f"{ticker} Closing Price")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("K-Line / Candlestick Chart")
    fig = go.Figure(data=[
        go.Candlestick(
            x=hist.index,
            open=hist["Open"],
            high=hist["High"],
            low=hist["Low"],
            close=hist["Close"]
        )
    ])
    fig.update_layout(title=f"{ticker} Candlestick Chart", xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)


elif page == "DCF Valuation":
    st.subheader("DCF Valuation")

    latest_fcf = summary_df["Free Cash Flow"].dropna().iloc[0] if not summary_df.empty and "Free Cash Flow" in summary_df.columns else None

    st.write("Latest FCF:", fmt_money(latest_fcf))

    growth = st.slider("FCF Growth Rate", 0.00, 0.20, 0.06)
    discount_rate = st.slider("Discount Rate / WACC", 0.05, 0.15, 0.09)
    terminal_growth = st.slider("Terminal Growth Rate", 0.00, 0.05, 0.025)

    fair_value = simple_dcf(latest_fcf, growth, discount_rate, terminal_growth, shares)

    col1, col2 = st.columns(2)
    col1.metric("DCF Fair Value / Share", fmt_money(fair_value))
    col2.metric("Current Price", fmt_money(price))


elif page == "P/E EPS Valuation":
    st.subheader("P/E × EPS Valuation")

    base_eps = float(forward_eps or eps or 5.0)

    st.write("Forward EPS:", forward_eps if forward_eps else "N/A")
    st.write("Trailing EPS:", eps if eps else "N/A")

    base_pe = st.slider("Base Case P/E", 5, 60, 20)
    bear_pe = st.slider("Bear Case P/E", 5, 60, int(forward_pe) if forward_pe else 25)
    bull_pe = st.slider("Bull Case P/E", 5, 80, 35)

    custom_eps = st.number_input("Adjust EPS", value=base_eps)

    bear_price = bear_pe * custom_eps
    base_price = base_pe * custom_eps
    bull_price = bull_pe * custom_eps

    col1, col2, col3 = st.columns(3)
    col1.metric("Base Case Price", fmt_money(base_price))
    col2.metric("Bear Case Price", fmt_money(bear_price))
    col3.metric("Bull Case Price", fmt_money(bull_price))

    valuation_df = pd.DataFrame({
        "Case": ["Bear", "Base", "Bull"],
        "EPS": [custom_eps, custom_eps, custom_eps],
        "P/E": [bear_pe, base_pe, bull_pe],
        "Implied Price": [bear_price, base_price, bull_price]
    })

    st.dataframe(valuation_df, use_container_width=True)


elif page == "Analyst Price Targets":
    st.subheader("Analyst Price Targets")

    target_df = pd.DataFrame({
        "Estimate": ["Low Target", "Mean Target", "High Target"],
        "Price": [target_low, target_mean, target_high]
    })

    st.dataframe(target_df, use_container_width=True)

    fig = px.bar(target_df, x="Estimate", y="Price", title="Analyst Target Price Range")
    st.plotly_chart(fig, use_container_width=True)

    st.caption("Free Yahoo Finance data does not disclose which investment bank issued each target. For GS/MS/JPM-specific estimates, paid APIs such as Bloomberg, FactSet, Refinitiv, Benzinga, or Financial Modeling Prep paid plans are usually required.")


elif page == "Business & Moat":
    st.subheader("Company Introduction, Moat & Fundamentals")

    prompt = f"""
You are an equity research analyst.

Company: {company_name}
Ticker: {ticker}
Sector: {sector}
Industry: {industry}
Market Cap: {market_cap}
P/E: {pe}

Write a professional company analysis including:
1. Company introduction
2. Main business segments
3. Revenue sources
4. Business model
5. Business fundamentals
6. Economic moat
7. Switching costs
8. Scale advantages
9. Competitive risks
10. Long-term outlook
"""

    if st.button("Generate Business & Moat Analysis"):
        with st.spinner("Generating..."):
            st.markdown(ai_report(prompt))


elif page == "Competitors & Risk":
    st.subheader("Competitor Comparison")

    default_peers = "MSFT,AMZN,SAP,CRM"
    peer_input = st.text_input("Peer tickers", default_peers)
    peer_tickers = [x.strip().upper() for x in peer_input.split(",") if x.strip()]

    comp_df = get_competitor_table([ticker] + peer_tickers)

    display_comp = comp_df.copy()

    if not display_comp.empty:
        display_comp["Market Cap"] = display_comp["Market Cap"].apply(fmt_money)
        display_comp["Revenue Growth"] = display_comp["Revenue Growth"].apply(fmt_pct)
        display_comp["Profit Margin"] = display_comp["Profit Margin"].apply(fmt_pct)
        display_comp["ROE"] = display_comp["ROE"].apply(fmt_pct)

    st.dataframe(display_comp, use_container_width=True)

    st.subheader("Risk Analysis")

    prompt = f"""
You are an equity research analyst.

Company: {company_name}
Ticker: {ticker}
Sector: {sector}
Industry: {industry}

Competitors:
{comp_df.to_string()}

Write a company-specific risk analysis including:
1. Competitive risk
2. Margin risk
3. Valuation risk
4. Business model risk
5. Macroeconomic risk
6. Execution risk

Also compare the company against its competitors.
"""

    if st.button("Generate Risk Analysis"):
        with st.spinner("Generating risk analysis..."):
            st.markdown(ai_report(prompt))


elif page == "News & AI Analysis":
    st.subheader("Latest Company News")

    news = get_news(stock)

    news_rows = []

    for n in news:
        try:
            content = n.get("content", n)
            title = content.get("title", n.get("title", "N/A"))
            publisher = content.get("provider", {}).get("displayName", n.get("publisher", "N/A"))
            link = content.get("canonicalUrl", {}).get("url", n.get("link", None))

            news_rows.append({
                "Title": title,
                "Publisher": publisher,
                "Link": link
            })
        except:
            pass

    news_df = pd.DataFrame(news_rows)
    st.dataframe(news_df, use_container_width=True)

    st.caption("Free Yahoo Finance news may not always include Bloomberg or Reuters. Bloomberg/Reuters direct feeds usually require paid access.")

    prompt = f"""
You are an equity research analyst.

Company: {company_name}
Ticker: {ticker}

Latest news:
{news_df.to_string()}

Summarize the news and explain:
1. What happened
2. Whether it is positive, negative, or neutral
3. Possible impact on revenue, margins, valuation, or sentiment
4. Key risks to monitor
"""

    if st.button("Generate News AI Analysis"):
        with st.spinner("Analyzing news..."):
            st.markdown(ai_report(prompt))


elif page == "AI Report":
    st.subheader("AI Equity Research Report")

    financial_text = summary_df.to_string()

    prompt = f"""
You are a professional equity research analyst.

Analyze this company using the financial data below.

Company: {company_name}
Ticker: {ticker}
Sector: {sector}
Industry: {industry}
Current Price: {price}
Market Cap: {market_cap}
P/E: {pe}
Forward P/E: {forward_pe}

Financial Data:
{financial_text}

Write a structured equity research report:
1. Business Overview
2. Business Fundamentals
3. Revenue and Profitability Analysis
4. Cash Flow Analysis
5. Moat Analysis
6. Growth Drivers
7. Bull Case
8. Bear Case
9. Risks
10. Investment Conclusion

Do not give personalized financial advice.
"""

    if st.button("Generate AI Report"):
        with st.spinner("Generating AI Report..."):
            st.markdown(ai_report(prompt))


elif page == "Watchlist":
    st_autorefresh(interval=60_000, key="watchlist_refresh")

    st.subheader("Real-Time Watchlist")

    tickers = st.text_area("Enter tickers separated by commas", "ORCL,MSFT,NVDA,AAPL,TSM")
    tickers = [t.strip().upper() for t in tickers.split(",") if t.strip()]

    data = []

    for t in tickers:
        try:
            s = yf.Ticker(t)
            i = s.info

            data.append({
                "Ticker": t,
                "Name": i.get("shortName", "N/A"),
                "Regular Price": i.get("regularMarketPrice", i.get("currentPrice", None)),
                "Pre-Market": i.get("preMarketPrice", None),
                "After-Hours": i.get("postMarketPrice", None),
                "Market Cap": i.get("marketCap", None),
                "P/E": i.get("trailingPE", None),
                "Sector": i.get("sector", "N/A"),
            })
        except:
            pass

    df = pd.DataFrame(data)

    for col in ["Regular Price", "Pre-Market", "After-Hours", "Market Cap"]:
        if col in df.columns:
            df[col] = df[col].apply(fmt_money)

    st.dataframe(df, use_container_width=True)

    st.caption("Prices refresh every 60 seconds while the app is open. Data source: Yahoo Finance via yfinance.")