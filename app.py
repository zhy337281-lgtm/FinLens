import os
import numbers
import requests
import pandas as pd
import yfinance as yf
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# -----------------------
# Optional dependencies
# -----------------------

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

try:
    from streamlit_autorefresh import st_autorefresh
except ImportError:
    def st_autorefresh(*args, **kwargs):
        return None


# -----------------------
# OpenAI Setup
# -----------------------

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client = None

if OPENAI_API_KEY and OpenAI is not None:
    try:
        client = OpenAI(api_key=OPENAI_API_KEY)
    except Exception:
        client = None


# -----------------------
# Streamlit Setup
# -----------------------

st.set_page_config(
    page_title="FinLens AI",
    page_icon="📈",
    layout="wide"
)


# ============================================================
# Helper Functions
# ============================================================

def valid_number(x):
    try:
        return x is not None and not pd.isna(x)
    except Exception:
        return False


def fmt_money(x):
    try:
        if not valid_number(x):
            return "N/A"

        x = float(x)

        if abs(x) >= 1_000_000_000_000:
            return f"${x / 1_000_000_000_000:.2f}T"

        if abs(x) >= 1_000_000_000:
            return f"${x / 1_000_000_000:.2f}B"

        if abs(x) >= 1_000_000:
            return f"${x / 1_000_000:.2f}M"

        return f"${x:,.2f}"

    except Exception:
        return "N/A"


def fmt_pct(x):
    try:
        if not valid_number(x):
            return "N/A"

        return f"{float(x) * 100:.1f}%"

    except Exception:
        return "N/A"


def safe_get(info, key, default=None):
    try:
        value = info.get(key, default)

        if value in ["", None]:
            return default

        if pd.isna(value):
            return default

        return value

    except Exception:
        return default


def safe_divide(a, b):
    try:
        if not valid_number(a) or not valid_number(b):
            return None

        if float(b) == 0:
            return None

        return float(a) / float(b)

    except Exception:
        return None


def get_statement_value(df, row_name, column):
    """
    Safely retrieve a financial statement item.
    """
    try:
        if df is None or df.empty:
            return None

        if row_name not in df.index:
            return None

        if column not in df.columns:
            return None

        value = df.loc[row_name, column]

        if not valid_number(value):
            return None

        return float(value)

    except Exception:
        return None


# ============================================================
# Ticker Search
# ============================================================

def search_ticker(query):
    query = query.strip()

    if not query:
        return "ORCL"

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

    # If user already entered a likely ticker
    if query.isupper() and len(query) <= 6:
        return query

    try:
        url = "https://query2.finance.yahoo.com/v1/finance/search"

        params = {
            "q": query,
            "quotes_count": 5,
            "news_count": 0
        }

        headers = {
            "User-Agent": "Mozilla/5.0"
        }

        response = requests.get(
            url,
            params=params,
            headers=headers,
            timeout=8
        )

        response.raise_for_status()

        data = response.json()

        quotes = data.get("quotes", [])

        for quote in quotes:
            if (
                quote.get("quoteType") == "EQUITY"
                and quote.get("symbol")
            ):
                return quote["symbol"]

    except Exception:
        pass

    return query.upper()


# ============================================================
# Company / Market Data
# ============================================================

def get_logo_url(info):
    website = safe_get(info, "website")

    if not website:
        return None

    try:
        domain = (
            website
            .replace("https://", "")
            .replace("http://", "")
            .split("/")[0]
        )

        return f"https://logo.clearbit.com/{domain}"

    except Exception:
        return None


@st.cache_resource(show_spinner=False)
def get_company(ticker):
    return yf.Ticker(ticker)


@st.cache_data(ttl=300, show_spinner=False)
def get_company_info(ticker):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        if not info:
            return {}

        return info

    except Exception:
        return {}


@st.cache_data(ttl=600, show_spinner=False)
def get_financial_data(ticker):
    try:
        stock = yf.Ticker(ticker)

        income = stock.financials
        balance_sheet = stock.balance_sheet
        cashflow = stock.cashflow

        return income, balance_sheet, cashflow

    except Exception:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()


# ============================================================
# Financial Summary
# ============================================================

def get_financial_summary(income, bs, cf):

    rows = []

    if income is None or income.empty:
        return pd.DataFrame()

    for year in income.columns:

        revenue = get_statement_value(
            income,
            "Total Revenue",
            year
        )

        gross_profit = get_statement_value(
            income,
            "Gross Profit",
            year
        )

        ebitda = get_statement_value(
            income,
            "EBITDA",
            year
        )

        operating_income = get_statement_value(
            income,
            "Operating Income",
            year
        )

        net_income = get_statement_value(
            income,
            "Net Income",
            year
        )

        operating_cf = get_statement_value(
            cf,
            "Operating Cash Flow",
            year
        )

        capex = get_statement_value(
            cf,
            "Capital Expenditure",
            year
        )

        fcf = get_statement_value(
            cf,
            "Free Cash Flow",
            year
        )

        total_debt = get_statement_value(
            bs,
            "Total Debt",
            year
        )

        cash = get_statement_value(
            bs,
            "Cash And Cash Equivalents",
            year
        )

        # Fallback FCF calculation
        if fcf is None:
            if operating_cf is not None and capex is not None:
                # CapEx often comes from Yahoo as negative
                fcf = operating_cf + capex

        try:
            year_display = pd.Timestamp(year).strftime("%Y-%m-%d")
        except Exception:
            year_display = str(year)

        rows.append({
            "Year": year_display,
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

            "Gross Margin": safe_divide(
                gross_profit,
                revenue
            ),

            "EBITDA Margin": safe_divide(
                ebitda,
                revenue
            ),

            "Operating Margin": safe_divide(
                operating_income,
                revenue
            ),

            "Net Margin": safe_divide(
                net_income,
                revenue
            ),

            "FCF Margin": safe_divide(
                fcf,
                revenue
            ),
        })

    return pd.DataFrame(rows)


# ============================================================
# DCF
# ============================================================

def simple_dcf(
    latest_fcf,
    growth,
    discount_rate,
    terminal_growth,
    shares
):

    try:

        if not valid_number(latest_fcf):
            return None

        if not valid_number(shares):
            return None

        if shares <= 0:
            return None

        if discount_rate <= terminal_growth:
            return None

        fcfs = []

        for year in range(1, 6):
            future_fcf = latest_fcf * ((1 + growth) ** year)
            fcfs.append(future_fcf)

        pv_fcfs = []

        for year, fcf in enumerate(fcfs, start=1):
            pv = fcf / ((1 + discount_rate) ** year)
            pv_fcfs.append(pv)

        terminal_value = (
            fcfs[-1]
            * (1 + terminal_growth)
            / (discount_rate - terminal_growth)
        )

        pv_terminal = (
            terminal_value
            / ((1 + discount_rate) ** 5)
        )

        total_value = (
            sum(pv_fcfs)
            + pv_terminal
        )

        fair_value_per_share = (
            total_value
            / shares
        )

        return fair_value_per_share

    except Exception:
        return None


# ============================================================
# OpenAI
# ============================================================

def ai_report(prompt):

    if client is None:

        return """
### OpenAI API is not configured

The rest of FinLens works normally.

To enable AI features:

1. Create a `.env` file in your project directory.
2. Add:

`OPENAI_API_KEY=your_api_key_here`

3. Restart Streamlit.
"""

    try:

        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {
                    "role": "system",
                    "content":
                        "You are a professional equity research analyst."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        return response.choices[0].message.content

    except Exception as e:

        return f"""
### AI analysis failed

Error:

`{e}`
"""


# ============================================================
# Competitor Data
# ============================================================

@st.cache_data(ttl=300, show_spinner=False)
def get_competitor_table(tickers):

    rows = []

    for ticker in tickers:

        try:

            info = yf.Ticker(ticker).info

            rows.append({
                "Ticker": ticker,
                "Name": info.get(
                    "shortName",
                    "N/A"
                ),
                "Price": info.get(
                    "currentPrice"
                ),
                "Market Cap": info.get(
                    "marketCap"
                ),
                "P/E": info.get(
                    "trailingPE"
                ),
                "Forward P/E": info.get(
                    "forwardPE"
                ),
                "Revenue Growth": info.get(
                    "revenueGrowth"
                ),
                "Profit Margin": info.get(
                    "profitMargins"
                ),
                "ROE": info.get(
                    "returnOnEquity"
                ),
            })

        except Exception:
            continue

    return pd.DataFrame(rows)


def get_news(stock):

    try:
        news = stock.news

        if not news:
            return []

        return news[:10]

    except Exception:
        return []


# ============================================================
# Sidebar
# ============================================================

st.sidebar.title("📈 FinLens AI")

default_company = st.session_state.get(
    "company",
    "Oracle"
)

query = st.sidebar.text_input(
    "Search company or ticker",
    value=default_company
)

ticker = search_ticker(query)

page = st.sidebar.radio(
    "Pages",
    [
        "Home",
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


# ============================================================
# Home Page
# ============================================================

if page == "Home":

    st.title("📈 FinLens AI")

    st.markdown(
        """
### AI-powered Equity Research Platform

Professional Equity Research for:

- Investors
- Finance Students
- Analysts
- Equity Research Learners
"""
    )

    st.write("")

    home_query = st.text_input(
        "Search Company",
        placeholder="Oracle / Microsoft / NVIDIA"
    )

    if st.button("Start Research"):

        if home_query.strip():

            st.session_state["company"] = (
                home_query.strip()
            )

            st.success(
                "Company selected. "
                "Open Dashboard from the sidebar."
            )

    st.subheader("Popular Companies")

    c1, c2, c3, c4, c5 = st.columns(5)

    if c1.button("Oracle"):
        st.session_state["company"] = "Oracle"
        st.rerun()

    if c2.button("Microsoft"):
        st.session_state["company"] = "Microsoft"
        st.rerun()

    if c3.button("NVIDIA"):
        st.session_state["company"] = "NVIDIA"
        st.rerun()

    if c4.button("Apple"):
        st.session_state["company"] = "Apple"
        st.rerun()

    if c5.button("TSMC"):
        st.session_state["company"] = "TSMC"
        st.rerun()

    st.stop()


# ============================================================
# Load Company Data
# ============================================================

with st.spinner(
    f"Loading {ticker} data..."
):

    stock = get_company(ticker)

    info = get_company_info(ticker)

    income, bs, cf = get_financial_data(
        ticker
    )

    summary_df = get_financial_summary(
        income,
        bs,
        cf
    )


# ============================================================
# Validate Company
# ============================================================

if not info:

    st.error(
        f"Could not load data for `{ticker}`. "
        "Please check the company name or ticker."
    )

    st.stop()


# ============================================================
# Core Company Information
# ============================================================

company_name = safe_get(
    info,
    "longName",
    ticker
)

price = safe_get(
    info,
    "currentPrice",
    safe_get(
        info,
        "regularMarketPrice"
    )
)

regular_price = safe_get(
    info,
    "regularMarketPrice",
    price
)

pre_price = safe_get(
    info,
    "preMarketPrice"
)

post_price = safe_get(
    info,
    "postMarketPrice"
)

market_cap = safe_get(
    info,
    "marketCap"
)

pe = safe_get(
    info,
    "trailingPE"
)

forward_pe = safe_get(
    info,
    "forwardPE"
)

eps = safe_get(
    info,
    "trailingEps"
)

forward_eps = safe_get(
    info,
    "forwardEps"
)

shares = safe_get(
    info,
    "sharesOutstanding"
)

sector = safe_get(
    info,
    "sector",
    "N/A"
)

industry = safe_get(
    info,
    "industry",
    "N/A"
)

target_mean = safe_get(
    info,
    "targetMeanPrice"
)

target_high = safe_get(
    info,
    "targetHighPrice"
)

target_low = safe_get(
    info,
    "targetLowPrice"
)


# ============================================================
# Header
# ============================================================

logo_url = get_logo_url(info)

col_title, col_logo = st.columns(
    [8, 1]
)

with col_title:

    st.title(
        f"{company_name} ({ticker})"
    )

    st.caption(
        f"{sector} | {industry}"
    )

with col_logo:

    if logo_url:

        st.markdown(
            f"""
<img
src="{logo_url}"
width="70"
onerror="this.style.display='none'"
>
""",
            unsafe_allow_html=True
        )


# ============================================================
# Dashboard
# ============================================================

if page == "Dashboard":

    st_autorefresh(
        interval=60_000,
        key="dashboard_refresh"
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Current Price",
        fmt_money(price)
    )

    col2.metric(
        "Market Cap",
        fmt_money(market_cap)
    )

    col3.metric(
        "P/E",
        f"{pe:.2f}"
        if valid_number(pe)
        else "N/A"
    )

    col4.metric(
        "Forward P/E",
        f"{forward_pe:.2f}"
        if valid_number(forward_pe)
        else "N/A"
    )

    st.subheader(
        "Financial Snapshot"
    )

    if summary_df.empty:

        st.warning(
            "Financial statement data "
            "is unavailable."
        )

    else:

        display_df = summary_df.copy()

        money_cols = [
            "Revenue",
            "Gross Profit",
            "EBITDA",
            "Operating Income",
            "Net Income",
            "Operating Cash Flow",
            "CapEx",
            "Free Cash Flow",
            "Total Debt",
            "Cash"
        ]

        pct_cols = [
            "Gross Margin",
            "EBITDA Margin",
            "Operating Margin",
            "Net Margin",
            "FCF Margin"
        ]

        for col in money_cols:

            if col in display_df.columns:
                display_df[col] = (
                    display_df[col]
                    .apply(fmt_money)
                )

        for col in pct_cols:

            if col in display_df.columns:
                display_df[col] = (
                    display_df[col]
                    .apply(fmt_pct)
                )

        st.dataframe(
            display_df,
            use_container_width=True
        )


# ============================================================
# Financial Statements
# ============================================================

elif page == "Financial Statements":

    def format_statement(df):

        if df is None or df.empty:
            return pd.DataFrame()

        formatted = df.T.copy()

        for col in formatted.columns:

            formatted[col] = (
                formatted[col]
                .apply(
                    lambda x:
                        fmt_money(x)
                        if isinstance(
                            x,
                            numbers.Number
                        )
                        else x
                )
            )

        return formatted


    st.subheader(
        "Income Statement"
    )

    if income.empty:
        st.warning("No income statement available.")
    else:
        st.dataframe(
            format_statement(income),
            use_container_width=True
        )


    st.subheader(
        "Balance Sheet"
    )

    if bs.empty:
        st.warning("No balance sheet available.")
    else:
        st.dataframe(
            format_statement(bs),
            use_container_width=True
        )


    st.subheader(
        "Cash Flow Statement"
    )

    if cf.empty:
        st.warning("No cash flow statement available.")
    else:
        st.dataframe(
            format_statement(cf),
            use_container_width=True
        )


# ============================================================
# Market Data
# ============================================================

elif page == "Market Data":

    st_autorefresh(
        interval=60_000,
        key="market_refresh"
    )

    pre_change = (
        safe_divide(
            pre_price - regular_price,
            regular_price
        )
        if valid_number(pre_price)
        and valid_number(regular_price)
        else None
    )

    post_change = (
        safe_divide(
            post_price - regular_price,
            regular_price
        )
        if valid_number(post_price)
        and valid_number(regular_price)
        else None
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Regular Market Price",
        fmt_money(regular_price)
    )

    col2.metric(
        "Pre-Market Price",
        fmt_money(pre_price),
        fmt_pct(pre_change)
        if valid_number(pre_change)
        else None
    )

    col3.metric(
        "After-Hours Price",
        fmt_money(post_price),
        fmt_pct(post_change)
        if valid_number(post_change)
        else None
    )

    col4, col5, col6 = st.columns(3)

    col4.metric(
        "Market Cap",
        fmt_money(market_cap)
    )

    col5.metric(
        "Beta",
        safe_get(info, "beta", "N/A")
    )

    col6.metric(
        "Volume",
        safe_get(info, "volume", "N/A")
    )

    col7, col8, col9 = st.columns(3)

    col7.metric(
        "52W High",
        fmt_money(
            safe_get(
                info,
                "fiftyTwoWeekHigh"
            )
        )
    )

    col8.metric(
        "52W Low",
        fmt_money(
            safe_get(
                info,
                "fiftyTwoWeekLow"
            )
        )
    )

    col9.metric(
        "Dividend Yield",
        fmt_pct(
            safe_get(
                info,
                "dividendYield"
            )
        )
    )

    st.caption(
        "Pre-market and after-hours "
        "data availability depends on "
        "Yahoo Finance coverage."
    )


# ============================================================
# Charts
# ============================================================

elif page == "Charts & K-Line":

    period = st.selectbox(
        "Period",
        [
            "1mo",
            "3mo",
            "6mo",
            "1y",
            "5y"
        ],
        index=3
    )

    try:

        hist = stock.history(
            period=period
        )

    except Exception:

        hist = pd.DataFrame()


    if hist.empty:

        st.warning(
            "Price history unavailable."
        )

    else:

        st.subheader(
            "Line Chart"
        )

        fig = px.line(
            hist,
            x=hist.index,
            y="Close",
            title=f"{ticker} Closing Price"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


        st.subheader(
            "K-Line / Candlestick Chart"
        )

        fig = go.Figure(
            data=[
                go.Candlestick(
                    x=hist.index,
                    open=hist["Open"],
                    high=hist["High"],
                    low=hist["Low"],
                    close=hist["Close"]
                )
            ]
        )

        fig.update_layout(
            title=f"{ticker} Candlestick Chart",
            xaxis_rangeslider_visible=False
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


# ============================================================
# DCF
# ============================================================

elif page == "DCF Valuation":

    st.subheader(
        "DCF Valuation"
    )

    latest_fcf = None

    if (
        not summary_df.empty
        and "Free Cash Flow"
        in summary_df.columns
    ):

        fcf_series = (
            summary_df["Free Cash Flow"]
            .dropna()
        )

        if not fcf_series.empty:
            latest_fcf = (
                float(fcf_series.iloc[0])
            )


    st.write(
        "Latest FCF:",
        fmt_money(latest_fcf)
    )

    growth = st.slider(
        "FCF Growth Rate",
        0.00,
        0.20,
        0.06
    )

    discount_rate = st.slider(
        "Discount Rate / WACC",
        0.05,
        0.15,
        0.09
    )

    terminal_growth = st.slider(
        "Terminal Growth Rate",
        0.00,
        0.05,
        0.025
    )

    fair_value = simple_dcf(
        latest_fcf,
        growth,
        discount_rate,
        terminal_growth,
        shares
    )

    col1, col2 = st.columns(2)

    col1.metric(
        "DCF Fair Value / Share",
        fmt_money(fair_value)
    )

    col2.metric(
        "Current Price",
        fmt_money(price)
    )

    st.caption(
        "This simplified DCF is for "
        "educational purposes and is "
        "not investment advice."
    )


# ============================================================
# P/E EPS Valuation
# ============================================================

elif page == "P/E EPS Valuation":

    st.subheader(
        "P/E × EPS Valuation"
    )

    base_eps = (
        float(forward_eps)
        if valid_number(forward_eps)
        else (
            float(eps)
            if valid_number(eps)
            else 5.0
        )
    )

    st.write(
        "Forward EPS:",
        forward_eps
        if valid_number(forward_eps)
        else "N/A"
    )

    st.write(
        "Trailing EPS:",
        eps
        if valid_number(eps)
        else "N/A"
    )

    bear_pe = st.slider(
        "Bear Case P/E",
        5,
        60,
        15
    )

    base_pe = st.slider(
        "Base Case P/E",
        5,
        60,
        25
    )

    bull_pe = st.slider(
        "Bull Case P/E",
        5,
        80,
        35
    )

    custom_eps = st.number_input(
        "Adjust EPS",
        value=base_eps
    )

    bear_price = (
        bear_pe
        * custom_eps
    )

    base_price = (
        base_pe
        * custom_eps
    )

    bull_price = (
        bull_pe
        * custom_eps
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Bear Case Price",
        fmt_money(bear_price)
    )

    col2.metric(
        "Base Case Price",
        fmt_money(base_price)
    )

    col3.metric(
        "Bull Case Price",
        fmt_money(bull_price)
    )

    valuation_df = pd.DataFrame({
        "Case": [
            "Bear",
            "Base",
            "Bull"
        ],
        "EPS": [
            custom_eps,
            custom_eps,
            custom_eps
        ],
        "P/E": [
            bear_pe,
            base_pe,
            bull_pe
        ],
        "Implied Price": [
            bear_price,
            base_price,
            bull_price
        ]
    })

    st.dataframe(
        valuation_df,
        use_container_width=True
    )


# ============================================================
# Analyst Targets
# ============================================================

elif page == "Analyst Price Targets":

    st.subheader(
        "Analyst Price Targets"
    )

    target_df = pd.DataFrame({
        "Estimate": [
            "Low Target",
            "Mean Target",
            "High Target"
        ],
        "Price": [
            target_low,
            target_mean,
            target_high
        ]
    })

    st.dataframe(
        target_df,
        use_container_width=True
    )

    valid_targets = (
        target_df
        .dropna(subset=["Price"])
    )

    if not valid_targets.empty:

        fig = px.bar(
            valid_targets,
            x="Estimate",
            y="Price",
            title="Analyst Target Price Range"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    else:

        st.warning(
            "Analyst price target data "
            "is unavailable."
        )


# ============================================================
# Business & Moat
# ============================================================

elif page == "Business & Moat":

    st.subheader(
        "Company Introduction, Moat & Fundamentals"
    )

    prompt = f"""
Company: {company_name}
Ticker: {ticker}
Sector: {sector}
Industry: {industry}
Market Cap: {market_cap}
P/E: {pe}

Write a professional equity research analysis covering:

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

    if st.button(
        "Generate Business & Moat Analysis"
    ):

        with st.spinner(
            "Generating..."
        ):

            st.markdown(
                ai_report(prompt)
            )


# ============================================================
# Competitors & Risk
# ============================================================

elif page == "Competitors & Risk":

    st.subheader(
        "Competitor Comparison"
    )

    default_peers = (
        "MSFT,AMZN,SAP,CRM"
    )

    peer_input = st.text_input(
        "Peer tickers",
        default_peers
    )

    peer_tickers = [
        x.strip().upper()
        for x
        in peer_input.split(",")
        if x.strip()
    ]

    comp_df = get_competitor_table(
        [ticker] + peer_tickers
    )

    display_comp = (
        comp_df.copy()
    )

    if not display_comp.empty:

        if "Market Cap" in display_comp:
            display_comp["Market Cap"] = (
                display_comp["Market Cap"]
                .apply(fmt_money)
            )

        for col in [
            "Revenue Growth",
            "Profit Margin",
            "ROE"
        ]:

            if col in display_comp:

                display_comp[col] = (
                    display_comp[col]
                    .apply(fmt_pct)
                )

        st.dataframe(
            display_comp,
            use_container_width=True
        )

    else:

        st.warning(
            "Competitor data unavailable."
        )


    st.subheader(
        "Risk Analysis"
    )

    prompt = f"""
Company: {company_name}
Ticker: {ticker}
Sector: {sector}
Industry: {industry}

Competitor Data:

{comp_df.to_string()}

Write a company-specific risk analysis:

1. Competitive risk
2. Margin risk
3. Valuation risk
4. Business model risk
5. Macroeconomic risk
6. Execution risk

Also compare the company against its competitors.
"""

    if st.button(
        "Generate Risk Analysis"
    ):

        with st.spinner(
            "Generating risk analysis..."
        ):

            st.markdown(
                ai_report(prompt)
            )


# ============================================================
# News
# ============================================================

elif page == "News & AI Analysis":

    st.subheader(
        "Latest Company News"
    )

    news = get_news(stock)

    news_rows = []

    for item in news:

        try:

            content = item.get(
                "content",
                item
            )

            title = content.get(
                "title",
                item.get(
                    "title",
                    "N/A"
                )
            )

            provider = content.get(
                "provider",
                {}
            )

            publisher = (
                provider.get(
                    "displayName",
                    item.get(
                        "publisher",
                        "N/A"
                    )
                )
            )

            canonical_url = (
                content.get(
                    "canonicalUrl",
                    {}
                )
            )

            link = (
                canonical_url.get(
                    "url",
                    item.get(
                        "link"
                    )
                )
            )

            news_rows.append({
                "Title": title,
                "Publisher": publisher,
                "Link": link
            })

        except Exception:
            continue


    news_df = pd.DataFrame(
        news_rows
    )

    if news_df.empty:

        st.warning(
            "No news available."
        )

    else:

        st.dataframe(
            news_df,
            use_container_width=True
        )


    prompt = f"""
Company: {company_name}
Ticker: {ticker}

Latest news:

{news_df.to_string()}

Summarize the latest news and explain:

1. What happened
2. Positive / negative / neutral
3. Impact on revenue
4. Impact on margins
5. Impact on valuation
6. Impact on investor sentiment
7. Key risks to monitor
"""

    if st.button(
        "Generate News AI Analysis"
    ):

        with st.spinner(
            "Analyzing news..."
        ):

            st.markdown(
                ai_report(prompt)
            )


# ============================================================
# AI Equity Research Report
# ============================================================

elif page == "AI Report":

    st.subheader(
        "AI Equity Research Report"
    )

    financial_text = (
        summary_df.to_string()
        if not summary_df.empty
        else "Financial data unavailable."
    )

    prompt = f"""
Analyze this company as a professional equity research analyst.

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
3. Revenue Analysis
4. Profitability Analysis
5. Cash Flow Analysis
6. Moat Analysis
7. Growth Drivers
8. Bull Case
9. Bear Case
10. Key Risks
11. Investment Conclusion

Do not provide personalized financial advice.
"""

    if st.button(
        "Generate AI Report"
    ):

        with st.spinner(
            "Generating AI Report..."
        ):

            st.markdown(
                ai_report(prompt)
            )


# ============================================================
# Watchlist
# ============================================================

elif page == "Watchlist":

    st_autorefresh(
        interval=60_000,
        key="watchlist_refresh"
    )

    st.subheader(
        "Real-Time Watchlist"
    )

    tickers_input = st.text_area(
        "Enter tickers separated by commas",
        "ORCL,MSFT,NVDA,AAPL,TSM"
    )

    tickers = [
        ticker.strip().upper()
        for ticker
        in tickers_input.split(",")
        if ticker.strip()
    ]

    data = []

    for ticker_item in tickers:

        try:

            info_item = (
                yf.Ticker(
                    ticker_item
                ).info
            )

            data.append({
                "Ticker": ticker_item,
                "Name": info_item.get(
                    "shortName",
                    "N/A"
                ),
                "Regular Price":
                    info_item.get(
                        "regularMarketPrice",
                        info_item.get(
                            "currentPrice"
                        )
                    ),
                "Pre-Market":
                    info_item.get(
                        "preMarketPrice"
                    ),
                "After-Hours":
                    info_item.get(
                        "postMarketPrice"
                    ),
                "Market Cap":
                    info_item.get(
                        "marketCap"
                    ),
                "P/E":
                    info_item.get(
                        "trailingPE"
                    ),
                "Sector":
                    info_item.get(
                        "sector",
                        "N/A"
                    ),
            })

        except Exception:
            continue


    df = pd.DataFrame(data)

    if df.empty:

        st.warning(
            "Watchlist data unavailable."
        )

    else:

        for col in [
            "Regular Price",
            "Pre-Market",
            "After-Hours",
            "Market Cap"
        ]:

            if col in df.columns:

                df[col] = (
                    df[col]
                    .apply(fmt_money)
                )

        st.dataframe(
            df,
            use_container_width=True
        )

    st.caption(
        "Data source: Yahoo Finance via yfinance. "
        "Market data may be delayed or unavailable."
    )