# FinLens

FinLens is a private Streamlit research terminal for personal equity research,
intrinsic valuation and portfolio risk analysis.

## What it includes

- Company price, market profile, historical fundamentals and peer comparison
- Enterprise-to-equity DCF with net debt and two-dimensional sensitivity
- Editable or CSV-imported portfolio holdings
- Expected return, volatility, Sharpe, Sortino, beta and tracking error
- Historical VaR/CVaR, maximum drawdown and historical stress windows
- Correlation, asset risk/return mapping and portfolio risk contribution
- Monte Carlo terminal-value scenarios
- OpenAI-powered investment memos using the Responses API
- Fail-closed password protection for the deployed app

## Local setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create an ignored `.env.local` file:

```text
FINLENS_PASSWORD=choose-a-long-private-password
OPENAI_API_KEY=your-project-key
OPENAI_MODEL=gpt-4.1-mini
```

Then run:

```bash
streamlit run app.py
```

## Deployment privacy

In Streamlit Community Cloud, add `FINLENS_PASSWORD`, `OPENAI_API_KEY`, and
optionally `OPENAI_MODEL` in **App settings → Secrets**. The app does not open
without `FINLENS_PASSWORD`. Do not commit `.env.local` or
`.streamlit/secrets.toml`.

Holdings are kept in the Streamlit session only. They are sent nowhere by the
portfolio analytics. The AI Memo page sends only the displayed public-company
context and analyst notes after the user explicitly clicks Generate.

## Holdings CSV

Use `sample_holdings.csv` as the schema:

```text
Ticker,Shares,Cost basis
MSFT,10,350
```

## Model limitations

Yahoo Finance data can be delayed, incomplete or restated. Expected returns,
VaR and Monte Carlo outputs are historical model estimates, not forecasts or
investment advice. See the in-app Methodology page for calculation details.
