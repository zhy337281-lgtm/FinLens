"""Pure portfolio and valuation analytics for FinLens.

The functions in this module intentionally avoid Streamlit so the financial
logic can be unit-tested independently from the interface.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import NormalDist
from typing import Iterable

import numpy as np
import pandas as pd


TRADING_DAYS = 252


@dataclass(frozen=True)
class PortfolioMetrics:
    expected_return: float
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    beta: float | None
    max_drawdown: float
    var_95_daily: float
    cvar_95_daily: float
    tracking_error: float | None
    information_ratio: float | None


def normalize_weights(weights: Iterable[float]) -> np.ndarray:
    values = np.asarray(list(weights), dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("At least one portfolio weight is required.")
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Portfolio weights must be finite and non-negative.")
    total = float(values.sum())
    if total <= 0:
        raise ValueError("Portfolio weights must sum to more than zero.")
    return values / total


def clean_prices(prices: pd.DataFrame) -> pd.DataFrame:
    if prices is None or prices.empty:
        return pd.DataFrame()
    cleaned = prices.copy()
    cleaned = cleaned.apply(pd.to_numeric, errors="coerce")
    cleaned = cleaned.loc[:, cleaned.notna().sum() >= 3]
    return cleaned.ffill().dropna(how="any")


def daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
    cleaned = clean_prices(prices)
    if cleaned.empty:
        return pd.DataFrame()
    return cleaned.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan).dropna()


def max_drawdown(returns: pd.Series) -> float:
    if returns is None or returns.empty:
        return float("nan")
    wealth = (1.0 + returns.fillna(0.0)).cumprod()
    drawdown = wealth / wealth.cummax() - 1.0
    return float(drawdown.min())


def expected_asset_returns(
    returns: pd.DataFrame,
    method: str = "Historical mean",
    span: int = 126,
) -> pd.Series:
    if returns.empty:
        return pd.Series(dtype=float)
    if method == "Exponentially weighted":
        estimate = returns.ewm(span=span, adjust=False).mean().iloc[-1]
    elif method == "Historical CAGR":
        years = len(returns) / TRADING_DAYS
        if years <= 0:
            return pd.Series(index=returns.columns, dtype=float)
        estimate = (1.0 + returns).prod() ** (1.0 / years) - 1.0
        return estimate.astype(float)
    else:
        estimate = returns.mean() * TRADING_DAYS
    return estimate.astype(float)


def portfolio_metrics(
    asset_returns: pd.DataFrame,
    weights: Iterable[float],
    benchmark_returns: pd.Series | None = None,
    risk_free_rate: float = 0.03,
    expected_method: str = "Historical mean",
) -> PortfolioMetrics:
    if asset_returns is None or asset_returns.empty:
        raise ValueError("Return history is required.")

    returns = asset_returns.dropna(how="any")
    normalized = normalize_weights(weights)
    if returns.shape[1] != normalized.size:
        raise ValueError("Weight count must match the number of assets.")

    portfolio = returns.mul(normalized, axis=1).sum(axis=1)
    expected_assets = expected_asset_returns(returns, expected_method)
    expected = float(expected_assets.to_numpy() @ normalized)
    volatility = float(portfolio.std(ddof=1) * np.sqrt(TRADING_DAYS))
    sharpe = (expected - risk_free_rate) / volatility if volatility > 0 else float("nan")

    downside = portfolio[portfolio < 0].std(ddof=1) * np.sqrt(TRADING_DAYS)
    sortino = (expected - risk_free_rate) / downside if pd.notna(downside) and downside > 0 else float("nan")

    var_95 = max(0.0, -float(portfolio.quantile(0.05)))
    tail = portfolio[portfolio <= portfolio.quantile(0.05)]
    cvar_95 = max(0.0, -float(tail.mean())) if not tail.empty else var_95

    beta = None
    tracking_error = None
    information_ratio = None
    if benchmark_returns is not None and not benchmark_returns.empty:
        joined = pd.concat(
            [portfolio.rename("portfolio"), benchmark_returns.rename("benchmark")], axis=1
        ).dropna()
        if len(joined) > 2:
            benchmark_variance = float(joined["benchmark"].var(ddof=1))
            if benchmark_variance > 0:
                beta = float(joined.cov().loc["portfolio", "benchmark"] / benchmark_variance)
            active = joined["portfolio"] - joined["benchmark"]
            tracking_error = float(active.std(ddof=1) * np.sqrt(TRADING_DAYS))
            active_return = float(active.mean() * TRADING_DAYS)
            if tracking_error > 0:
                information_ratio = active_return / tracking_error

    return PortfolioMetrics(
        expected_return=expected,
        volatility=volatility,
        sharpe_ratio=float(sharpe),
        sortino_ratio=float(sortino),
        beta=beta,
        max_drawdown=max_drawdown(portfolio),
        var_95_daily=var_95,
        cvar_95_daily=cvar_95,
        tracking_error=tracking_error,
        information_ratio=information_ratio,
    )


def risk_contributions(asset_returns: pd.DataFrame, weights: Iterable[float]) -> pd.Series:
    returns = asset_returns.dropna(how="any")
    normalized = normalize_weights(weights)
    covariance = returns.cov().to_numpy() * TRADING_DAYS
    portfolio_variance = float(normalized @ covariance @ normalized)
    if portfolio_variance <= 0:
        return pd.Series(0.0, index=returns.columns, name="Risk contribution")
    contribution = normalized * (covariance @ normalized) / portfolio_variance
    return pd.Series(contribution, index=returns.columns, name="Risk contribution")


def historical_stress(asset_returns: pd.DataFrame, weights: Iterable[float]) -> pd.DataFrame:
    returns = asset_returns.dropna(how="any")
    normalized = normalize_weights(weights)
    portfolio = returns.mul(normalized, axis=1).sum(axis=1)
    rolling_5d = (1.0 + portfolio).rolling(5).apply(np.prod, raw=True) - 1.0
    rolling_21d = (1.0 + portfolio).rolling(21).apply(np.prod, raw=True) - 1.0
    rows = [
        ("Worst day", portfolio.min()),
        ("Worst 5 trading days", rolling_5d.min()),
        ("Worst 21 trading days", rolling_21d.min()),
        ("Best day", portfolio.max()),
    ]
    return pd.DataFrame(rows, columns=["Scenario", "Portfolio return"])


def monte_carlo_terminal_values(
    asset_returns: pd.DataFrame,
    weights: Iterable[float],
    initial_value: float,
    years: int = 5,
    simulations: int = 5_000,
    seed: int = 42,
) -> np.ndarray:
    returns = asset_returns.dropna(how="any")
    normalized = normalize_weights(weights)
    portfolio = returns.mul(normalized, axis=1).sum(axis=1)
    annual_mean = float(portfolio.mean() * TRADING_DAYS)
    annual_volatility = float(portfolio.std(ddof=1) * np.sqrt(TRADING_DAYS))
    rng = np.random.default_rng(seed)
    shocks = rng.normal(size=(simulations, years))
    annual_log_returns = (
        annual_mean - 0.5 * annual_volatility**2
    ) + annual_volatility * shocks
    return initial_value * np.exp(annual_log_returns.sum(axis=1))


def terminal_summary(values: np.ndarray, initial_value: float) -> dict[str, float]:
    if values.size == 0:
        raise ValueError("Simulation results are required.")
    return {
        "Bear (10th percentile)": float(np.quantile(values, 0.10)),
        "Median": float(np.median(values)),
        "Bull (90th percentile)": float(np.quantile(values, 0.90)),
        "Probability of loss": float(np.mean(values < initial_value)),
    }


def dcf_valuation(
    latest_fcf: float,
    shares: float,
    net_debt: float,
    growth_rate: float,
    discount_rate: float,
    terminal_growth: float,
    forecast_years: int = 5,
) -> dict[str, float | list[float]]:
    values = [latest_fcf, shares, net_debt, growth_rate, discount_rate, terminal_growth]
    if not np.isfinite(values).all():
        raise ValueError("DCF inputs must be finite.")
    if latest_fcf <= 0 or shares <= 0:
        raise ValueError("Free cash flow and shares must be positive.")
    if discount_rate <= terminal_growth:
        raise ValueError("Discount rate must exceed terminal growth.")
    if forecast_years < 1:
        raise ValueError("Forecast period must be at least one year.")

    projected_fcfs = [latest_fcf * (1.0 + growth_rate) ** year for year in range(1, forecast_years + 1)]
    discounted_fcfs = [
        fcf / (1.0 + discount_rate) ** year
        for year, fcf in enumerate(projected_fcfs, start=1)
    ]
    terminal_value = projected_fcfs[-1] * (1.0 + terminal_growth) / (
        discount_rate - terminal_growth
    )
    present_terminal = terminal_value / (1.0 + discount_rate) ** forecast_years
    enterprise_value = float(sum(discounted_fcfs) + present_terminal)
    equity_value = enterprise_value - net_debt
    return {
        "projected_fcfs": projected_fcfs,
        "enterprise_value": enterprise_value,
        "equity_value": equity_value,
        "fair_value_per_share": equity_value / shares,
        "terminal_value_share": present_terminal / enterprise_value,
    }


def parametric_var(confidence: float, daily_mean: float, daily_volatility: float) -> float:
    if not 0.5 < confidence < 1:
        raise ValueError("Confidence must be between 0.5 and 1.0.")
    z_score = NormalDist().inv_cdf(1.0 - confidence)
    return max(0.0, -(daily_mean + z_score * daily_volatility))
