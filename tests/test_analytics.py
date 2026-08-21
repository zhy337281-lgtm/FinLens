import numpy as np
import pandas as pd

from analytics import (
    dcf_valuation,
    max_drawdown,
    normalize_weights,
    portfolio_metrics,
    risk_contributions,
    terminal_summary,
)


def sample_returns():
    return pd.DataFrame(
        {
            "A": [0.01, -0.02, 0.005, 0.012, -0.004, 0.009],
            "B": [0.004, -0.006, 0.003, 0.007, -0.002, 0.005],
        },
        index=pd.date_range("2025-01-01", periods=6),
    )


def test_normalize_weights():
    assert np.allclose(normalize_weights([2, 3]), [0.4, 0.6])


def test_risk_contributions_sum_to_one():
    contributions = risk_contributions(sample_returns(), [0.6, 0.4])
    assert np.isclose(contributions.sum(), 1.0)


def test_portfolio_metrics_are_finite():
    metrics = portfolio_metrics(sample_returns(), [0.6, 0.4], risk_free_rate=0.0)
    assert np.isfinite(metrics.volatility)
    assert metrics.var_95_daily >= 0
    assert metrics.cvar_95_daily >= metrics.var_95_daily


def test_max_drawdown():
    returns = pd.Series([0.10, -0.20, 0.05])
    assert np.isclose(max_drawdown(returns), -0.20)


def test_dcf_bridges_enterprise_to_equity_value():
    result = dcf_valuation(
        latest_fcf=100,
        shares=10,
        net_debt=50,
        growth_rate=0.05,
        discount_rate=0.10,
        terminal_growth=0.02,
    )
    assert np.isclose(result["equity_value"], result["enterprise_value"] - 50)
    assert np.isclose(result["fair_value_per_share"], result["equity_value"] / 10)


def test_terminal_summary_probability():
    summary = terminal_summary(np.array([80.0, 90.0, 100.0, 120.0]), 100.0)
    assert summary["Probability of loss"] == 0.5
