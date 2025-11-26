"""
Package Import
"""
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import quantstats as qs
import gurobipy as gp
import warnings
import argparse
import sys

"""
Project Setup
"""
warnings.simplefilter(action="ignore", category=FutureWarning)

assets = [
    "SPY",
    "XLB",
    "XLC",
    "XLE",
    "XLF",
    "XLI",
    "XLK",
    "XLP",
    "XLRE",
    "XLU",
    "XLV",
    "XLY",
]

# Initialize Bdf and df
Bdf = pd.DataFrame()
for asset in assets:
    raw = yf.download(asset, start="2012-01-01", end="2024-04-01", auto_adjust = False)
    Bdf[asset] = raw['Adj Close']

df = Bdf.loc["2019-01-01":"2024-04-01"]

"""
Strategy Creation

Create your own strategy, you can add parameter but please remain "price" and "exclude" unchanged
"""


class MyPortfolio:
    """
    NOTE: You can modify the initialization function
    """

    def __init__(self, price, exclude, lookback=50, gamma=0):
        self.price = price
        self.returns = price.pct_change().fillna(0)
        self.exclude = exclude
        self.lookback = lookback
        self.gamma = gamma

    def calculate_weights(self):
        # Get the assets by excluding the specified column
        assets = self.price.columns[self.price.columns != self.exclude]

        # Calculate the portfolio weights
        self.portfolio_weights = pd.DataFrame(
            index=self.price.index, columns=self.price.columns
        )

        """
        TODO: Complete Task 4 Below
        """

        top_k = 3  # Each time select top k momentum sectors
        defensive = ["XLP", "XLU", "XLV"]  # defensive sector

        for current_date in self.price.index:
            hist = self._get_historical_window(current_date, assets)
            if hist is None:
                continue

            hist_ret = hist.pct_change().dropna()
            if hist_ret.empty:
                continue

            candidates = self._select_candidates(hist, hist_ret, assets, top_k, defensive)
            if not candidates:
                continue

            weights = self._calculate_candidate_weights(hist_ret, candidates)
            self._assign_weights(current_date, candidates, weights)

        """
        TODO: Complete Task 4 Above
        """

        self.portfolio_weights.ffill(inplace=True)
        self.portfolio_weights.fillna(0, inplace=True)

    def _get_historical_window(self, current_date, assets):
        """Get historical price window for the lookback period."""
        hist = self.price.loc[:current_date, assets]
        if len(hist) < self.lookback:
            return None
        return hist.iloc[-self.lookback:]

    def _select_candidates(self, hist, hist_ret, assets, top_k, defensive):
        """Select candidate assets based on momentum strategy."""
        momentum = hist.iloc[-1] / hist.iloc[0] - 1.0
        pos_mom = momentum[momentum > 0].sort_values(ascending=False)

        if len(pos_mom) == 0:
            # If all are declining, switch to defensive sectors
            return [a for a in defensive if a in assets]
        else:
            return list(pos_mom.index[:top_k])

    def _calculate_candidate_weights(self, hist_ret, candidates):
        """Calculate weights for candidates using inverse volatility."""
        vol = hist_ret[candidates].std()
        inv_vol = 1.0 / vol.replace(0, np.nan)

        if inv_vol.isna().all():
            return pd.Series(1.0 / len(candidates), index=candidates)

        inv_vol = inv_vol.fillna(inv_vol.mean()).clip(lower=1e-6)
        return inv_vol / inv_vol.sum()

    def _assign_weights(self, current_date, candidates, weights):
        """Assign calculated weights to portfolio for given date."""
        self.portfolio_weights.loc[current_date, :] = 0.0
        self.portfolio_weights.loc[current_date, candidates] = weights.values
        self.portfolio_weights.loc[current_date, self.exclude] = 0.0

    def calculate_portfolio_returns(self):
        # Ensure weights are calculated
        if not hasattr(self, "portfolio_weights"):
            self.calculate_weights()

        # Calculate the portfolio returns
        self.portfolio_returns = self.returns.copy()
        assets = self.price.columns[self.price.columns != self.exclude]
        self.portfolio_returns["Portfolio"] = (
            self.portfolio_returns[assets]
            .mul(self.portfolio_weights[assets])
            .sum(axis=1)
        )

    def get_results(self):
        # Ensure portfolio returns are calculated
        if not hasattr(self, "portfolio_returns"):
            self.calculate_portfolio_returns()

        return self.portfolio_weights, self.portfolio_returns


if __name__ == "__main__":
    # Import grading system (protected file in GitHub Classroom)
    from grader_2 import AssignmentJudge
    
    parser = argparse.ArgumentParser(
        description="Introduction to Fintech Assignment 3 Part 12"
    )

    parser.add_argument(
        "--score",
        action="append",
        help="Score for assignment",
    )

    parser.add_argument(
        "--allocation",
        action="append",
        help="Allocation for asset",
    )

    parser.add_argument(
        "--performance",
        action="append",
        help="Performance for portfolio",
    )

    parser.add_argument(
        "--report", action="append", help="Report for evaluation metric"
    )

    parser.add_argument(
        "--cumulative", action="append", help="Cumulative product result"
    )

    args = parser.parse_args()

    judge = AssignmentJudge()
    
    # All grading logic is protected in grader_2.py
    judge.run_grading(args)
