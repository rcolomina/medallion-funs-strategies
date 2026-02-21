"""
Tests for synthetic data generation.
"""

import numpy as np
import pandas as pd
import pytest

from data_generator.data import generate_regime_data


class TestGenerateRegimeData:
    def test_return_types(self):
        returns, prices, states, dates = generate_regime_data(n_samples=100)
        assert isinstance(returns, np.ndarray)
        assert isinstance(prices, np.ndarray)
        assert isinstance(states, np.ndarray)
        assert isinstance(dates, pd.DatetimeIndex)

    def test_shapes(self):
        n = 500
        returns, prices, states, dates = generate_regime_data(n_samples=n)
        assert returns.shape == (n,)
        assert prices.shape == (n,)
        assert states.shape == (n,)
        assert len(dates) == n

    def test_states_in_range(self):
        _, _, states, _ = generate_regime_data(n_samples=1000)
        assert set(states).issubset({0, 1, 2})

    def test_prices_positive(self):
        _, prices, _, _ = generate_regime_data(n_samples=1000)
        assert np.all(prices > 0)

    def test_reproducibility(self):
        r1, _, s1, _ = generate_regime_data(seed=123)
        r2, _, s2, _ = generate_regime_data(seed=123)
        np.testing.assert_array_equal(r1, r2)
        np.testing.assert_array_equal(s1, s2)

    def test_different_seeds_differ(self):
        r1, _, _, _ = generate_regime_data(seed=1)
        r2, _, _, _ = generate_regime_data(seed=2)
        assert not np.array_equal(r1, r2)

    def test_regime_means_direction(self):
        """Bull returns should be higher than bear returns on average."""
        returns, _, states, _ = generate_regime_data(n_samples=10000, seed=42)
        bull_mean = returns[states == 0].mean()
        bear_mean = returns[states == 2].mean()
        assert bull_mean > bear_mean

    def test_regime_volatility_ordering(self):
        """Bear regime should have higher volatility than bull."""
        returns, _, states, _ = generate_regime_data(n_samples=10000, seed=42)
        bull_vol = returns[states == 0].std()
        bear_vol = returns[states == 2].std()
        assert bear_vol > bull_vol

    def test_starts_in_bull(self):
        _, _, states, _ = generate_regime_data()
        assert states[0] == 0

    def test_dates_are_business_days(self):
        _, _, _, dates = generate_regime_data(n_samples=100)
        # No Saturdays (5) or Sundays (6)
        weekdays = dates.weekday
        assert not np.any(weekdays >= 5)
