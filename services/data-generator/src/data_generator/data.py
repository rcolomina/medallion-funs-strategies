"""
Synthetic market data generation with known hidden regimes.

Used for validating that the HMM can recover the true hidden states.
"""

import numpy as np
import pandas as pd


def generate_regime_data(n_samples=2000, seed=42):
    """
    Generate synthetic market returns with 3 known regimes.

    Regimes:
        0 = Bull:     mu=+0.08%/day, sigma=0.8%   (high return, low vol)
        1 = Sideways: mu=+0.01%/day, sigma=1.2%   (flat, medium vol)
        2 = Bear:     mu=-0.10%/day, sigma=2.0%   (negative return, high vol)

    Parameters
    ----------
    n_samples : int
        Number of observations to generate.
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    returns : ndarray
        Daily returns.
    prices : ndarray
        Cumulative price series starting at 100.
    states : ndarray of int
        True hidden state at each time step.
    dates : DatetimeIndex
        Business day dates.
    """
    rng = np.random.RandomState(seed)

    true_means = [0.0008, 0.0001, -0.001]
    true_stds = [0.008, 0.012, 0.020]

    true_A = np.array(
        [
            [0.95, 0.04, 0.01],  # Bull -> Bull (sticky)
            [0.10, 0.80, 0.10],  # Sideways transitions freely
            [0.05, 0.10, 0.85],  # Bear -> Bear (sticky)
        ]
    )

    # Generate regime sequence using Markov chain
    states = np.zeros(n_samples, dtype=int)
    states[0] = 0  # Start in bull
    for t in range(1, n_samples):
        states[t] = rng.choice(3, p=true_A[states[t - 1]])

    # Generate returns from regime-specific distributions
    returns = np.zeros(n_samples)
    for t in range(n_samples):
        s = states[t]
        returns[t] = rng.normal(true_means[s], true_stds[s])

    # Convert to price series
    prices = 100 * np.exp(np.cumsum(returns))

    # Create dates
    dates = pd.date_range("2015-01-01", periods=n_samples, freq="B")

    return returns, prices, states, dates
