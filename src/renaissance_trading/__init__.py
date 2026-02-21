"""
Renaissance Trading Methods
===========================

Hidden Markov Model implementations for market regime detection,
inspired by the quantitative methods of Renaissance Technologies.
"""

from renaissance_trading.hmm import GaussianHMM
from renaissance_trading.data import generate_regime_data

__all__ = ["GaussianHMM", "generate_regime_data"]
