"""
Trading engine — HMM regime detection and Kelly criterion sizing.

Extracted from the original main.py pipeline.
"""

import numpy as np

from renaissance_core.hmm import GaussianHMM


REGIME_NAMES = ["Bear", "Sideways", "Bull"]


class TradingEngine:
    """
    Runs the HMM regime detection pipeline:
    fit → decode → predict_proba → Kelly sizing.
    """

    def __init__(self, n_states: int = 3, n_iter: int = 200, tol: float = 1e-8):
        self.model = GaussianHMM(n_states=n_states, n_iter=n_iter, tol=tol)
        self._fitted = False
        self._state_order: np.ndarray | None = None
        self._returns: np.ndarray | None = None

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    def fit(self, returns: np.ndarray) -> dict:
        """Fit the HMM on observed returns."""
        returns = np.asarray(returns, dtype=np.float64)
        self._returns = returns
        self.model.fit(returns)
        # Sort states by mean: index 0 = lowest mean (Bear), last = highest (Bull)
        self._state_order = np.argsort(self.model.means)
        self._fitted = True
        return self.status()

    def status(self) -> dict:
        """Return model state information."""
        if not self._fitted:
            return {"fitted": False}
        return {
            "fitted": True,
            "n_states": self.model.n_states,
            "n_observations": len(self._returns),
            "log_likelihood": float(self.model.log_likelihood_),
            "means": [float(self.model.means[s]) for s in self._state_order],
            "variances": [
                float(self.model.variances[s]) for s in self._state_order
            ],
        }

    def decode(self) -> np.ndarray:
        """Viterbi decode and remap to sorted state order."""
        if not self._fitted:
            raise RuntimeError("Model not fitted yet")
        raw_states = self.model.decode(self._returns)
        remap = {int(old): new for new, old in enumerate(self._state_order)}
        return np.array([remap[s] for s in raw_states])

    def predict_proba(self) -> np.ndarray:
        """Posterior probabilities, columns reordered by state_order."""
        if not self._fitted:
            raise RuntimeError("Model not fitted yet")
        gamma = self.model.predict_proba(self._returns)
        return gamma[:, self._state_order]

    def current_regime(self) -> dict:
        """Current regime (last time step) with confidence."""
        if not self._fitted:
            raise RuntimeError("Model not fitted yet")
        gamma = self.predict_proba()
        last_probs = gamma[-1]
        regime_idx = int(np.argmax(last_probs))
        return {
            "regime": REGIME_NAMES[regime_idx],
            "regime_index": regime_idx,
            "confidence": float(last_probs[regime_idx]),
            "probabilities": {
                name: float(p) for name, p in zip(REGIME_NAMES, last_probs)
            },
        }

    def kelly_positions(self) -> list[dict]:
        """Kelly criterion optimal fraction per regime: f* = mu / sigma^2."""
        if not self._fitted:
            raise RuntimeError("Model not fitted yet")
        positions = []
        for idx, s in enumerate(self._state_order):
            kelly = float(self.model.means[s] / self.model.variances[s])
            positions.append(
                {
                    "regime": REGIME_NAMES[idx],
                    "kelly_fraction": kelly,
                    "direction": "LONG" if kelly > 0 else "SHORT/FLAT",
                }
            )
        return positions

    def run_pipeline(self, returns: np.ndarray) -> dict:
        """Full pipeline: fit → decode → regime → Kelly → summary."""
        returns = np.asarray(returns, dtype=np.float64)
        self.fit(returns)

        decoded_states = self.decode()
        regime = self.current_regime()
        kelly = self.kelly_positions()

        # Regime statistics
        regime_stats = []
        for idx, name in enumerate(REGIME_NAMES):
            mask = decoded_states == idx
            n_days = int(mask.sum())
            if n_days > 0:
                mean_ret = float(returns[mask].mean() * 100 * 252)
                vol = float(returns[mask].std() * 100 * np.sqrt(252))
                sharpe = mean_ret / vol if vol > 0 else 0.0
            else:
                mean_ret = vol = sharpe = 0.0
            regime_stats.append(
                {
                    "regime": name,
                    "n_days": n_days,
                    "pct": round(n_days / len(returns) * 100, 1),
                    "annualized_return_pct": round(mean_ret, 1),
                    "annualized_vol_pct": round(vol, 1),
                    "sharpe": round(sharpe, 2),
                }
            )

        return {
            "status": self.status(),
            "current_regime": regime,
            "kelly_positions": kelly,
            "regime_stats": regime_stats,
        }
