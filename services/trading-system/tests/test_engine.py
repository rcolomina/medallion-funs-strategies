"""
Tests for the TradingEngine class.
"""

import numpy as np
import pytest

from trading_system.engine import TradingEngine, REGIME_NAMES


@pytest.fixture
def engine():
    return TradingEngine(n_states=2, n_iter=50, tol=1e-6)


@pytest.fixture
def simple_returns():
    """Two-regime synthetic returns."""
    rng = np.random.RandomState(0)
    low = rng.normal(-0.01, 0.002, 200)
    high = rng.normal(0.01, 0.002, 200)
    return np.concatenate([low, high])


@pytest.fixture
def fitted_engine(engine, simple_returns):
    engine.fit(simple_returns)
    return engine


class TestFit:
    def test_not_fitted_initially(self, engine):
        assert engine.is_fitted is False

    def test_fit_marks_fitted(self, fitted_engine):
        assert fitted_engine.is_fitted is True

    def test_fit_returns_status(self, engine, simple_returns):
        result = engine.fit(simple_returns)
        assert result["fitted"] is True
        assert "log_likelihood" in result

    def test_status_before_fit(self, engine):
        status = engine.status()
        assert status["fitted"] is False

    def test_status_after_fit(self, fitted_engine):
        status = fitted_engine.status()
        assert status["fitted"] is True
        assert status["n_states"] == 2
        assert status["n_observations"] == 400
        assert np.isfinite(status["log_likelihood"])


class TestDecode:
    def test_raises_before_fit(self, engine):
        with pytest.raises(RuntimeError, match="not fitted"):
            engine.decode()

    def test_output_shape(self, fitted_engine):
        states = fitted_engine.decode()
        assert states.shape == (400,)

    def test_states_in_range(self, fitted_engine):
        states = fitted_engine.decode()
        assert np.all(states >= 0)
        assert np.all(states < 2)


class TestPredictProba:
    def test_raises_before_fit(self, engine):
        with pytest.raises(RuntimeError, match="not fitted"):
            engine.predict_proba()

    def test_rows_sum_to_one(self, fitted_engine):
        gamma = fitted_engine.predict_proba()
        np.testing.assert_allclose(gamma.sum(axis=1), 1.0, atol=1e-6)


class TestCurrentRegime:
    def test_raises_before_fit(self, engine):
        with pytest.raises(RuntimeError, match="not fitted"):
            engine.current_regime()

    def test_has_expected_fields(self, fitted_engine):
        regime = fitted_engine.current_regime()
        assert "regime" in regime
        assert "confidence" in regime
        assert "probabilities" in regime
        assert regime["confidence"] > 0

    def test_regime_name_valid(self, fitted_engine):
        regime = fitted_engine.current_regime()
        assert regime["regime"] in REGIME_NAMES


class TestKellyPositions:
    def test_raises_before_fit(self, engine):
        with pytest.raises(RuntimeError, match="not fitted"):
            engine.kelly_positions()

    def test_returns_list(self, fitted_engine):
        positions = fitted_engine.kelly_positions()
        assert isinstance(positions, list)
        assert len(positions) == 2

    def test_position_fields(self, fitted_engine):
        positions = fitted_engine.kelly_positions()
        for pos in positions:
            assert "regime" in pos
            assert "kelly_fraction" in pos
            assert "direction" in pos


class TestRunPipeline:
    def test_full_pipeline(self, engine, simple_returns):
        result = engine.run_pipeline(simple_returns)
        assert result["status"]["fitted"] is True
        assert "current_regime" in result
        assert "kelly_positions" in result
        assert "regime_stats" in result
