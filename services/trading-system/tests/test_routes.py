"""
Tests for trading system API endpoints.
"""

import numpy as np
import pytest
import respx
import httpx
from fastapi.testclient import TestClient

from trading_system.app import create_app
from trading_system.client import DataGeneratorClient
from trading_system.engine import TradingEngine
from trading_system import routes


DATA_GENERATOR_URL = "http://mock-data-generator:8000"


def _make_batch_response(n_samples=100, seed=42):
    """Build a mock batch response matching data-generator format."""
    rng = np.random.RandomState(seed)
    returns = rng.normal(0.0005, 0.01, n_samples)
    prices = 100 * np.exp(np.cumsum(returns))
    return {
        "n_samples": n_samples,
        "data": [
            {
                "index": i,
                "date": "2015-01-01",
                "return": float(returns[i]),
                "price": float(prices[i]),
                "true_state": 0,
            }
            for i in range(n_samples)
        ],
    }


@pytest.fixture
def client():
    app = create_app()
    # Reset engine and client for each test
    routes.set_engine(TradingEngine())
    routes.set_client(DataGeneratorClient(base_url=DATA_GENERATOR_URL))
    with TestClient(app) as c:
        yield c


class TestHealth:
    def test_health_check(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["service"] == "trading-system"


class TestStatus:
    def test_status_not_fitted(self, client):
        resp = client.get("/status")
        assert resp.status_code == 200
        assert resp.json()["fitted"] is False


class TestRegime:
    def test_regime_not_fitted(self, client):
        resp = client.get("/regime")
        assert resp.status_code == 409

    @respx.mock
    def test_regime_after_run(self, client):
        batch = _make_batch_response(n_samples=500)
        respx.get(f"{DATA_GENERATOR_URL}/data/batch").mock(
            return_value=httpx.Response(200, json=batch)
        )
        client.post("/run?n_samples=500&seed=42")
        resp = client.get("/regime")
        assert resp.status_code == 200
        data = resp.json()
        assert "regime" in data
        assert "confidence" in data


class TestPortfolio:
    def test_portfolio_not_fitted(self, client):
        resp = client.get("/portfolio")
        assert resp.status_code == 409

    @respx.mock
    def test_portfolio_after_run(self, client):
        batch = _make_batch_response(n_samples=500)
        respx.get(f"{DATA_GENERATOR_URL}/data/batch").mock(
            return_value=httpx.Response(200, json=batch)
        )
        client.post("/run?n_samples=500&seed=42")
        resp = client.get("/portfolio")
        assert resp.status_code == 200
        data = resp.json()
        assert "positions" in data


class TestRun:
    @respx.mock
    def test_run_pipeline(self, client):
        batch = _make_batch_response(n_samples=500)
        respx.get(f"{DATA_GENERATOR_URL}/data/batch").mock(
            return_value=httpx.Response(200, json=batch)
        )
        resp = client.post("/run?n_samples=500&seed=42")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"]["fitted"] is True
        assert "current_regime" in data
        assert "kelly_positions" in data
        assert "regime_stats" in data

    @respx.mock
    def test_run_data_generator_down(self, client):
        respx.get(f"{DATA_GENERATOR_URL}/data/batch").mock(
            side_effect=httpx.ConnectError("Connection refused")
        )
        resp = client.post("/run?n_samples=500&seed=42")
        assert resp.status_code == 502
