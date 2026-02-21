"""
Tests for the DataGeneratorClient with mocked HTTP responses.
"""

import pytest
import respx
import httpx

from trading_system.client import DataGeneratorClient


BASE_URL = "http://test-data-generator:8000"


@pytest.fixture
def client():
    return DataGeneratorClient(base_url=BASE_URL)


class TestHealth:
    @respx.mock
    def test_health(self, client):
        respx.get(f"{BASE_URL}/health").mock(
            return_value=httpx.Response(
                200, json={"status": "ok", "service": "data-generator"}
            )
        )
        result = client.health()
        assert result["status"] == "ok"

    @respx.mock
    def test_health_error(self, client):
        respx.get(f"{BASE_URL}/health").mock(
            return_value=httpx.Response(500, json={"detail": "error"})
        )
        with pytest.raises(httpx.HTTPStatusError):
            client.health()


class TestGetBatch:
    @respx.mock
    def test_get_batch(self, client):
        mock_data = {
            "n_samples": 10,
            "data": [
                {
                    "index": i,
                    "date": "2015-01-01",
                    "return": 0.001 * i,
                    "price": 100.0 + i,
                    "true_state": 0,
                }
                for i in range(10)
            ],
        }
        respx.get(f"{BASE_URL}/data/batch").mock(
            return_value=httpx.Response(200, json=mock_data)
        )
        result = client.get_batch(n_samples=10, seed=42)
        assert result["n_samples"] == 10
        assert len(result["data"]) == 10

    @respx.mock
    def test_get_batch_params(self, client):
        respx.get(f"{BASE_URL}/data/batch").mock(
            return_value=httpx.Response(200, json={"n_samples": 5, "data": []})
        )
        client.get_batch(n_samples=5, seed=99)
        request = respx.calls[0].request
        assert "n_samples=5" in str(request.url)
        assert "seed=99" in str(request.url)


class TestSession:
    @respx.mock
    def test_create_session(self, client):
        respx.post(f"{BASE_URL}/data/session").mock(
            return_value=httpx.Response(
                200, json={"session_id": "abc-123", "n_samples": 100}
            )
        )
        result = client.create_session(n_samples=100, seed=42)
        assert result["session_id"] == "abc-123"

    @respx.mock
    def test_stream(self, client):
        respx.get(f"{BASE_URL}/data/stream").mock(
            return_value=httpx.Response(
                200,
                json={
                    "session_id": "abc-123",
                    "cursor": 5,
                    "remaining": 95,
                    "data": [],
                },
            )
        )
        result = client.stream(session_id="abc-123", count=5)
        assert result["cursor"] == 5

    @respx.mock
    def test_delete_session(self, client):
        respx.delete(f"{BASE_URL}/data/session/abc-123").mock(
            return_value=httpx.Response(
                200, json={"status": "deleted", "session_id": "abc-123"}
            )
        )
        result = client.delete_session("abc-123")
        assert result["status"] == "deleted"
