"""
Tests for data generator API endpoints.
"""

import pytest
from fastapi.testclient import TestClient

from data_generator.app import create_app
from data_generator.routes import get_sessions_store


@pytest.fixture
def client():
    app = create_app()
    with TestClient(app) as c:
        yield c


@pytest.fixture(autouse=True)
def clear_sessions():
    """Clear session store between tests."""
    store = get_sessions_store()
    store.clear()
    yield
    store.clear()


class TestHealth:
    def test_health_check(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["service"] == "data-generator"


class TestBatch:
    def test_default_batch(self, client):
        resp = client.get("/data/batch")
        assert resp.status_code == 200
        data = resp.json()
        assert data["n_samples"] == 2000
        assert len(data["data"]) == 2000

    def test_custom_batch(self, client):
        resp = client.get("/data/batch?n_samples=50&seed=99")
        assert resp.status_code == 200
        data = resp.json()
        assert data["n_samples"] == 50
        assert len(data["data"]) == 50

    def test_batch_data_fields(self, client):
        resp = client.get("/data/batch?n_samples=5")
        data = resp.json()
        point = data["data"][0]
        assert "index" in point
        assert "date" in point
        assert "return" in point
        assert "price" in point
        assert "true_state" in point

    def test_batch_reproducibility(self, client):
        r1 = client.get("/data/batch?n_samples=10&seed=42").json()
        r2 = client.get("/data/batch?n_samples=10&seed=42").json()
        assert r1["data"] == r2["data"]

    def test_batch_invalid_n_samples(self, client):
        resp = client.get("/data/batch?n_samples=0")
        assert resp.status_code == 422


class TestSession:
    def test_create_session(self, client):
        resp = client.post("/data/session", json={"n_samples": 100, "seed": 42})
        assert resp.status_code == 200
        data = resp.json()
        assert "session_id" in data
        assert data["n_samples"] == 100

    def test_stream_data(self, client):
        # Create session
        create_resp = client.post(
            "/data/session", json={"n_samples": 10, "seed": 42}
        )
        session_id = create_resp.json()["session_id"]

        # Stream 3 points
        resp = client.get(f"/data/stream?session_id={session_id}&count=3")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["data"]) == 3
        assert data["cursor"] == 3
        assert data["remaining"] == 7

    def test_stream_advances_cursor(self, client):
        create_resp = client.post(
            "/data/session", json={"n_samples": 10, "seed": 42}
        )
        session_id = create_resp.json()["session_id"]

        # First stream
        r1 = client.get(f"/data/stream?session_id={session_id}&count=4").json()
        assert r1["cursor"] == 4

        # Second stream
        r2 = client.get(f"/data/stream?session_id={session_id}&count=4").json()
        assert r2["cursor"] == 8
        assert r2["data"][0]["index"] == 4

    def test_stream_exhausted(self, client):
        create_resp = client.post(
            "/data/session", json={"n_samples": 5, "seed": 42}
        )
        session_id = create_resp.json()["session_id"]

        # Consume all
        client.get(f"/data/stream?session_id={session_id}&count=5")

        # Should be exhausted
        resp = client.get(f"/data/stream?session_id={session_id}&count=1")
        assert resp.status_code == 410

    def test_stream_unknown_session(self, client):
        resp = client.get("/data/stream?session_id=nonexistent&count=1")
        assert resp.status_code == 404

    def test_delete_session(self, client):
        create_resp = client.post(
            "/data/session", json={"n_samples": 10, "seed": 42}
        )
        session_id = create_resp.json()["session_id"]

        resp = client.delete(f"/data/session/{session_id}")
        assert resp.status_code == 200
        assert resp.json()["status"] == "deleted"

        # Session should be gone
        resp = client.get(f"/data/stream?session_id={session_id}&count=1")
        assert resp.status_code == 404

    def test_delete_unknown_session(self, client):
        resp = client.delete("/data/session/nonexistent")
        assert resp.status_code == 404
