"""
HTTP client for the data-generator service.
"""

import os

import httpx

DATA_GENERATOR_URL = os.environ.get(
    "DATA_GENERATOR_URL", "http://data-generator:8000"
)


class DataGeneratorClient:
    """Synchronous client for the data-generator API."""

    def __init__(self, base_url: str | None = None, timeout: float = 30.0):
        self.base_url = base_url or DATA_GENERATOR_URL
        self.timeout = timeout

    def health(self) -> dict:
        with httpx.Client(base_url=self.base_url, timeout=self.timeout) as client:
            resp = client.get("/health")
            resp.raise_for_status()
            return resp.json()

    def get_batch(self, n_samples: int = 2000, seed: int | None = 42) -> dict:
        params = {"n_samples": n_samples}
        if seed is not None:
            params["seed"] = seed
        with httpx.Client(base_url=self.base_url, timeout=self.timeout) as client:
            resp = client.get("/data/batch", params=params)
            resp.raise_for_status()
            return resp.json()

    def create_session(
        self, n_samples: int = 2000, seed: int | None = 42
    ) -> dict:
        with httpx.Client(base_url=self.base_url, timeout=self.timeout) as client:
            resp = client.post(
                "/data/session",
                json={"n_samples": n_samples, "seed": seed},
            )
            resp.raise_for_status()
            return resp.json()

    def stream(self, session_id: str, count: int = 1) -> dict:
        with httpx.Client(base_url=self.base_url, timeout=self.timeout) as client:
            resp = client.get(
                "/data/stream",
                params={"session_id": session_id, "count": count},
            )
            resp.raise_for_status()
            return resp.json()

    def delete_session(self, session_id: str) -> dict:
        with httpx.Client(base_url=self.base_url, timeout=self.timeout) as client:
            resp = client.delete(f"/data/session/{session_id}")
            resp.raise_for_status()
            return resp.json()
