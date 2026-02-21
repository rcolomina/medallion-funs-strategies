"""
API routes for the data generator service.
"""

import uuid

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from data_generator.data import generate_regime_data

router = APIRouter()

# In-memory session store: session_id -> {data, cursor}
_sessions: dict[str, dict] = {}


class SessionRequest(BaseModel):
    n_samples: int = Field(default=2000, ge=1, le=100_000)
    seed: int | None = Field(default=42)


class SessionResponse(BaseModel):
    session_id: str
    n_samples: int


class DataPoint(BaseModel):
    index: int
    date: str
    return_: float = Field(alias="return")
    price: float
    true_state: int

    model_config = {"populate_by_name": True}


class BatchResponse(BaseModel):
    n_samples: int
    data: list[DataPoint]


class StreamResponse(BaseModel):
    session_id: str
    cursor: int
    remaining: int
    data: list[DataPoint]


@router.get("/health")
def health():
    return {"status": "ok", "service": "data-generator"}


@router.get("/data/batch", response_model=BatchResponse)
def get_batch(
    n_samples: int = Query(default=2000, ge=1, le=100_000),
    seed: int | None = Query(default=42),
):
    """Return a full dataset for initial HMM training."""
    returns, prices, states, dates = generate_regime_data(
        n_samples=n_samples, seed=seed
    )
    data = [
        DataPoint(
            index=i,
            date=str(dates[i].date()),
            **{"return": float(returns[i])},
            price=float(prices[i]),
            true_state=int(states[i]),
        )
        for i in range(n_samples)
    ]
    return BatchResponse(n_samples=n_samples, data=data)


@router.post("/data/session", response_model=SessionResponse)
def create_session(req: SessionRequest):
    """Create a streaming session with pre-generated data."""
    session_id = str(uuid.uuid4())
    returns, prices, states, dates = generate_regime_data(
        n_samples=req.n_samples, seed=req.seed
    )
    _sessions[session_id] = {
        "returns": returns,
        "prices": prices,
        "states": states,
        "dates": dates,
        "cursor": 0,
        "n_samples": req.n_samples,
    }
    return SessionResponse(session_id=session_id, n_samples=req.n_samples)


@router.get("/data/stream", response_model=StreamResponse)
def stream_data(
    session_id: str = Query(...),
    count: int = Query(default=1, ge=1, le=1000),
):
    """Poll next N data points from a session."""
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session = _sessions[session_id]
    cursor = session["cursor"]
    n_samples = session["n_samples"]

    if cursor >= n_samples:
        raise HTTPException(status_code=410, detail="Session exhausted")

    end = min(cursor + count, n_samples)
    data = [
        DataPoint(
            index=i,
            date=str(session["dates"][i].date()),
            **{"return": float(session["returns"][i])},
            price=float(session["prices"][i]),
            true_state=int(session["states"][i]),
        )
        for i in range(cursor, end)
    ]

    session["cursor"] = end
    remaining = n_samples - end

    return StreamResponse(
        session_id=session_id,
        cursor=end,
        remaining=remaining,
        data=data,
    )


@router.delete("/data/session/{session_id}")
def delete_session(session_id: str):
    """Cleanup a session."""
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    del _sessions[session_id]
    return {"status": "deleted", "session_id": session_id}


def get_sessions_store():
    """Expose session store for testing."""
    return _sessions
