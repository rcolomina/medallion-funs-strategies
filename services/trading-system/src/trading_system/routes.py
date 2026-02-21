"""
API routes for the trading system service.
"""

from fastapi import APIRouter, HTTPException, Query

from trading_system.client import DataGeneratorClient
from trading_system.engine import TradingEngine

router = APIRouter()

# Module-level engine instance (re-created on each /run)
_engine = TradingEngine()
_client = DataGeneratorClient()


def get_engine() -> TradingEngine:
    return _engine


def get_client() -> DataGeneratorClient:
    return _client


def set_engine(engine: TradingEngine) -> None:
    global _engine
    _engine = engine


def set_client(client: DataGeneratorClient) -> None:
    global _client
    _client = client


@router.get("/health")
def health():
    return {"status": "ok", "service": "trading-system"}


@router.post("/run")
def run_pipeline(
    n_samples: int = Query(default=2000, ge=1, le=100_000),
    seed: int | None = Query(default=42),
):
    """Fetch batch from data-generator, fit HMM, decode, return results."""
    try:
        batch = _client.get_batch(n_samples=n_samples, seed=seed)
    except Exception as e:
        raise HTTPException(
            status_code=502,
            detail=f"Failed to fetch data from data-generator: {e}",
        )

    returns = [point["return"] for point in batch["data"]]
    result = _engine.run_pipeline(returns)
    return result


@router.get("/status")
def model_status():
    """Model state (fitted, n_obs, log-likelihood)."""
    return _engine.status()


@router.get("/regime")
def current_regime():
    """Current regime + confidence + probabilities."""
    if not _engine.is_fitted:
        raise HTTPException(status_code=409, detail="Model not fitted yet")
    return _engine.current_regime()


@router.get("/portfolio")
def portfolio():
    """Kelly positions per regime."""
    if not _engine.is_fitted:
        raise HTTPException(status_code=409, detail="Model not fitted yet")
    return {"positions": _engine.kelly_positions()}
