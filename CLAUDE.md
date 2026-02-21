# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Multi-service trading simulation implementing Hidden Markov Models (HMM) for market regime detection, inspired by Renaissance Technologies' Medallion Fund. The system is split into three packages communicating via REST APIs:

1. **`packages/core`** — Shared `GaussianHMM` library (numpy only)
2. **`services/data-generator`** — FastAPI service generating synthetic market data (port 8000)
3. **`services/trading-system`** — FastAPI service running HMM regime detection + Kelly sizing (port 8001)

## Commands

```bash
# Install individual packages (editable, with dev tools)
pip install -e "packages/core[dev]"
pip install -e "services/data-generator[dev]"
pip install -e "services/trading-system[dev]"        # requires core installed first

# Run tests per package
cd packages/core && pytest
cd services/data-generator && pytest
cd services/trading-system && pytest

# Lint
ruff check packages/ services/
ruff format packages/ services/

# Docker — run both services
docker compose up --build

# Docker — run test profiles
docker compose --profile test run test-core
docker compose --profile test run test-data-generator
docker compose --profile test run test-trading-system
```

## Architecture

### `packages/core/src/renaissance_core/`

- **`hmm.py`** — `GaussianHMM` class. All HMM math lives here: forward/backward with scaling for numerical stability, Baum-Welch EM for parameter learning, Viterbi in log-space for decoding. Public API: `fit(obs)`, `decode(obs)`, `predict_proba(obs)`.

**Numerical stability patterns used throughout `hmm.py`:**
- Scaled forward/backward variables (prevents underflow on long sequences)
- Log-space Viterbi (avoids multiplying many small probabilities)
- Floor values at `1e-300` before `log()`, variance floor at `1e-6`

### `services/data-generator/src/data_generator/`

- **`data.py`** — `generate_regime_data()` produces synthetic 3-regime market data (Bull/Sideways/Bear) with known transition matrix.
- **`app.py`** — FastAPI application factory.
- **`routes.py`** — Endpoints: `/health`, `/data/batch`, `/data/session`, `/data/stream`, `DELETE /data/session/{id}`. Sessions store pre-generated data with a cursor for streaming.

### `services/trading-system/src/trading_system/`

- **`engine.py`** — `TradingEngine` class: fit HMM → Viterbi decode → posterior probabilities → Kelly criterion sizing → regime stats.
- **`client.py`** — `DataGeneratorClient` using httpx to call data-generator over HTTP.
- **`app.py`** — FastAPI application factory.
- **`routes.py`** — Endpoints: `/health`, `/run`, `/status`, `/regime`, `/portfolio`.

### Communication

- Trading system calls data-generator via httpx over Docker internal DNS (`http://data-generator:8000`)
- `DATA_GENERATOR_URL` env var for configurability
- `depends_on` + healthcheck ensures startup ordering

## Test Conventions

- Tests use pytest fixtures for shared setup
- Float comparisons: `pytest.approx()` for scalars, `np.testing.assert_allclose()` for arrays
- Core HMM tests grouped by algorithm component: `TestForward`, `TestBackward`, `TestEStep`, `TestFit`, `TestDecode`, `TestPredictProba`, `TestEdgeCases`
- Route tests use FastAPI `TestClient`
- Client tests use `respx` for HTTP mocking

## Style

- Ruff: line-length 88, Python 3.10 target, rules E/F/I/W
- Core deps: numpy only
- Data-generator deps: numpy, pandas, fastapi, uvicorn
- Trading-system deps: renaissance-core, numpy, scipy, fastapi, uvicorn, httpx
