FROM python:3.12-slim AS base

WORKDIR /app

# Install core library first
COPY packages/core/pyproject.toml /core/pyproject.toml
COPY packages/core/src/ /core/src/
RUN pip install --no-cache-dir /core

# Install trading-system
COPY services/trading-system/pyproject.toml .
COPY services/trading-system/src/ src/

RUN pip install --no-cache-dir -e ".[dev]"

COPY services/trading-system/tests/ tests/

# --- Run target: serve the API ---
FROM base AS run
EXPOSE 8001
CMD ["uvicorn", "trading_system.app:app", "--host", "0.0.0.0", "--port", "8001"]

# --- Test target: run the test suite ---
FROM base AS test
CMD ["pytest", "--tb=short", "-v"]
