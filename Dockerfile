FROM python:3.12-slim AS base

WORKDIR /app

COPY pyproject.toml .
COPY src/ src/

RUN pip install --no-cache-dir ".[dev]"

COPY tests/ tests/

# --- Run target: execute the pipeline ---
FROM base AS run
CMD ["renaissance-trading"]

# --- Test target: run the test suite ---
FROM base AS test
CMD ["pytest", "--tb=short", "-v"]
