FROM python:3.12-slim AS base

WORKDIR /app

COPY pyproject.toml .
COPY src/ src/

RUN pip install --no-cache-dir -e ".[dev]"

COPY tests/ tests/

# --- Test target: run the test suite ---
FROM base AS test
CMD ["pytest", "--tb=short", "-v"]
