FROM python:3.12-slim AS base

WORKDIR /app

COPY pyproject.toml .
COPY src/ src/

RUN pip install --no-cache-dir -e ".[dev]"

COPY tests/ tests/

# --- Run target: serve the API ---
FROM base AS run
EXPOSE 8000
CMD ["uvicorn", "data_generator.app:app", "--host", "0.0.0.0", "--port", "8000"]

# --- Test target: run the test suite ---
FROM base AS test
CMD ["pytest", "--tb=short", "-v"]
