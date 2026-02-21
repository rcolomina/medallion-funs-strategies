"""
FastAPI application factory for the data generator service.
"""

from fastapi import FastAPI

from data_generator.routes import router


def create_app() -> FastAPI:
    app = FastAPI(
        title="Data Generator",
        description="Synthetic market data generation service",
        version="0.1.0",
    )
    app.include_router(router)
    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
