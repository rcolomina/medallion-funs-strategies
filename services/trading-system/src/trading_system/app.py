"""
FastAPI application factory for the trading system service.
"""

from fastapi import FastAPI

from trading_system.routes import router


def create_app() -> FastAPI:
    app = FastAPI(
        title="Trading System",
        description="HMM-based market regime detection and trading engine",
        version="0.1.0",
    )
    app.include_router(router)
    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)
