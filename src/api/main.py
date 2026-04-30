"""FastAPI application entrypoint."""

from fastapi import FastAPI

from src.api.routes import forecasts, pricing, products


app = FastAPI(title="Pricing Optimization System", version="0.1.0")
app.include_router(products.router)
app.include_router(pricing.router)
app.include_router(forecasts.router)


@app.get("/health")
def health_check() -> dict[str, str]:
    """Return basic service health."""
    return {"status": "ok"}
