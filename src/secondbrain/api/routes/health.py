import httpx
import redis
from fastapi import APIRouter
from sqlalchemy import text

from secondbrain.config import settings
from secondbrain.db.session import engine

router = APIRouter(tags=["health"])


@router.get("/health")
def health() -> dict:
    """Liveness only: no dependencies touched."""
    return {"status": "ok"}


@router.get("/health/deps")
def health_deps() -> dict:
    """Readiness: checks Postgres, Redis and the Ollama endpoint."""
    checks: dict[str, str] = {}
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        checks["postgres"] = "ok"
    except Exception as exc:
        checks["postgres"] = f"error: {exc.__class__.__name__}"
    try:
        redis.Redis.from_url(settings.redis_url, socket_connect_timeout=2).ping()
        checks["redis"] = "ok"
    except Exception as exc:
        checks["redis"] = f"error: {exc.__class__.__name__}"
    try:
        httpx.get(f"{settings.ollama_base_url}/api/tags", timeout=2).raise_for_status()
        checks["ollama"] = "ok"
    except Exception as exc:
        checks["ollama"] = f"error: {exc.__class__.__name__}"
    return {"status": "ok" if all(v == "ok" for v in checks.values()) else "degraded", **checks}
