import secrets

from fastapi import Header, HTTPException, status

from secondbrain.config import settings


def require_api_key(x_api_key: str = Header(default="")) -> None:
    """Single-user auth. Disabled when API_KEY is empty (local dev)."""
    if settings.api_key and not secrets.compare_digest(x_api_key, settings.api_key):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid or missing X-API-Key")
