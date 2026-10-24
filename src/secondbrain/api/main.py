from fastapi import Depends, FastAPI

from secondbrain.api.deps import require_api_key
from secondbrain.api.routes import ask, documents, health, search
from secondbrain.config import settings
from secondbrain.logging import configure_logging


def create_app() -> FastAPI:
    configure_logging(settings.log_level)
    app = FastAPI(title="Second Brain RAG", version="0.1.0")
    app.include_router(health.router)
    protected = [Depends(require_api_key)]
    for router in (documents.router, search.router, ask.router):
        app.include_router(router, dependencies=protected)
    return app


app = create_app()
