from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes import health
from app.core.config import get_settings
from app.core.logging import configure_logging, get_logger

log = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    log.info(
        "app.startup",
        app=settings.app_name,
        env=settings.env,
        host=settings.host,
        port=settings.port,
    )
    # Phase 2: start long-lived resources (audio thread, etc.) here...
    yield
    # ...and stop them here, after the server stops taking requests.
    log.info("app.shutdown", app=settings.app_name)


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging(settings.log_level)

    app = FastAPI(title=settings.app_name, lifespan=lifespan)
    app.include_router(health.router, prefix="/api")
    return app


app = create_app()
