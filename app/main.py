"""FastAPI ASGI app: Telegram webhook receiver + health check.

Deployed to Render's free tier with:
    uvicorn app.main:app --host 0.0.0.0 --port $PORT
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Optional

from aiogram.types import Update
from fastapi import FastAPI, Header, Request, Response

from app import db
from app.bot_instance import bot, dp
from app.config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Startup: ensure tables exist, then register the webhook with Telegram.
    await db.init_models()
    await bot.set_webhook(
        url=settings.webhook_full_url,
        secret_token=settings.webhook_secret or None,
        drop_pending_updates=True,
        allowed_updates=dp.resolve_used_update_types(),
    )
    logger.info("Webhook set to %s", settings.webhook_full_url)
    try:
        yield
    finally:
        # Shutdown: unhook and release the HTTP session cleanly.
        await bot.delete_webhook()
        await bot.session.close()
        logger.info("Webhook deleted and bot session closed")


app = FastAPI(title=settings.app_name, lifespan=lifespan)


@app.get("/health")
async def health() -> dict[str, str]:
    """Liveness probe used by UptimeRobot / cron-job.org to keep the dyno awake."""
    return {"status": "ok"}


@app.post(settings.webhook_path)
async def telegram_webhook(
    request: Request,
    # FastAPI evaluates this annotation at runtime, so use typing.Optional
    # rather than PEP 604 `str | None` for portability to Python 3.9+.
    x_telegram_bot_api_secret_token: Optional[str] = Header(default=None),
) -> Response:
    # Reject spoofed requests when a secret is configured.
    if settings.webhook_secret and x_telegram_bot_api_secret_token != settings.webhook_secret:
        logger.warning("Rejected webhook call with bad secret token")
        return Response(status_code=403)

    data = await request.json()
    update = Update.model_validate(data, context={"bot": bot})
    await dp.feed_webhook_update(bot, update)
    return Response(status_code=200)
