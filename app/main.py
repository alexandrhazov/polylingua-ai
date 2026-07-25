"""FastAPI ASGI app: Telegram webhook receiver + health check.

Deployed to Render's free tier with:
    uvicorn app.main:app --host 0.0.0.0 --port $PORT
"""
from __future__ import annotations

import asyncio
import contextlib
import logging
from contextlib import asynccontextmanager
from typing import Optional

from aiogram.exceptions import TelegramNetworkError, TelegramRetryAfter
from aiogram.types import Update
from fastapi import FastAPI, Header, Request, Response

from app import db
from app.bot_instance import bot, dp
from app.config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# How many times to retry registering the webhook before giving up.
_WEBHOOK_MAX_ATTEMPTS = 6


async def _run_startup() -> None:
    """Initialise the DB and register the Telegram webhook.

    Runs as a background task (see ``lifespan``) so it never blocks the server
    from accepting connections — the health check must stay responsive during
    deploys. Tolerates a rate-limited (429) or transient setWebhook by
    retrying, instead of stalling startup or crashing the app.
    """
    try:
        await db.init_models()
    except Exception:  # noqa: BLE001 — never let a DB hiccup crash startup
        logger.exception("init_models failed during startup")

    for attempt in range(1, _WEBHOOK_MAX_ATTEMPTS + 1):
        try:
            await bot.set_webhook(
                url=settings.webhook_full_url,
                secret_token=settings.webhook_secret or None,
                drop_pending_updates=True,
                allowed_updates=dp.resolve_used_update_types(),
            )
            logger.info("Webhook set to %s", settings.webhook_full_url)
            return
        except TelegramRetryAfter as exc:
            delay = exc.retry_after + 1
            logger.warning(
                "setWebhook rate-limited (429); retrying in %ss (attempt %s/%s)",
                delay, attempt, _WEBHOOK_MAX_ATTEMPTS,
            )
            await asyncio.sleep(delay)
        except TelegramNetworkError as exc:
            delay = 2 * attempt
            logger.warning(
                "setWebhook network error (%s); retrying in %ss (attempt %s/%s)",
                exc, delay, attempt, _WEBHOOK_MAX_ATTEMPTS,
            )
            await asyncio.sleep(delay)
    logger.error("Failed to register webhook after %s attempts", _WEBHOOK_MAX_ATTEMPTS)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Run startup work (DB init + Telegram setWebhook) in the background so the
    # server starts listening immediately and /health is available right away.
    # Blocking here would make the health check time out during deploys, and a
    # slow or rate-limited (429) setWebhook could stall or crash startup.
    startup_task = asyncio.create_task(_run_startup())
    try:
        yield
    finally:
        # Shutdown: cancel startup if still running, then release the HTTP
        # session. Do NOT delete the webhook — Render may start the new
        # instance before stopping the old one, so deleting here would race
        # with (and wipe out) the webhook the incoming instance just set.
        startup_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await startup_task
        await bot.session.close()
        logger.info("Bot session closed")


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
