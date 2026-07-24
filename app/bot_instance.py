"""Singleton Bot and Dispatcher instances with routers registered."""
from __future__ import annotations

from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode

from app.config import settings
from app.fsm_storage import PostgresStorage
from app.handlers import practice, start

bot = Bot(
    token=settings.bot_token,
    default=DefaultBotProperties(parse_mode=ParseMode.HTML),
)

# Postgres-backed FSM storage: survives restarts/redeploys, unlike in-memory
# storage, which would otherwise strand users mid-conversation on every deploy.
dp = Dispatcher(storage=PostgresStorage())

dp.include_router(start.router)
dp.include_router(practice.router)
