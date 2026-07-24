"""Singleton Bot and Dispatcher instances with routers registered."""
from __future__ import annotations

from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.fsm.storage.memory import MemoryStorage

from app.config import settings
from app.handlers import practice, start

bot = Bot(
    token=settings.bot_token,
    default=DefaultBotProperties(parse_mode=ParseMode.HTML),
)

# In-memory FSM storage: correct for a single Render free-tier worker.
dp = Dispatcher(storage=MemoryStorage())

dp.include_router(start.router)
dp.include_router(practice.router)
