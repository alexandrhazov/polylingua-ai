"""Postgres-backed aiogram FSM storage.

Keeps conversation state (current step + in-flight round data: sentences,
word_ids, in-progress setup fields) in the same database as everything else,
so a Render restart/redeploy never strands a user mid-conversation the way
the default in-memory storage does.
"""
from __future__ import annotations

from typing import Any

from aiogram.fsm.storage.base import BaseStorage, StorageKey

from app.db import FSMState, async_session


def _row_key(key: StorageKey) -> str:
    return f"{key.bot_id}:{key.chat_id}:{key.user_id}:{key.destiny}"


class PostgresStorage(BaseStorage):
    async def set_state(self, key: StorageKey, state: Any = None) -> None:
        state_str = state.state if hasattr(state, "state") else state
        row_key = _row_key(key)
        async with async_session() as session:
            row = await session.get(FSMState, row_key)
            if row is None:
                session.add(FSMState(key=row_key, state=state_str, data={}))
            else:
                row.state = state_str
            await session.commit()

    async def get_state(self, key: StorageKey) -> str | None:
        async with async_session() as session:
            row = await session.get(FSMState, _row_key(key))
            return row.state if row else None

    async def set_data(self, key: StorageKey, data: dict[str, Any]) -> None:
        row_key = _row_key(key)
        async with async_session() as session:
            row = await session.get(FSMState, row_key)
            if row is None:
                session.add(FSMState(key=row_key, state=None, data=dict(data)))
            else:
                row.data = dict(data)
            await session.commit()

    async def get_data(self, key: StorageKey) -> dict[str, Any]:
        async with async_session() as session:
            row = await session.get(FSMState, _row_key(key))
            return dict(row.data) if row and row.data else {}

    async def close(self) -> None:
        pass
