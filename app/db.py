"""Persistent storage: async SQLAlchemy engine, models, and query helpers.

Backed by Postgres (Neon free tier by default) so vocabulary and practice
progress survive restarts/redeploys, unlike the in-memory FSM storage used
for transient conversation state.
"""
from __future__ import annotations

import datetime as dt
from typing import Sequence

from sqlalchemy import BigInteger, Boolean, DateTime, ForeignKey, String, func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from app.config import settings

# statement_cache_size=0 disables asyncpg's prepared-statement cache, which is
# required when connecting through Neon's PgBouncer pooler (transaction-mode
# pooling doesn't support reusing prepared statements across connections).
engine = create_async_engine(
    settings.database_url,
    connect_args={"statement_cache_size": 0},
)
async_session = async_sessionmaker(engine, expire_on_commit=False)

# Score at/above this (out of 10) retires a word from the practice rotation.
MASTERY_THRESHOLD = 7.0


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    telegram_id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    level: Mapped[str] = mapped_column(String(2))
    native_language: Mapped[str] = mapped_column(String(64))
    target_language: Mapped[str] = mapped_column(String(64))
    # "native_to_target": sentences in native_language, translate into target_language.
    # "target_to_native": sentences in target_language, translate into native_language.
    direction: Mapped[str] = mapped_column(String(16))

    words: Mapped[list["Word"]] = relationship(back_populates="user", cascade="all, delete-orphan")


class Word(Base):
    __tablename__ = "words"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.telegram_id", ondelete="CASCADE"))
    text: Mapped[str] = mapped_column(String(200))
    mastered: Mapped[bool] = mapped_column(Boolean, default=False)
    practiced_count: Mapped[int] = mapped_column(default=0)
    last_practiced_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    user: Mapped[User] = relationship(back_populates="words")


async def init_models() -> None:
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def get_user(telegram_id: int) -> User | None:
    async with async_session() as session:
        return await session.get(User, telegram_id)


async def upsert_profile(
    telegram_id: int, level: str, native_language: str, target_language: str, direction: str
) -> None:
    async with async_session() as session:
        user = await session.get(User, telegram_id)
        if user is None:
            user = User(telegram_id=telegram_id)
            session.add(user)
        user.level = level
        user.native_language = native_language
        user.target_language = target_language
        user.direction = direction
        await session.commit()


async def add_words(telegram_id: int, words: Sequence[str]) -> int:
    """Add new vocabulary words for a user, skipping ones already stored.

    Returns the number of genuinely new words added.
    """
    async with async_session() as session:
        existing = set(
            (
                await session.execute(
                    select(Word.text).where(Word.user_id == telegram_id)
                )
            ).scalars()
        )
        new_words = []
        seen: set[str] = set()
        for w in words:
            key = w.casefold()
            if key in existing or key in seen:
                continue
            seen.add(key)
            new_words.append(Word(user_id=telegram_id, text=w))
        session.add_all(new_words)
        await session.commit()
        return len(new_words)


async def next_round(telegram_id: int, size: int) -> list[Word]:
    """Fetch up to ``size`` unmastered words, least-recently-practiced first."""
    async with async_session() as session:
        result = await session.execute(
            select(Word)
            .where(Word.user_id == telegram_id, Word.mastered.is_(False))
            .order_by(Word.last_practiced_at.asc().nulls_first())
            .limit(size)
        )
        return list(result.scalars())


async def record_round_result(word_ids: Sequence[int], score: float) -> None:
    async with async_session() as session:
        result = await session.execute(select(Word).where(Word.id.in_(word_ids)))
        words = result.scalars().all()
        now = dt.datetime.now(dt.timezone.utc)
        for word in words:
            word.practiced_count += 1
            word.last_practiced_at = now
            if score >= MASTERY_THRESHOLD:
                word.mastered = True
        await session.commit()


async def skip_words(word_ids: Sequence[int]) -> None:
    """Bump ``last_practiced_at`` for skipped words without recording a grade.

    Sends them to the back of the rotation queue so the next round picks
    different words, while leaving mastery/practiced_count untouched.
    """
    async with async_session() as session:
        result = await session.execute(select(Word).where(Word.id.in_(word_ids)))
        words = result.scalars().all()
        now = dt.datetime.now(dt.timezone.utc)
        for word in words:
            word.last_practiced_at = now
        await session.commit()


async def remaining_count(telegram_id: int) -> int:
    async with async_session() as session:
        result = await session.execute(
            select(func.count())
            .select_from(Word)
            .where(Word.user_id == telegram_id, Word.mastered.is_(False))
        )
        return result.scalar_one()
