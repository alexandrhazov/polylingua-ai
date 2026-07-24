"""Vocabulary collection (text or file), sentence generation, and translation
evaluation — auto-cycling through a persisted word list a few words at a time.
"""
from __future__ import annotations

import html
import re

from aiogram import F, Router
from aiogram.fsm.context import FSMContext
from aiogram.types import Message
from aiogram.utils.chat_action import ChatActionSender

from app import db
from app.config import settings
from app.services import ai_tutor
from app.services.ai_tutor import TutorError
from app.states.learning import Learning

router = Router(name="practice")

# Split on commas, semicolons, newlines, or runs of whitespace.
_SPLIT_RE = re.compile(r"[,\n;]+|\s{2,}")

# Only plain text vocab files are accepted (file upload path).
_ALLOWED_VOCAB_MIME_PREFIXES = ("text/",)


def _parse_words(text: str) -> list[str]:
    raw = _SPLIT_RE.split(text) if ("," in text or "\n" in text or ";" in text) else text.split()
    return [w.strip() for w in raw if w.strip()]


def _languages_for(user: db.User) -> tuple[str, str]:
    """Return (source_language, dest_language) for the user's chosen direction."""
    if user.direction == "native_to_target":
        return user.native_language, user.target_language
    return user.target_language, user.native_language


async def _start_round(message: Message, state: FSMContext, telegram_id: int) -> None:
    """Fetch the next batch of unmastered words and generate practice sentences.

    If there are no unmastered words (none stored yet, or all mastered), asks
    the learner to send more vocabulary instead.
    """
    user = await db.get_user(telegram_id)
    assert user is not None  # profile is created before awaiting_vocab is reached

    words = await db.next_round(telegram_id, settings.round_size)
    if not words:
        await state.set_state(Learning.awaiting_vocab)
        await message.answer(
            "🎉 You've mastered every word in your list! Send more vocabulary "
            "(text or a .txt file) to keep practicing."
        )
        return

    source_language, _ = _languages_for(user)

    async with ChatActionSender.typing(bot=message.bot, chat_id=message.chat.id):
        try:
            sentences = await ai_tutor.generate_sentences(
                [w.text for w in words], user.level, source_language
            )
        except TutorError as exc:
            await message.answer(str(exc))
            return

    await state.update_data(sentences=sentences, word_ids=[w.id for w in words])
    await state.set_state(Learning.awaiting_translation)

    numbered = "\n".join(
        f"{i}. {html.escape(s)}" for i, s in enumerate(sentences, start=1)
    )
    remaining = await db.remaining_count(telegram_id)
    await message.answer(
        f"Here are your <b>{source_language}</b> practice sentences ({user.level}):\n\n"
        f"{numbered}\n\n"
        f"✍️ Send your translation. ({remaining} word(s) left to master.)"
    )


@router.message(Learning.awaiting_vocab, F.text)
async def collect_vocab_text(message: Message, state: FSMContext) -> None:
    words = _parse_words(message.text or "")
    if not words:
        await message.answer(
            "Please send at least one vocabulary word, e.g. "
            "<i>negotiate, resilient, breakthrough</i>, or upload a .txt file."
        )
        return

    added = await db.add_words(message.from_user.id, words)
    skipped = len(words) - added
    note = f" ({skipped} duplicate(s) skipped)" if skipped else ""
    await message.answer(f"Saved {added} new word(s){note}. 📚")
    await _start_round(message, state, message.from_user.id)


@router.message(Learning.awaiting_vocab, F.document)
async def collect_vocab_file(message: Message, state: FSMContext) -> None:
    document = message.document
    if document.mime_type and not document.mime_type.startswith(_ALLOWED_VOCAB_MIME_PREFIXES):
        await message.answer("Please upload a plain <b>.txt</b> file with one word per line.")
        return

    buffer = await message.bot.download(document)
    raw_text = buffer.read().decode("utf-8", errors="ignore")
    words = _parse_words(raw_text)
    if not words:
        await message.answer("That file didn't contain any recognizable words.")
        return

    added = await db.add_words(message.from_user.id, words)
    skipped = len(words) - added
    note = f" ({skipped} duplicate(s) skipped)" if skipped else ""
    await message.answer(f"Saved {added} new word(s) from your file{note}. 📚")
    await _start_round(message, state, message.from_user.id)


@router.message(Learning.awaiting_translation, F.text)
async def evaluate(message: Message, state: FSMContext) -> None:
    telegram_id = message.from_user.id
    user = await db.get_user(telegram_id)
    assert user is not None

    data = await state.get_data()
    sentences = data.get("sentences", [])
    word_ids = data.get("word_ids", [])
    original = "\n".join(sentences)
    source_language, dest_language = _languages_for(user)

    async with ChatActionSender.typing(bot=message.bot, chat_id=message.chat.id):
        try:
            result = await ai_tutor.evaluate_translation(
                original, message.text or "", source_language, dest_language
            )
        except TutorError as exc:
            await message.answer(str(exc))
            return

    if word_ids:
        await db.record_round_result(word_ids, result["score"])

    feedback = ai_tutor.format_feedback(result)
    await message.answer(f"{feedback}\n\n———")
    await _start_round(message, state, telegram_id)


@router.message(Learning.awaiting_vocab)
async def vocab_needs_text(message: Message) -> None:
    await message.answer("Please send vocabulary words as text or a .txt file. 🙂")


@router.message(Learning.awaiting_translation)
async def translation_needs_text(message: Message) -> None:
    await message.answer("Please send your translation as text. 🙂")
