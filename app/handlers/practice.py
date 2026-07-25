"""Two practice modes:

- Vocabulary mode: learner supplies words (text or file), persisted and
  cycled through forever in rounds (least-recently-practiced first). No
  mastery concept — a word never "graduates" out of rotation.
- Random mode (/generate or the button): fully stateless — the AI picks its
  own sentences for the learner's level/language each round, nothing is
  stored, no vocabulary or database involved.

A round's ``word_ids`` FSM data (present for vocabulary mode, absent for
random mode) is what distinguishes which mode drives grading/continuation.
"""
from __future__ import annotations

import html
import re

from aiogram import F, Router
from aiogram.filters import Command
from aiogram.fsm.context import FSMContext
from aiogram.types import CallbackQuery, Message
from aiogram.utils.chat_action import ChatActionSender

from app import db
from app.handlers.start import _generate_keyboard
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


def _numbered_sentences(sentences: list[str]) -> str:
    return "\n".join(f"{i}. {html.escape(s)}" for i, s in enumerate(sentences, start=1))


async def _start_round(message: Message, state: FSMContext, telegram_id: int) -> None:
    """Fetch the next batch of words (least-recently-practiced first) and
    generate practice sentences. If no words are saved yet, asks the learner
    to send some, or use random practice instead.
    """
    user = await db.get_user(telegram_id)
    assert user is not None  # profile is created before awaiting_vocab is reached

    words = await db.next_round(telegram_id, user.round_size)
    if not words:
        await state.set_state(Learning.awaiting_vocab)
        await message.answer(
            "No saved words yet. Send some (text or a .txt file), or tap for "
            "random practice.",
            reply_markup=_generate_keyboard(),
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

    await message.answer(
        f"Your <b>{source_language}</b> sentences ({user.level}):\n\n"
        f"{_numbered_sentences(sentences)}\n\n"
        "✍️ Translate away."
    )


async def _start_random_round(message: Message, state: FSMContext, telegram_id: int) -> None:
    """Generate standalone practice sentences with no vocabulary or DB involved."""
    user = await db.get_user(telegram_id)
    assert user is not None
    source_language, _ = _languages_for(user)

    async with ChatActionSender.typing(bot=message.bot, chat_id=message.chat.id):
        try:
            sentences = await ai_tutor.generate_random_sentences(
                user.level, source_language, user.round_size
            )
        except TutorError as exc:
            await message.answer(str(exc))
            return

    # No word_ids: marks this round as random/stateless for evaluate()/skip().
    await state.update_data(sentences=sentences, word_ids=[])
    await state.set_state(Learning.awaiting_translation)

    await message.answer(
        f"Your <b>{source_language}</b> sentences ({user.level}):\n\n"
        f"{_numbered_sentences(sentences)}\n\n"
        "✍️ Translate away."
    )


# No state filter: /generate should start a fresh random round from anywhere
# (including mid-translation), not only from awaiting_vocab. Registered first
# in this router so it wins over the awaiting_translation text handler below —
# otherwise "/generate" gets graded as if it were a submitted translation.
@router.message(Command("generate"))
async def random_practice_command(message: Message, state: FSMContext) -> None:
    if await db.get_user(message.from_user.id) is None:
        await message.answer("Send /start to set up first. 🙂")
        return
    await _start_random_round(message, state, message.from_user.id)


@router.callback_query(Learning.awaiting_vocab, F.data == "random_practice")
async def random_practice_button(callback: CallbackQuery, state: FSMContext) -> None:
    # Acknowledge immediately — the AI call below can easily exceed
    # Telegram's short callback-answer window, invalidating the query.
    await callback.answer()
    await _start_random_round(callback.message, state, callback.from_user.id)  # type: ignore[arg-type]


@router.message(Learning.awaiting_translation, Command("skip"))
async def skip_round(message: Message, state: FSMContext) -> None:
    telegram_id = message.from_user.id
    data = await state.get_data()
    word_ids = data.get("word_ids", [])
    await message.answer("Skipped. ⏭️")
    if word_ids:
        await db.skip_words(word_ids)
        await _start_round(message, state, telegram_id)
    else:
        await _start_random_round(message, state, telegram_id)


@router.message(Learning.awaiting_vocab, F.text & ~F.text.startswith("/"))
async def collect_vocab_text(message: Message, state: FSMContext) -> None:
    words = _parse_words(message.text or "")
    if not words:
        await message.answer(
            "Send at least one word, e.g. "
            "<i>negotiate, resilient, breakthrough</i> — or a .txt file."
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
        await message.answer("Send a plain <b>.txt</b> file, one word per line.")
        return

    buffer = await message.bot.download(document)
    raw_text = buffer.read().decode("utf-8", errors="ignore")
    words = _parse_words(raw_text)
    if not words:
        await message.answer("Couldn't find any words in that file.")
        return

    added = await db.add_words(message.from_user.id, words)
    skipped = len(words) - added
    note = f" ({skipped} duplicate(s) skipped)" if skipped else ""
    await message.answer(f"Saved {added} new word(s) from your file{note}. 📚")
    await _start_round(message, state, message.from_user.id)


@router.message(Learning.awaiting_translation, F.text & ~F.text.startswith("/"))
async def evaluate(message: Message, state: FSMContext) -> None:
    telegram_id = message.from_user.id
    user = await db.get_user(telegram_id)
    assert user is not None

    data = await state.get_data()
    sentences = data.get("sentences", [])
    word_ids = data.get("word_ids", [])
    source_language, dest_language = _languages_for(user)

    async with ChatActionSender.typing(bot=message.bot, chat_id=message.chat.id):
        try:
            result = await ai_tutor.evaluate_translation(
                sentences,
                message.text or "",
                source_language,
                dest_language,
                user.native_language,
            )
        except TutorError as exc:
            await message.answer(str(exc))
            return

    feedback = ai_tutor.format_feedback(result)
    await message.answer(f"{feedback}\n\n———")

    if word_ids:
        await db.record_round_result(word_ids)
        await _start_round(message, state, telegram_id)
    else:
        await _start_random_round(message, state, telegram_id)


@router.message(Learning.awaiting_vocab)
async def vocab_needs_text(message: Message) -> None:
    await message.answer("Send words as text or a .txt file. 🙂")


@router.message(Learning.awaiting_translation)
async def translation_needs_text(message: Message) -> None:
    await message.answer("Send your translation as text. 🙂")


@router.message()
async def fallback(message: Message) -> None:
    """Catches anything with no matching state (e.g. a brand-new user texting
    before /start, or state that predates this app version). Registered last
    so every more specific handler above gets first refusal.
    """
    await message.answer("Send /start to begin. 🙂")
