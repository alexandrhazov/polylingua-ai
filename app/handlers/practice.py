"""Vocabulary collection, sentence generation, and translation evaluation."""
from __future__ import annotations

import html
import re

from aiogram import F, Router
from aiogram.fsm.context import FSMContext
from aiogram.types import Message
from aiogram.utils.chat_action import ChatActionSender

from app.services import ai_tutor
from app.services.ai_tutor import TutorError
from app.states.learning import Learning

router = Router(name="practice")

# Split on commas, semicolons, newlines, or runs of whitespace.
_SPLIT_RE = re.compile(r"[,\n;]+|\s{2,}")


def _parse_words(text: str) -> list[str]:
    raw = _SPLIT_RE.split(text) if ("," in text or "\n" in text or ";" in text) else text.split()
    return [w.strip() for w in raw if w.strip()]


@router.message(Learning.awaiting_vocab, F.text)
async def collect_vocab(message: Message, state: FSMContext) -> None:
    words = _parse_words(message.text or "")
    if not words:
        await message.answer(
            "Please send at least one vocabulary word, e.g. "
            "<i>negotiate, resilient, breakthrough</i>."
        )
        return

    data = await state.get_data()
    level, language = data["level"], data["language"]

    async with ChatActionSender.typing(bot=message.bot, chat_id=message.chat.id):
        try:
            sentences = await ai_tutor.generate_sentences(words, level, language)
        except TutorError as exc:
            await message.answer(str(exc))
            return

    await state.update_data(sentences=sentences)
    await state.set_state(Learning.awaiting_translation)

    numbered = "\n".join(
        f"{i}. {html.escape(s)}" for i, s in enumerate(sentences, start=1)
    )
    await message.answer(
        f"Here are your <b>{language}</b> practice sentences ({level}):\n\n"
        f"{numbered}\n\n"
        "✍️ Now send me your translation and I'll grade it."
    )


@router.message(Learning.awaiting_translation, F.text)
async def evaluate(message: Message, state: FSMContext) -> None:
    data = await state.get_data()
    language = data["language"]
    sentences = data.get("sentences", [])
    original = "\n".join(sentences)

    async with ChatActionSender.typing(bot=message.bot, chat_id=message.chat.id):
        try:
            result = await ai_tutor.evaluate_translation(
                original, message.text or "", language
            )
        except TutorError as exc:
            await message.answer(str(exc))
            return

    feedback = ai_tutor.format_feedback(result)
    await state.set_state(Learning.awaiting_vocab)
    await message.answer(
        f"{feedback}\n\n"
        "———\n"
        "Send more vocabulary to keep practicing, or /start to change level or language."
    )


@router.message(Learning.awaiting_vocab)
async def vocab_needs_text(message: Message) -> None:
    await message.answer("Please send vocabulary words as text. 🙂")


@router.message(Learning.awaiting_translation)
async def translation_needs_text(message: Message) -> None:
    await message.answer("Please send your translation as text. 🙂")
