"""/start command, proficiency-level selection, and target-language selection."""
from __future__ import annotations

from aiogram import F, Router
from aiogram.filters import Command, CommandStart
from aiogram.fsm.context import FSMContext
from aiogram.types import (
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
)

from app.config import settings
from app.states.learning import Learning

router = Router(name="start")

LEVELS = ["A1", "A2", "B1", "B2", "C1"]
# A short list of suggestions; users can also type any language they like.
SUGGESTED_LANGUAGES = ["Spanish", "French", "German", "Hebrew", "Italian", "Japanese"]


def _levels_keyboard() -> InlineKeyboardMarkup:
    row = [InlineKeyboardButton(text=lvl, callback_data=f"level:{lvl}") for lvl in LEVELS]
    return InlineKeyboardMarkup(inline_keyboard=[row])


def _languages_keyboard() -> InlineKeyboardMarkup:
    buttons = [
        InlineKeyboardButton(text=lang, callback_data=f"lang:{lang}")
        for lang in SUGGESTED_LANGUAGES
    ]
    # Two per row, then a hint row telling users they can type their own.
    rows = [buttons[i : i + 2] for i in range(0, len(buttons), 2)]
    return InlineKeyboardMarkup(inline_keyboard=rows)


@router.message(CommandStart())
async def cmd_start(message: Message, state: FSMContext) -> None:
    await state.clear()
    await state.set_state(Learning.choosing_level)
    await message.answer(
        f"👋 Welcome to <b>{settings.app_name}</b> — your multilingual AI language tutor!\n\n"
        "I generate practice sentences from your vocabulary and grade your "
        "translations, in <b>any</b> language and at your level.\n\n"
        "First, choose your proficiency level:",
        reply_markup=_levels_keyboard(),
    )


@router.message(Command("help"))
async def cmd_help(message: Message) -> None:
    await message.answer(
        f"<b>{settings.app_name}</b> — how it works:\n\n"
        "1. /start and pick your level (A1–C1).\n"
        "2. Pick or type your target practice language.\n"
        "3. Send vocabulary words (e.g. <i>negotiate, resilient, breakthrough</i>).\n"
        "4. I'll give you 3 sentences to translate.\n"
        "5. Send your translation and I'll score it with corrections.\n\n"
        "Use /start any time to begin again."
    )


@router.callback_query(Learning.choosing_level, F.data.startswith("level:"))
async def choose_level(callback: CallbackQuery, state: FSMContext) -> None:
    level = callback.data.split(":", 1)[1]
    await state.update_data(level=level)
    await state.set_state(Learning.choosing_language)
    await callback.message.edit_text(  # type: ignore[union-attr]
        f"Level set to <b>{level}</b>. 🎯\n\n"
        "Now choose your target practice language — tap one below, "
        "or just <b>type any language</b> you want to practice:",
        reply_markup=_languages_keyboard(),
    )
    await callback.answer()


@router.callback_query(Learning.choosing_language, F.data.startswith("lang:"))
async def choose_language_button(callback: CallbackQuery, state: FSMContext) -> None:
    language = callback.data.split(":", 1)[1]
    await _set_language(callback.message, state, language)  # type: ignore[arg-type]
    await callback.answer()


@router.message(Learning.choosing_language, F.text)
async def choose_language_text(message: Message, state: FSMContext) -> None:
    language = (message.text or "").strip()
    if not language:
        await message.answer("Please type a language name, e.g. <i>Portuguese</i>.")
        return
    await _set_language(message, state, language)


async def _set_language(message: Message, state: FSMContext, language: str) -> None:
    await state.update_data(language=language)
    await state.set_state(Learning.awaiting_vocab)
    await message.answer(
        f"Great — we'll practice <b>{language}</b>. 🌍\n\n"
        "Send me some vocabulary words to practice (comma-, space-, or "
        "newline-separated), for example:\n"
        "<i>negotiate, resilient, breakthrough</i>"
    )
