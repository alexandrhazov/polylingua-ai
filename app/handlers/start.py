"""/start command, and the level / native-language / target-language / direction setup flow."""
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

from app import db
from app.config import settings
from app.states.learning import Learning

router = Router(name="start")

LEVELS = ["A1", "A2", "B1", "B2", "C1"]
# A short list of suggestions; users can also type any language they like.
SUGGESTED_LANGUAGES = ["Spanish", "French", "German", "Hebrew", "Italian", "Japanese"]


def _levels_keyboard() -> InlineKeyboardMarkup:
    row = [InlineKeyboardButton(text=lvl, callback_data=f"level:{lvl}") for lvl in LEVELS]
    return InlineKeyboardMarkup(inline_keyboard=[row])


def _languages_keyboard(prefix: str) -> InlineKeyboardMarkup:
    buttons = [
        InlineKeyboardButton(text=lang, callback_data=f"{prefix}:{lang}")
        for lang in SUGGESTED_LANGUAGES
    ]
    # Two per row, then a hint row telling users they can type their own.
    rows = [buttons[i : i + 2] for i in range(0, len(buttons), 2)]
    return InlineKeyboardMarkup(inline_keyboard=rows)


def _direction_keyboard(native: str, target: str) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(
                text=f"{native} ➜ {target} (production)",
                callback_data="dir:native_to_target",
            )],
            [InlineKeyboardButton(
                text=f"{target} ➜ {native} (comprehension)",
                callback_data="dir:target_to_native",
            )],
        ]
    )


def _generate_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[[InlineKeyboardButton(
            text="🎲 Quiz me at random", callback_data="random_practice",
        )]]
    )


async def _begin_setup(message: Message, state: FSMContext, greeting: str) -> None:
    await state.clear()
    await state.set_state(Learning.choosing_level)
    await message.answer(
        f"{greeting}\n\nFirst, choose your proficiency level:",
        reply_markup=_levels_keyboard(),
    )


@router.message(CommandStart())
async def cmd_start(message: Message, state: FSMContext) -> None:
    await _begin_setup(
        message,
        state,
        f"👋 Welcome to <b>{settings.app_name}</b> — your multilingual AI language tutor!\n\n"
        "I generate practice sentences from your vocabulary and grade your "
        "translations, in <b>any</b> language and at your level.",
    )


@router.message(Command("settings"))
async def cmd_settings(message: Message, state: FSMContext) -> None:
    await _begin_setup(
        message,
        state,
        "⚙️ Let's update your level, languages, or direction. Your saved "
        "vocabulary won't be touched.",
    )


@router.message(Command("help"))
async def cmd_help(message: Message) -> None:
    await message.answer(
        f"<b>{settings.app_name}</b> — how it works:\n\n"
        "1. /start and pick your level (A1–C1).\n"
        "2. Pick the language you already know, then the one you're learning.\n"
        "3. Pick a direction: translate INTO your target language (production) "
        "or FROM it (comprehension).\n"
        "4. Send vocabulary words as text, or upload a .txt file with one word "
        "per line — send as many as you like, I'll remember them all and quiz "
        "you on them, cycling through the list round after round. Or skip "
        "this and use /generate (or the button) for random AI-picked "
        "sentences instead — no list needed, nothing saved.\n"
        "5. Either way, I grade each round with a score and corrections.\n\n"
        "Other commands:\n"
        "• /generate — random sentences for your level, no list needed\n"
        "• /skip — skip the current round without grading it\n"
        "• /settings — change level, languages, or direction\n"
        "• /start — full reset\n"
    )


@router.callback_query(Learning.choosing_level, F.data.startswith("level:"))
async def choose_level(callback: CallbackQuery, state: FSMContext) -> None:
    level = callback.data.split(":", 1)[1]
    await state.update_data(level=level)
    await state.set_state(Learning.choosing_native_language)
    await callback.message.edit_text(  # type: ignore[union-attr]
        f"Level set to <b>{level}</b>. 🎯\n\n"
        "Which language do you already know well (your native/fluent "
        "language)? Tap one below, or <b>type any language</b>:",
        reply_markup=_languages_keyboard("native"),
    )
    await callback.answer()


@router.callback_query(Learning.choosing_native_language, F.data.startswith("native:"))
async def choose_native_language_button(callback: CallbackQuery, state: FSMContext) -> None:
    language = callback.data.split(":", 1)[1]
    await _set_native_language(callback.message, state, language)  # type: ignore[arg-type]
    await callback.answer()


@router.message(Learning.choosing_native_language, F.text)
async def choose_native_language_text(message: Message, state: FSMContext) -> None:
    language = (message.text or "").strip()
    if not language:
        await message.answer("Please type a language name, e.g. <i>Russian</i>.")
        return
    await _set_native_language(message, state, language)


async def _set_native_language(message: Message, state: FSMContext, language: str) -> None:
    await state.update_data(native_language=language)
    await state.set_state(Learning.choosing_target_language)
    await message.answer(
        f"Got it — you know <b>{language}</b>. 🌍\n\n"
        "Now, which language do you want to practice? Tap one below, or "
        "<b>type any language</b>:",
        reply_markup=_languages_keyboard("target"),
    )


@router.callback_query(Learning.choosing_target_language, F.data.startswith("target:"))
async def choose_target_language_button(callback: CallbackQuery, state: FSMContext) -> None:
    language = callback.data.split(":", 1)[1]
    await _set_target_language(callback.message, state, language)  # type: ignore[arg-type]
    await callback.answer()


@router.message(Learning.choosing_target_language, F.text)
async def choose_target_language_text(message: Message, state: FSMContext) -> None:
    language = (message.text or "").strip()
    if not language:
        await message.answer("Please type a language name, e.g. <i>Hebrew</i>.")
        return
    await _set_target_language(message, state, language)


async def _set_target_language(message: Message, state: FSMContext, language: str) -> None:
    data = await state.update_data(target_language=language)
    await state.set_state(Learning.choosing_direction)
    native = data["native_language"]
    await message.answer(
        f"Great — <b>{native} → {language}</b>. 🎯\n\n"
        "Which direction do you want to practice?\n"
        f"• <b>{native} ➜ {language}</b>: I show a sentence in {native}, you "
        f"translate it into {language} (production practice).\n"
        f"• <b>{language} ➜ {native}</b>: I show a sentence in {language}, you "
        f"translate it into {native} (comprehension practice).",
        reply_markup=_direction_keyboard(native, language),
    )


@router.callback_query(Learning.choosing_direction, F.data.startswith("dir:"))
async def choose_direction(callback: CallbackQuery, state: FSMContext) -> None:
    # Acknowledge immediately — the DB round-trip below could occasionally be
    # slow enough to invalidate the callback query otherwise.
    await callback.answer()
    direction = callback.data.split(":", 1)[1]
    data = await state.update_data(direction=direction)
    await db.upsert_profile(
        telegram_id=callback.from_user.id,
        level=data["level"],
        native_language=data["native_language"],
        target_language=data["target_language"],
        direction=direction,
    )
    await state.set_state(Learning.awaiting_vocab)
    await callback.message.edit_text(  # type: ignore[union-attr]
        "Setup complete! ✅\n\n"
        "Send me vocabulary words to practice — as a message (comma-, space-, "
        "or newline-separated) or as a <b>.txt file</b> with one word per "
        "line. Send as many as you like, e.g. hundreds at once — I'll "
        "remember them all and quiz you on them, a few at a time, cycling "
        "through the list round after round.\n\n"
        "Example: <i>negotiate, resilient, breakthrough</i>\n\n"
        "Don't have a list? Tap below to get random sentences for your level instead.",
        reply_markup=_generate_keyboard(),
    )
