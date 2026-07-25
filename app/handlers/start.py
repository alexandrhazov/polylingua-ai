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
SUGGESTED_LANGUAGES = ["Russian", "English", "Hebrew", "Spanish", "French"]

# Bounds for the /count command (sentences per round).
MIN_ROUND_SIZE = 1
MAX_ROUND_SIZE = 10


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
        f"{greeting}\n\nWhat's your level?",
        reply_markup=_levels_keyboard(),
    )


@router.message(CommandStart())
async def cmd_start(message: Message, state: FSMContext) -> None:
    await _begin_setup(
        message,
        state,
        f"👋 Hi! I'm <b>{settings.app_name}</b>. I make practice sentences and "
        "grade your translations — any language, your level.",
    )


@router.message(Command("settings"))
async def cmd_settings(message: Message, state: FSMContext) -> None:
    await _begin_setup(
        message,
        state,
        "⚙️ Let's update your setup. Your saved words stay put.",
    )


@router.message(Command("help"))
async def cmd_help(message: Message) -> None:
    await message.answer(
        f"<b>{settings.app_name}</b> — quick guide:\n\n"
        "Send me words (text or a .txt file) and I'll quiz you on them, a few "
        "at a time. No list? Use /generate for random sentences. Either way, "
        "you translate and I grade you.\n\n"
        "Commands:\n"
        "• /generate — random sentences, no list needed\n"
        "• /count — set sentences per round (e.g. /count 5)\n"
        "• /skip — skip this round\n"
        "• /settings — change level, languages, or direction\n"
        "• /start — start over\n"
    )


@router.message(Command("count"))
async def cmd_count(message: Message) -> None:
    user = await db.get_user(message.from_user.id)
    if user is None:
        await message.answer("Finish /start first, then you can set this. 🙂")
        return

    arg = (message.text or "").split(maxsplit=1)
    if len(arg) < 2:
        await message.answer(
            f"You're getting <b>{user.round_size}</b> sentence(s) per round.\n"
            f"Change it with e.g. <code>/count 5</code> "
            f"({MIN_ROUND_SIZE}–{MAX_ROUND_SIZE})."
        )
        return

    try:
        size = int(arg[1].strip())
    except ValueError:
        await message.answer(f"Give me a number, e.g. <code>/count 5</code> "
                             f"({MIN_ROUND_SIZE}–{MAX_ROUND_SIZE}).")
        return

    if not MIN_ROUND_SIZE <= size <= MAX_ROUND_SIZE:
        await message.answer(f"Pick a number between {MIN_ROUND_SIZE} and {MAX_ROUND_SIZE}.")
        return

    await db.set_round_size(message.from_user.id, size)
    await message.answer(f"Done — <b>{size}</b> sentence(s) per round from now on. 👍")


@router.callback_query(Learning.choosing_level, F.data.startswith("level:"))
async def choose_level(callback: CallbackQuery, state: FSMContext) -> None:
    level = callback.data.split(":", 1)[1]
    await state.update_data(level=level)
    await state.set_state(Learning.choosing_native_language)
    await callback.message.edit_text(  # type: ignore[union-attr]
        f"<b>{level}</b> it is. 🎯\n\n"
        "Which language do you already know? Tap one or type your own:",
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
        await message.answer("Type a language name, e.g. <i>Russian</i>.")
        return
    await _set_native_language(message, state, language)


async def _set_native_language(message: Message, state: FSMContext, language: str) -> None:
    await state.update_data(native_language=language)
    await state.set_state(Learning.choosing_target_language)
    await message.answer(
        f"Got it — you know <b>{language}</b>. 🌍\n\n"
        "And which one do you want to practice? Tap or type:",
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
        await message.answer("Type a language name, e.g. <i>Hebrew</i>.")
        return
    await _set_target_language(message, state, language)


async def _set_target_language(message: Message, state: FSMContext, language: str) -> None:
    data = await state.update_data(target_language=language)
    await state.set_state(Learning.choosing_direction)
    native = data["native_language"]
    await message.answer(
        f"Nice — <b>{native} → {language}</b>. 🎯\n\n"
        "Which way do you want to practice?",
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
        "All set! ✅\n\n"
        "Send me some words to practice — a message or a <b>.txt file</b> "
        "(one per line). As many as you like; I'll quiz you a few at a time.\n\n"
        "Example: <i>negotiate, resilient, breakthrough</i>\n\n"
        "No list? Tap below for random sentences.",
        reply_markup=_generate_keyboard(),
    )
