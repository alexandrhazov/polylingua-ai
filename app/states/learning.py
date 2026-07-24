"""Finite-state machine for the learning flow.

Flow: /start -> choosing_level -> choosing_language -> awaiting_vocab
      -> awaiting_translation -> (loop back to awaiting_vocab)

FSM ``data`` carries:
    level:     str        proficiency level, e.g. "B1"
    language:  str        target practice language, e.g. "Spanish"
    sentences: list[str]  the 3 generated practice sentences
"""
from aiogram.fsm.state import State, StatesGroup


class Learning(StatesGroup):
    choosing_level = State()
    choosing_language = State()
    awaiting_vocab = State()
    awaiting_translation = State()
