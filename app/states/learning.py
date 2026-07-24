"""Finite-state machine for the learning flow.

Flow: /start -> choosing_level -> choosing_native_language
      -> choosing_target_language -> choosing_direction -> awaiting_vocab
      -> awaiting_translation -> (loop back to awaiting_vocab)

The profile (level/native_language/target_language/direction) is persisted to
the database once complete (see app/db.py) — FSM ``data`` only carries
transient, in-flight round state:
    level:            str        proficiency level, e.g. "B1" (during setup)
    native_language:  str        language the learner already knows (during setup)
    target_language:  str        language the learner is practicing (during setup)
    sentences:        list[str]  the current round's generated practice sentences
    word_ids:         list[int]  DB ids of the words used in the current round
"""
from aiogram.fsm.state import State, StatesGroup


class Learning(StatesGroup):
    choosing_level = State()
    choosing_native_language = State()
    choosing_target_language = State()
    choosing_direction = State()
    awaiting_vocab = State()
    awaiting_translation = State()
