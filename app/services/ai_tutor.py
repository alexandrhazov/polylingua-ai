"""AI tutor service — wraps the Gemini API for sentence generation and grading.

Uses the async google-genai SDK with structured (JSON-schema) outputs so
responses parse reliably. The model, temperature, and token budget all come
from ``settings`` so there are no hardcoded values scattered across the
codebase.
"""
from __future__ import annotations

import json
import logging
from typing import Any

from google import genai
from google.genai import errors as genai_errors
from google.genai import types

from app.config import settings

logger = logging.getLogger(__name__)

client = genai.Client(api_key=settings.gemini_api_key)

# --- System prompt ---------------------------------------------------------
# Forces the model to support ANY natural language and to calibrate difficulty
# to the requested CEFR proficiency level. Kept as a single template so the
# same instructions drive both generation and grading.
_SYSTEM_PROMPT = (
    "You are {app_name}, an expert multilingual language tutor. "
    "You fluently support EVERY natural language a learner might request — "
    "Spanish, French, German, Hebrew, Arabic, Mandarin, Japanese, Swahili, and "
    "any other — including right-to-left and non-Latin scripts. "
    "Always honor the requested languages and CEFR proficiency level "
    "(A1 = beginner ... C1 = advanced), calibrating vocabulary and grammar "
    "complexity to that level. Respond ONLY with the requested JSON structure; "
    "never add commentary outside it."
)

# JSON schemas for structured outputs (Gemini's OpenAPI-subset Schema format).
# Note: numeric/length constraints and `additionalProperties` are not
# supported, so ranges are enforced via the prompt.
_SENTENCES_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "sentences": {
            "type": "array",
            "items": {"type": "string"},
            "description": "One practice sentence per requested vocabulary word.",
        }
    },
    "required": ["sentences"],
}

_EVALUATION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "score": {
            "type": "number",
            "description": "Accuracy score from 0 to 10, one decimal allowed.",
        },
        "corrections": {
            "type": "string",
            "description": "Specific grammar/vocabulary corrections and notes. "
            "Empty string if the translation is already correct.",
        },
        "alternative": {
            "type": "string",
            "description": "A natural, native-sounding alternative phrasing.",
        },
    },
    "required": ["score", "corrections", "alternative"],
}


def _system() -> str:
    return _SYSTEM_PROMPT.format(app_name=settings.app_name)


async def _structured_call(prompt: str, schema: dict[str, Any]) -> dict[str, Any]:
    """Make one Gemini call constrained to ``schema`` and return parsed JSON."""
    response = await client.aio.models.generate_content(
        model=settings.model,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=_system(),
            temperature=settings.temperature,
            max_output_tokens=settings.max_tokens,
            response_mime_type="application/json",
            response_schema=schema,
        ),
    )
    return json.loads(response.text)


async def generate_sentences(words: list[str], level: str, source_language: str) -> list[str]:
    """Generate practice sentences in ``source_language`` using the given vocabulary.

    ``words`` should already be in ``source_language`` — the language the
    learner will read and then translate out of. Returns a list of sentences.
    Raises ``TutorError`` on API/parse failure so callers can present a
    friendly message.
    """
    word_list = ", ".join(words)
    prompt = (
        f"Sentence language: {source_language}\n"
        f"Proficiency level (CEFR): {level}\n"
        f"Vocabulary to practice (already in {source_language}): {word_list}\n\n"
        f"Write one distinct, natural sentence per vocabulary word, all in "
        f"{source_language}. Each sentence must use its corresponding "
        "vocabulary word and suit the proficiency level. The learner will "
        "translate these into another language. Return them in the "
        "`sentences` array, in the same order as the vocabulary words."
    )
    try:
        data = await _structured_call(prompt, _SENTENCES_SCHEMA)
        sentences = [s.strip() for s in data.get("sentences", []) if s.strip()]
        if not sentences:
            raise ValueError("model returned no sentences")
        return sentences[: len(words)]
    except (genai_errors.APIError, ValueError, json.JSONDecodeError, KeyError) as exc:
        logger.exception("generate_sentences failed")
        raise TutorError("I couldn't generate practice sentences right now. "
                         "Please try again in a moment.") from exc


async def evaluate_translation(
    original: str, user_translation: str, source_language: str, dest_language: str
) -> dict[str, Any]:
    """Grade a ``dest_language`` translation of ``original`` (in ``source_language``).

    Returns a dict with keys: score (float), corrections (str), alternative (str).
    Raises ``TutorError`` on API/parse failure.
    """
    prompt = (
        f"The following practice sentence(s) are in {source_language}:\n"
        f"{original}\n\n"
        f"The learner translated them into {dest_language}, submitting:\n"
        f"{user_translation}\n\n"
        f"Evaluate the {dest_language} translation. Provide:\n"
        "1. `score`: accuracy from 0 to 10 (one decimal place allowed).\n"
        "2. `corrections`: specific grammar and vocabulary notes (empty string "
        "if perfect).\n"
        f"3. `alternative`: a natural, native-sounding alternative phrasing in "
        f"{dest_language}."
    )
    try:
        data = await _structured_call(prompt, _EVALUATION_SCHEMA)
        # Clamp score defensively into 0-10.
        score = max(0.0, min(10.0, float(data.get("score", 0))))
        return {
            "score": round(score, 1),
            "corrections": str(data.get("corrections", "")).strip(),
            "alternative": str(data.get("alternative", "")).strip(),
        }
    except (genai_errors.APIError, ValueError, json.JSONDecodeError, KeyError) as exc:
        logger.exception("evaluate_translation failed")
        raise TutorError("I couldn't evaluate your translation right now. "
                         "Please try again in a moment.") from exc


def format_feedback(result: dict[str, Any]) -> str:
    """Render an evaluation dict into an HTML Telegram message."""
    lines = [f"⭐ <b>Accuracy:</b> {result['score']}/10"]
    if result["corrections"]:
        lines.append(f"\n✏️ <b>Corrections &amp; notes:</b>\n{result['corrections']}")
    else:
        lines.append("\n✅ No corrections — nicely done!")
    if result["alternative"]:
        lines.append(f"\n💡 <b>Natural phrasing:</b>\n{result['alternative']}")
    return "\n".join(lines)


class TutorError(Exception):
    """Raised when the AI tutor cannot fulfil a request; message is user-safe."""
