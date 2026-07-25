"""AI tutor service — wraps Groq's OpenAI-compatible chat API for sentence
generation and grading.

Uses Groq's async client with JSON mode (``response_format={"type":
"json_object"}``) so responses parse as valid JSON. Unlike Gemini's
schema-constrained structured output, JSON mode only guarantees valid JSON
syntax — not a specific shape — so each prompt spells out the exact structure
expected. The model, temperature, and token budget all come from ``settings``
so there are no hardcoded values scattered across the codebase.
"""
from __future__ import annotations

import html
import json
import logging
from typing import Any

import groq
from groq import AsyncGroq

from app.config import settings

logger = logging.getLogger(__name__)

# Bounded timeout + a single retry: the SDK's defaults (no timeout, 2 retries
# with exponential backoff honoring the server's suggested delay) can let one
# rate-limited call hang for minutes before finally raising — which felt like
# the whole bot going silent, since that one slow call held up the reply the
# user was waiting on. Failing fast means TutorError reaches the user quickly
# instead.
client = AsyncGroq(api_key=settings.groq_api_key, timeout=20.0, max_retries=1)

# --- System prompt ---------------------------------------------------------
# Forces the model to support ANY natural language and to calibrate difficulty
# to the requested CEFR proficiency level. Kept as a single template so the
# same instructions drive both generation and grading. Must mention "JSON"
# explicitly — Groq's JSON mode requires it somewhere in the messages.
_SYSTEM_PROMPT = (
    "You are {app_name}, an expert multilingual language tutor. "
    "You fluently support EVERY natural language a learner might request — "
    "Spanish, French, German, Hebrew, Arabic, Mandarin, Japanese, Swahili, and "
    "any other — including right-to-left and non-Latin scripts. "
    "Always honor the requested languages and CEFR proficiency level "
    "(A1 = beginner ... C1 = advanced), calibrating vocabulary and grammar "
    "complexity to that level. When grading, treat numerals and their "
    "spelled-out equivalents as identical (e.g. '3' and 'three') — never flag "
    "a difference in number formatting as a mistake, and keep the learner's "
    "original numeral style in any corrected text. Respond ONLY with a single "
    "valid JSON object matching the structure requested in the prompt; never "
    "add commentary or markdown fences outside it."
)


def _system() -> str:
    return _SYSTEM_PROMPT.format(app_name=settings.app_name)


async def _structured_call(prompt: str) -> dict[str, Any]:
    """Make one Groq call in JSON mode and return the parsed object."""
    response = await client.chat.completions.create(
        model=settings.model,
        temperature=settings.temperature,
        max_tokens=settings.max_tokens,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": _system()},
            {"role": "user", "content": prompt},
        ],
    )
    return json.loads(response.choices[0].message.content)


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
        "translate these into another language.\n\n"
        'Respond with ONLY a JSON object of this exact form: '
        '{"sentences": ["...", "..."]} — exactly one sentence per vocabulary '
        "word, in the same order as the vocabulary words."
    )
    try:
        data = await _structured_call(prompt)
        sentences = [s.strip() for s in data.get("sentences", []) if s.strip()]
        if not sentences:
            raise ValueError("model returned no sentences")
        return sentences[: len(words)]
    except (groq.APIError, ValueError, json.JSONDecodeError, KeyError, TypeError) as exc:
        logger.exception("generate_sentences failed")
        raise TutorError("I couldn't generate practice sentences right now. "
                         "Please try again in a moment.") from exc


async def generate_random_sentences(level: str, language: str, count: int) -> list[str]:
    """Generate ``count`` standalone practice sentences in ``language`` for ``level``.

    Used for stateless practice, when the learner hasn't supplied their own
    vocabulary — the model picks its own content, nothing is tracked or
    stored. Raises ``TutorError`` on API/parse failure.
    """
    prompt = (
        f"Sentence language: {language}\n"
        f"Proficiency level (CEFR): {level}\n\n"
        f"Write exactly {count} distinct, natural, useful sentences in "
        f"{language}, suited to a {level} learner. Vary the topics and "
        "vocabulary across the sentences. The learner will translate them.\n\n"
        'Respond with ONLY a JSON object of this exact form: '
        f'{{"sentences": ["...", "..."]}} — containing exactly {count} sentences.'
    )
    try:
        data = await _structured_call(prompt)
        sentences = [s.strip() for s in data.get("sentences", []) if s.strip()]
        if not sentences:
            raise ValueError("model returned no sentences")
        return sentences[:count]
    except (groq.APIError, ValueError, json.JSONDecodeError, KeyError, TypeError) as exc:
        logger.exception("generate_random_sentences failed")
        raise TutorError("I couldn't generate practice sentences right now. "
                         "Please try again in a moment.") from exc


async def evaluate_translation(
    sentences: list[str], user_translation: str, source_language: str, dest_language: str
) -> dict[str, Any]:
    """Grade a ``dest_language`` translation of ``sentences`` (in ``source_language``).

    Returns a dict with keys: ``score`` (float) and ``items`` (a list, one entry
    per practice sentence, each with ``mistakes`` (list[str]), ``corrected``
    (str), and ``grammar`` (str)). Raises ``TutorError`` on API/parse failure.
    """
    numbered = "\n".join(f"{i}. {s}" for i, s in enumerate(sentences, start=1))
    prompt = (
        f"The following {len(sentences)} practice sentence(s) are in {source_language}:\n"
        f"{numbered}\n\n"
        f"The learner translated them into {dest_language}, submitting:\n"
        f"{user_translation}\n\n"
        f"Evaluate the {dest_language} translation, one entry per numbered "
        "sentence, in the same order.\n\n"
        'Respond with ONLY a JSON object of this exact form: '
        '{"score": <overall number 0-10, one decimal allowed>, '
        f'"items": [{{"mistakes": ["<specific mistake in the learner\'s '
        'translation>", "..."], "corrected": "<the full, natural, '
        f'native-sounding correct translation of this sentence in {dest_language}>", '
        '"grammar": "<short explanation of the grammar behind the fix>"}}, ...]}. '
        f"Include exactly one item per practice sentence ({len(sentences)} total), "
        "in order. If a sentence's translation is already correct, use an empty "
        '"mistakes" list and an empty "grammar" string. Leave "grammar" empty '
        "when there is nothing worth explaining."
    )
    try:
        data = await _structured_call(prompt)
        # Clamp score defensively into 0-10.
        score = max(0.0, min(10.0, float(data.get("score", 0))))
        raw_items = data.get("items") or []
        if not isinstance(raw_items, list):
            raw_items = []
        items = []
        for entry in raw_items:
            if not isinstance(entry, dict):
                continue
            raw_mistakes = entry.get("mistakes") or []
            if not isinstance(raw_mistakes, list):
                raw_mistakes = []
            items.append({
                "mistakes": [str(m).strip() for m in raw_mistakes if str(m).strip()],
                "corrected": str(entry.get("corrected", "")).strip(),
                "grammar": str(entry.get("grammar", "")).strip(),
            })
        return {"score": round(score, 1), "items": items}
    except (groq.APIError, ValueError, json.JSONDecodeError, KeyError, TypeError) as exc:
        logger.exception("evaluate_translation failed")
        raise TutorError("I couldn't check your translation just now. "
                         "Give it another try in a moment.") from exc


def format_feedback(result: dict[str, Any]) -> str:
    """Render an evaluation dict into an HTML Telegram message."""
    items = result.get("items", [])
    lines = [f"⭐ <b>Accuracy:</b> {result['score']}/10"]
    multi = len(items) > 1
    for i, item in enumerate(items, start=1):
        block = ["", f"<b>Sentence {i}</b>"] if multi else [""]
        if not item["mistakes"]:
            block.append("✅ Correct — nicely done!")
        else:
            block += [f"{n}. {html.escape(m)}" for n, m in enumerate(item["mistakes"], start=1)]
            if item["corrected"]:
                block.append(f"✅ {html.escape(item['corrected'])}")
            if item["grammar"]:
                block.append(f"📝 {html.escape(item['grammar'])}")
        lines += block
    return "\n".join(lines)


class TutorError(Exception):
    """Raised when the AI tutor cannot fulfil a request; message is user-safe."""
