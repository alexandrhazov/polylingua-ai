"""Application configuration loaded from environment variables / .env file."""
from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration for PolyLingua AI.

    Values are read from environment variables (Render dashboard in production)
    or a local ``.env`` file during development.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Secrets (required) ---
    bot_token: str  # env: BOT_TOKEN — from @BotFather
    gemini_api_key: str  # env: GEMINI_API_KEY

    # --- Webhook ---
    # Public base URL of the deployed service, e.g. https://polylingua-ai.onrender.com
    webhook_url: str  # env: WEBHOOK_URL
    webhook_path: str = "/webhook"
    # Optional shared secret; Telegram echoes it back in the
    # X-Telegram-Bot-Api-Secret-Token header so we can reject spoofed calls.
    webhook_secret: str = ""

    # --- Product / model ---
    app_name: str = "PolyLingua AI"
    # Gemini's fastest, most cost-effective tier with strong multilingual coverage.
    model: str = "gemini-2.5-flash"
    temperature: float = 0.3
    max_tokens: int = 1024

    @property
    def webhook_full_url(self) -> str:
        """Absolute webhook URL Telegram should POST updates to."""
        return f"{self.webhook_url.rstrip('/')}{self.webhook_path}"


settings = Settings()  # type: ignore[call-arg]  # values supplied via env/.env
