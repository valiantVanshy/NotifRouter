"""Application settings, loaded from environment variables and an optional .env file."""

from functools import lru_cache
from pathlib import Path

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DATASET_DIR = PROJECT_DIR.parent / "dataset"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="NOTIF_ROUTER_",
        env_file=PROJECT_DIR / ".env",
        extra="ignore",
        populate_by_name=True,
    )

    # Any OpenAI-compatible provider works (OpenAI, Groq, Gemini, Ollama...).
    # The key is read from NOTIF_ROUTER_LLM_API_KEY, falling back to OPENAI_API_KEY.
    llm_api_key: str | None = Field(
        default=None, validation_alias=AliasChoices("NOTIF_ROUTER_LLM_API_KEY", "OPENAI_API_KEY")
    )
    llm_base_url: str | None = None  # None = api.openai.com

    model: str = "gpt-4o"
    # Comma-separated chat models to use when `model` hits its daily quota, e.g. "openai/gpt-oss-20b".
    fallback_models: str = ""
    vision_model: str = "gpt-4o"
    transcription_model: str = "whisper-1"
    temperature: float = 0.0
    seed: int = 7
    agent_max_steps: int = 6
    llm_timeout_seconds: float = 60.0
    llm_max_retries: int = 3
    # How long one call may wait out transient rate limits. Keep it short for the API;
    # batch commands (route/evaluate) raise it so free-tier runs pace instead of failing.
    llm_rate_limit_max_wait_seconds: float = 20.0

    dataset_dir: Path = DEFAULT_DATASET_DIR
    database_url: str = f"sqlite:///{PROJECT_DIR / 'notif_router.db'}"

    api_key: str | None = None
    log_level: str = "INFO"

    @property
    def fallback_model_list(self) -> list[str]:
        return [m.strip() for m in self.fallback_models.split(",") if m.strip()]

    @property
    def llm_enabled(self) -> bool:
        return bool(self.llm_api_key)


@lru_cache
def get_settings() -> Settings:
    return Settings()
