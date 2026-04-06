import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

from openai import AzureOpenAI, APIError, RateLimitError, APITimeoutError
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)

from config.settings import get_settings
from core.state import BiomarkerState

settings = get_settings()

logger = logging.getLogger(__name__)

_RETRY_EXCEPTIONS = (RateLimitError, APITimeoutError)


class BaseAgent(ABC):
    """
    Abstract base for all pipeline agents.

    Provides:
    - Shared AzureOpenAI client (one per agent class, not per request)
    - System prompt loading from file
    - LLM call with exponential-backoff retry on rate-limit / timeout
    - Structured logging at every LLM call
    """

    def __init__(self, deployment_name: str, system_prompt_path: str) -> None:
        self.deployment_name = deployment_name
        self.system_prompt = self._load_prompt(system_prompt_path)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.client = AzureOpenAI(
            azure_endpoint=settings.azure_openai_endpoint,
            api_key=settings.azure_openai_api_key,
            api_version=settings.azure_openai_api_version,
        )

    # ── Prompt loading ────────────────────────────────────────────────────────

    def _load_prompt(self, path: str) -> str:
        try:
            return Path(path).read_text(encoding="utf-8").strip()
        except FileNotFoundError:
            logger.warning("Prompt file not found: %s — using default.", path)
            return "You are a helpful biomarker discovery assistant."

    # ── LLM call with retry ───────────────────────────────────────────────────

    @retry(
        retry=retry_if_exception_type(_RETRY_EXCEPTIONS),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        stop=stop_after_attempt(4),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    def _call_llm(
        self,
        messages: list,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> str:
        self.logger.debug(
            "LLM call | model=%s | messages=%d | max_tokens=%d",
            self.deployment_name,
            len(messages),
            max_tokens,
        )
        try:
            response = self.client.chat.completions.create(
                model=self.deployment_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            content = response.choices[0].message.content or ""
            self.logger.debug(
                "LLM response | tokens_used=%d | finish=%s",
                response.usage.total_tokens,
                response.choices[0].finish_reason,
            )
            return content
        except APIError as exc:
            self.logger.error("LLM API error: %s", exc)
            raise

    # ── Pipeline entry point (each subclass implements) ───────────────────────

    @abstractmethod
    def run(self, state: BiomarkerState) -> BiomarkerState:
        """Process the current pipeline state and return an updated state."""
