"""
agents/base_agent.py
Abstract base for all pipeline agents.

LangSmith integration
─────────────────────
• _build_client()  wraps the AzureOpenAI client with langsmith.wrappers.wrap_openai
  so every chat.completions.create() call is automatically traced as a child span
  (latency, prompt tokens, completion tokens, model name, full input/output).
  This works for ALL subclasses without any per-agent change.

• json_mode flag   on _call_llm forces response_format={"type":"json_object"}.
  Use it on every call that expects structured JSON — it prevents the model from
  wrapping JSON in markdown fences or inserting prose before the payload, which
  are the two most common root causes of JSON parse failures and hallucination
  in structured-output pipelines.
"""
import logging
from abc import ABC, abstractmethod
from pathlib import Path

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
logger   = logging.getLogger(__name__)

_RETRY_EXCEPTIONS = (RateLimitError, APITimeoutError)


def _build_client() -> AzureOpenAI:
    """
    Create the AzureOpenAI client.  When LangSmith tracing is enabled in
    settings, wrap it with wrap_openai so every completion call is auto-traced
    as a LangSmith LLM span (latency, tokens, input messages, output text).
    Falls back to the bare client if langsmith is not installed.
    """
    raw = AzureOpenAI(
        azure_endpoint=settings.azure_openai_endpoint,
        api_key=settings.azure_openai_api_key,
        api_version=settings.azure_openai_api_version,
    )
    if settings.langsmith_tracing:
        try:
            from langsmith.wrappers import wrap_openai
            return wrap_openai(raw)
        except ImportError:
            logger.warning(
                "langsmith not installed — pip install langsmith to enable "
                "LLM call tracing (latency, tokens, prompts)."
            )
    return raw


class BaseAgent(ABC):
    """
    Abstract base for all pipeline agents.

    Provides:
    - AzureOpenAI client with optional LangSmith auto-tracing (wrap_openai)
    - System prompt loading from file
    - LLM call with exponential-backoff retry on rate-limit / timeout
    - json_mode=True on _call_llm to enforce JSON-only output (anti-hallucination)
    - Structured logging at every LLM call
    """

    def __init__(self, deployment_name: str, system_prompt_path: str) -> None:
        self.deployment_name = deployment_name
        self.system_prompt   = self._load_prompt(system_prompt_path)
        self.logger          = logging.getLogger(self.__class__.__name__)
        self.client          = _build_client()

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
        json_mode: bool = False,
    ) -> str:
        """
        Call the LLM and return the text response.

        Set json_mode=True whenever the expected response is a JSON object.
        This activates response_format={"type":"json_object"} on the Azure
        OpenAI API, which guarantees the model returns valid JSON and removes
        the need for regex-based markdown-fence stripping on the caller side.
        """
        self.logger.debug(
            "LLM call | model=%s | messages=%d | max_tokens=%d | json_mode=%s",
            self.deployment_name, len(messages), max_tokens, json_mode,
        )
        kwargs: dict = dict(
            model       = self.deployment_name,
            messages    = messages,
            max_tokens  = max_tokens,
            temperature = temperature,
        )
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        try:
            response = self.client.chat.completions.create(**kwargs)
            content  = response.choices[0].message.content or ""
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
