"""Universal model adapter using LiteLLM.

Supports 100+ LLM providers through a single interface:
- Gemini: gemini/gemini-2.0-flash, gemini/gemini-2.5-flash, gemini/gemini-3-flash-preview
- Anthropic: anthropic/claude-sonnet-4-5-20250929, etc.
- OpenAI: openai/gpt-4.1, etc.
- Any LiteLLM-supported model: https://docs.litellm.ai/docs/providers

Set the appropriate API key env var:
- GEMINI_API_KEY for Google models
- ANTHROPIC_API_KEY for Claude models
- OPENAI_API_KEY for GPT models
"""

from __future__ import annotations

import logging
import time

from .base import BaseAdapter

logger = logging.getLogger(__name__)

# Default timeout for API requests (seconds)
_DEFAULT_TIMEOUT = 120.0

# Maximum backoff cap (seconds)
_MAX_BACKOFF = 60.0


class LiteLLMAdapter(BaseAdapter):
    """Universal adapter using LiteLLM for any supported model.

    The ``litellm`` package is imported lazily — construction succeeds
    even when the SDK is not installed (useful for testing).

    Model names follow LiteLLM conventions:
    - "gemini/gemini-2.0-flash" for Google Gemini
    - "anthropic/claude-sonnet-4-5-20250929" for Anthropic
    - "openai/gpt-4.1" for OpenAI
    - Or bare names that LiteLLM can auto-detect
    """

    def __init__(
        self,
        model: str,
        max_tokens: int = 65536,
        timeout: float = _DEFAULT_TIMEOUT,
        temperature: float = 0.0,
        **kwargs,
    ) -> None:
        super().__init__(model=model, **kwargs)
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.temperature = temperature

    def _call_model(self, messages: list[dict], system: str) -> str:
        """Call any LLM via LiteLLM's unified completion interface."""
        from litellm import completion

        full_messages = [{"role": "system", "content": system}] + messages

        # Reasoning models (o1/o3/gpt-5) don't support temperature
        kwargs: dict = dict(
            model=self.model,
            messages=full_messages,
            max_tokens=self.max_tokens,
            timeout=self.timeout,
        )
        model_lower = self.model.lower()
        is_reasoning = any(t in model_lower for t in ("o1", "o3", "gpt-5"))
        if not is_reasoning:
            kwargs["temperature"] = self.temperature

        response = completion(**kwargs)

        choice = response.choices[0]
        msg = choice.message
        content = msg.content

        # Warn if output was truncated by token limit
        if choice.finish_reason == "length":
            logger.warning(
                "Response truncated (finish_reason=length, model=%s, max_tokens=%d). "
                "Consider increasing max_tokens.",
                self.model,
                self.max_tokens,
            )

        # Thinking models (e.g. Gemini 2.5 Flash) may put output in
        # reasoning_content / thinking_blocks with content=None.
        if not content:
            reasoning = getattr(msg, "reasoning_content", None)
            if reasoning:
                logger.info(
                    "Using reasoning_content as content (thinking model, %d chars)",
                    len(reasoning),
                )
                content = reasoning

        if not content:
            thinking = getattr(msg, "thinking_blocks", None)
            if thinking:
                # thinking_blocks is a list of dicts with "text" keys
                parts = []
                for block in thinking:
                    if isinstance(block, dict):
                        parts.append(block.get("text", ""))
                    else:
                        parts.append(str(block))
                joined = "\n".join(parts)
                if joined.strip():
                    logger.info(
                        "Using thinking_blocks as content (%d blocks, %d chars)",
                        len(thinking),
                        len(joined),
                    )
                    content = joined

        if not content:
            logger.warning(
                "LiteLLM response had no content (finish_reason=%s, model=%s)",
                response.choices[0].finish_reason,
                self.model,
            )
            return ""
        return content

    def _call_with_retry(self, messages: list[dict], system: str) -> str:
        """Call model with retry logic for transient errors.

        Rate-limit errors (429) get longer waits (60s base) and more retries
        since they are per-minute quotas.
        """
        last_error: Exception | None = None
        max_attempts = self.max_retries

        for attempt in range(max_attempts):
            try:
                return self._call_model(messages, system)
            except Exception as e:
                last_error = e
                error_str = str(e).lower()

                # Don't retry auth errors or invalid requests
                if any(code in error_str for code in ("401", "403", "404", "invalid")):
                    raise

                is_rate_limit = "429" in error_str or "rate_limit" in error_str
                is_overload = "503" in error_str or "overload" in error_str or "unavailable" in error_str

                if is_rate_limit:
                    # Rate limits are per-minute — wait 60-90s
                    wait = 60.0 + attempt * 15.0
                    # Give more attempts for rate limits
                    if attempt == max_attempts - 1 and max_attempts < 8:
                        max_attempts = 8
                elif is_overload:
                    # Temporary overload — wait 30-90s, more retries
                    wait = 30.0 + attempt * 20.0
                    if attempt == max_attempts - 1 and max_attempts < 6:
                        max_attempts = 6
                else:
                    wait = min(self.retry_delay * (2 ** attempt), _MAX_BACKOFF)

                logger.warning(
                    "LiteLLM error (attempt %d/%d): %s — waiting %.1fs",
                    attempt + 1,
                    max_attempts,
                    str(e)[:200],
                    wait,
                )
                if attempt < max_attempts - 1:
                    time.sleep(wait)

        raise RuntimeError(
            f"LiteLLM call failed after {max_attempts} retries: {last_error}"
        ) from last_error
