"""Anthropic (Claude) model adapter with production-grade error handling.

Handles:
- Exponential backoff retry on transient errors (429, 500, 529)
- Rate limit handling with Retry-After header respect
- Request timeouts (don't hang forever)
- Graceful extraction of response text
"""

from __future__ import annotations

import logging
import time

from .base import BaseAdapter

logger = logging.getLogger(__name__)

# HTTP status codes that warrant retry
_RETRYABLE_STATUS_CODES = {429, 500, 529}

# Default timeout for API requests (seconds)
_DEFAULT_TIMEOUT = 120.0

# Maximum backoff cap (seconds)
_MAX_BACKOFF = 60.0


class AnthropicAdapter(BaseAdapter):
    """Adapter for Claude models via the Anthropic Messages API.

    The ``anthropic`` package is imported lazily -- construction succeeds
    even when the SDK is not installed (useful for testing).
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4-5-20250929",
        max_tokens: int = 4096,
        timeout: float = _DEFAULT_TIMEOUT,
        **kwargs,
    ) -> None:
        super().__init__(model=model, **kwargs)
        self.max_tokens = max_tokens
        self.timeout = timeout
        self._client = None

    @property
    def client(self):
        if self._client is None:
            import anthropic

            self._client = anthropic.Anthropic(timeout=self.timeout)
        return self._client

    def _call_model(self, messages: list[dict], system: str) -> str:
        """Call the Anthropic Messages API and return the response text."""
        import anthropic

        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=messages,
        )
        # Extract text from response content blocks
        text_parts = []
        for block in response.content:
            if hasattr(block, "text"):
                text_parts.append(block.text)
        if not text_parts:
            logger.warning(
                "Anthropic response had no text blocks (stop_reason=%s)",
                response.stop_reason,
            )
            return ""
        return "\n".join(text_parts)

    def _call_with_retry(self, messages: list[dict], system: str) -> str:
        """Call model with Anthropic-specific retry logic.

        Handles:
        - 429 (rate limited): respects Retry-After header
        - 500 (server error): exponential backoff
        - 529 (overloaded): exponential backoff
        - APITimeoutError: retry with backoff
        - APIConnectionError: retry with backoff
        """
        import anthropic

        last_error: Exception | None = None

        for attempt in range(self.max_retries):
            try:
                return self._call_model(messages, system)

            except anthropic.RateLimitError as e:
                last_error = e
                # Try to respect Retry-After header
                retry_after = _extract_retry_after(e)
                wait = retry_after if retry_after else self.retry_delay * (2**attempt)
                wait = min(wait, _MAX_BACKOFF)
                logger.warning(
                    "Rate limited (attempt %d/%d), waiting %.1fs",
                    attempt + 1,
                    self.max_retries,
                    wait,
                )
                if attempt < self.max_retries - 1:
                    time.sleep(wait)

            except anthropic.InternalServerError as e:
                last_error = e
                wait = min(self.retry_delay * (2**attempt), _MAX_BACKOFF)
                logger.warning(
                    "Server error %s (attempt %d/%d), waiting %.1fs",
                    e.status_code,
                    attempt + 1,
                    self.max_retries,
                    wait,
                )
                if attempt < self.max_retries - 1:
                    time.sleep(wait)

            except anthropic.APIStatusError as e:
                if e.status_code in _RETRYABLE_STATUS_CODES:
                    last_error = e
                    wait = min(self.retry_delay * (2**attempt), _MAX_BACKOFF)
                    logger.warning(
                        "API error %s (attempt %d/%d), waiting %.1fs",
                        e.status_code,
                        attempt + 1,
                        self.max_retries,
                        wait,
                    )
                    if attempt < self.max_retries - 1:
                        time.sleep(wait)
                else:
                    # Non-retryable status (400, 401, 403, 404, etc.)
                    raise

            except anthropic.APITimeoutError as e:
                last_error = e
                wait = min(self.retry_delay * (2**attempt), _MAX_BACKOFF)
                logger.warning(
                    "API timeout (attempt %d/%d), waiting %.1fs",
                    attempt + 1,
                    self.max_retries,
                    wait,
                )
                if attempt < self.max_retries - 1:
                    time.sleep(wait)

            except anthropic.APIConnectionError as e:
                last_error = e
                wait = min(self.retry_delay * (2**attempt), _MAX_BACKOFF)
                logger.warning(
                    "Connection error (attempt %d/%d), waiting %.1fs",
                    attempt + 1,
                    self.max_retries,
                    wait,
                )
                if attempt < self.max_retries - 1:
                    time.sleep(wait)

        raise RuntimeError(
            f"Anthropic API call failed after {self.max_retries} retries: {last_error}"
        ) from last_error


def _extract_retry_after(error: Exception) -> float | None:
    """Extract Retry-After value from an API error response."""
    # The anthropic SDK exposes response headers on the error
    try:
        response = getattr(error, "response", None)
        if response is not None:
            headers = getattr(response, "headers", {})
            retry_after = headers.get("retry-after")
            if retry_after:
                return float(retry_after)
    except (ValueError, TypeError, AttributeError):
        pass
    return None
