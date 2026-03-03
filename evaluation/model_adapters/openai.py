"""OpenAI (GPT) model adapter."""

from __future__ import annotations

from .base import BaseAdapter


class OpenAIAdapter(BaseAdapter):
    """Adapter for GPT models via the OpenAI Chat Completions API.

    The ``openai`` package is imported lazily — construction succeeds
    even when the SDK is not installed (useful for testing).
    """

    def __init__(
        self,
        model: str = "gpt-4.1",
        max_tokens: int = 4096,
        **kwargs,
    ) -> None:
        super().__init__(model=model, **kwargs)
        self.max_tokens = max_tokens
        self._client = None

    @property
    def client(self):
        if self._client is None:
            import openai
            self._client = openai.OpenAI()
        return self._client

    def _call_model(self, messages: list[dict], system: str) -> str:
        full_messages = [{"role": "system", "content": system}] + messages
        response = self.client.chat.completions.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=full_messages,
        )
        return response.choices[0].message.content
