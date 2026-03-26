"""Reusable OpenAI client wrapper for grounded generation."""

from __future__ import annotations

import json
from typing import Any

from openai import APIError, AsyncOpenAI, RateLimitError

from core.config import Settings
from core.utils import LLMProviderError, async_retry, get_logger


logger = get_logger(__name__)


class OpenAILLMClient:
    """Thin wrapper around the OpenAI async client."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.client = AsyncOpenAI(api_key=settings.openai_api_key)

    async def generate_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float | None = None,
    ) -> dict[str, Any]:
        """Generate a JSON object from the LLM."""

        async def _request() -> dict[str, Any]:
            response = await self.client.chat.completions.create(
                model=self.settings.openai_model,
                temperature=temperature if temperature is not None else self.settings.llm_temperature,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )

            content = response.choices[0].message.content
            if not content:
                raise LLMProviderError("OpenAI returned an empty response.")

            try:
                return json.loads(content)
            except json.JSONDecodeError as exc:
                logger.exception("Failed to decode model JSON response.")
                raise LLMProviderError("OpenAI returned invalid JSON.") from exc

        try:
            return await async_retry(
                _request,
                retries=2,
                base_delay_seconds=1.0,
                retriable_exceptions=(APIError, RateLimitError, LLMProviderError),
            )
        except LLMProviderError:
            raise
        except Exception as exc:
            logger.exception("OpenAI request failed.")
            raise LLMProviderError("Failed to generate response from OpenAI.") from exc
