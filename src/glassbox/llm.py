"""Metered LLM client.

Every call records tokens and latency. Nothing else in the codebase talks to the
OpenAI SDK directly, so all cost accounting flows through one place.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential

from glassbox.config import (
    MAX_ATTEMPTS, OPENROUTER_BASE_URL, SYSTEM_MODEL, SYSTEM_TEMPERATURE,
)
from glassbox.usage import Usage


@dataclass(frozen=True)
class Completion:
    text: str
    usage: Usage
    model: str
    seconds: float


class LLMClient:
    def __init__(
        self,
        model: str = SYSTEM_MODEL,
        temperature: float | None = SYSTEM_TEMPERATURE,
        reasoning_effort: str | None = None,
    ) -> None:
        from openai import OpenAI

        load_dotenv()
        self.model = model
        self.temperature = temperature
        self.reasoning_effort = reasoning_effort
        self._client = OpenAI(
            base_url=OPENROUTER_BASE_URL,
            api_key=os.environ["OPENROUTER_API_KEY"],
        )

    @retry(stop=stop_after_attempt(MAX_ATTEMPTS),
           wait=wait_exponential(multiplier=2, min=2, max=60))
    def _call(self, messages: list[dict[str, str]]):
        kwargs: dict = {"model": self.model, "messages": messages}
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        if self.reasoning_effort is not None:
            kwargs["reasoning_effort"] = self.reasoning_effort
        return self._client.chat.completions.create(**kwargs)

    def complete(self, prompt: str, system: str | None = None) -> Completion:
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        started = time.monotonic()
        response = self._call(messages)
        seconds = time.monotonic() - started

        raw = response.usage
        details = getattr(raw, "completion_tokens_details", None)
        reasoning = getattr(details, "reasoning_tokens", 0) or 0
        visible = max((raw.completion_tokens or 0) - reasoning, 0)

        return Completion(
            text=response.choices[0].message.content or "",
            usage=Usage(
                input_tokens=raw.prompt_tokens or 0,
                output_tokens=visible,
                reasoning_tokens=reasoning,
                calls=1,
            ),
            model=self.model,
            seconds=seconds,
        )


@dataclass
class FakeLLMClient:
    """Test double. Returns queued responses and records what it was asked."""

    responses: list[str]
    model: str = "fake-model"
    # None, mirroring LLMClient. Recipes record getattr(client, "temperature", None)
    # into result metadata, so a default of 0.7 here would write a temperature into
    # test fixtures that the real client never sends.
    temperature: float | None = None
    reasoning_effort: str | None = None
    prompts: list[str] = field(default_factory=list)
    systems: list[str | None] = field(default_factory=list)
    _index: int = 0

    def complete(self, prompt: str, system: str | None = None) -> Completion:
        assert self._index < len(self.responses), (
            f"FakeLLMClient exhausted: {len(self.responses)} responses queued, "
            f"call {self._index + 1} requested"
        )
        text = self.responses[self._index]
        self._index += 1
        self.prompts.append(prompt)
        self.systems.append(system)
        return Completion(
            text=text,
            usage=Usage(
                input_tokens=len(prompt.split()),
                output_tokens=len(text.split()),
                reasoning_tokens=0,
                calls=1,
            ),
            model=self.model,
            seconds=0.0,
        )
