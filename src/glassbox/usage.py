"""Token and cost accounting.

Reasoning tokens are billed as output tokens, and are tracked separately so the
cost-versus-quality curve in spec section 9 can distinguish visible output from
hidden thinking.
"""

from __future__ import annotations

from dataclasses import dataclass

from glassbox.config import PRICING_PER_MTOK


@dataclass(frozen=True)
class Usage:
    input_tokens: int
    output_tokens: int
    reasoning_tokens: int
    calls: int

    @classmethod
    def zero(cls) -> "Usage":
        return cls(input_tokens=0, output_tokens=0, reasoning_tokens=0, calls=0)

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            reasoning_tokens=self.reasoning_tokens + other.reasoning_tokens,
            calls=self.calls + other.calls,
        )

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens + self.reasoning_tokens


def cost_usd(usage: Usage, model: str) -> float | None:
    """USD cost, or None when the model has no recorded price."""
    price = PRICING_PER_MTOK.get(model)
    if price is None:
        return None
    billed_output = usage.output_tokens + usage.reasoning_tokens
    return (
        usage.input_tokens * price["input"] + billed_output * price["output"]
    ) / 1_000_000
