"""Per-criterion sub-scores.

These are LEXam's own validated criteria - whether the answer identifies the legal
issues, recalls the applicable rules, and applies those rules to the facts - split
into separate numbers. They map onto the four pipeline stages, giving stage-level
attribution without inventing a rubric (spec section 7.1).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

SUBSCORE_PROMPT = """Assess a law examination answer against the examiner's reference \
answer on four criteria, each scored from 0.0 to 1.0:

  issue_spotting   - did it identify the legal issues the reference raises?
  rule_recall      - did it state the applicable rules correctly?
  rule_application - did it apply those rules to the facts, rather than restating law?
  conclusion       - is its conclusion supported by its own reasoning?

Return only JSON: {{"issue_spotting": 0.0, "rule_recall": 0.0, \
"rule_application": 0.0, "conclusion": 0.0}}

### Question
{question}

### Reference answer
{reference}

### Answer being assessed
{answer}
"""


@dataclass(frozen=True)
class SubScores:
    issue_spotting: float
    rule_recall: float
    rule_application: float
    conclusion: float


def _clamp(value) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def score_subscores(question: str, reference: str, answer_text: str, client) -> SubScores:
    completion = client.complete(
        SUBSCORE_PROMPT.format(question=question, reference=reference, answer=answer_text)
    )
    text = completion.text
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    payload_text = fenced.group(1) if fenced else text
    start, end = payload_text.find("{"), payload_text.rfind("}")
    payload = json.loads(payload_text[start : end + 1])

    return SubScores(
        issue_spotting=_clamp(payload.get("issue_spotting")),
        rule_recall=_clamp(payload.get("rule_recall")),
        rule_application=_clamp(payload.get("rule_application")),
        conclusion=_clamp(payload.get("conclusion")),
    )
