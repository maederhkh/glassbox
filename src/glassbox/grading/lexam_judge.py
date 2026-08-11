"""LEXam's official open-question judge.

Scale is 0.0-1.0 in 0.1 increments, emitted as [[score]]. The leaderboard reports
this multiplied by 100. Several judges may be supplied; the ensemble score is the
minimum, matching LEXam's September 2025 protocol.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

SCORE_PATTERN = re.compile(r"\[\[(\d\.\d)\]\]")

LEXAM_JUDGE_SYSTEM = (
    "Act as a Judge specializing in the evaluation of law school exams. Your task "
    "is to assess how well the response aligns with the reference answer, with a "
    "focus on accuracy, completeness, and legal reasoning."
)

LEXAM_JUDGE_PROMPT = """You are given an exam question, a reference answer written by \
a legal expert, and a candidate response.

Provide:
1. A brief explanation of how well the candidate response conforms to the reference answer.
2. Constructive feedback, as a professor would give.
3. A final correctness score on a scale from 0.0 to 1.0, in increments of 0.1, where \
1.0 = complete fulfillment (100%) and lower scores reflect proportional shortfalls.

Strictly follow this format for the score: "[[score]]", e.g., "The correctness score: [[0.5]]"

### Question
{question}

### Reference answer
{reference}

### Candidate response
{candidate}
"""


@dataclass(frozen=True)
class JudgeVerdict:
    score: float | None
    per_judge: list[float | None]
    explanations: list[str]


def parse_score(text: str) -> float | None:
    """Last [[d.d]] in the text, clamped to [0, 1]. None if absent."""
    matches = SCORE_PATTERN.findall(text or "")
    if not matches:
        return None
    value = float(matches[-1])
    return value if 0.0 <= value <= 1.0 else 0.0


def judge_once(client, question: str, reference: str, candidate: str) -> tuple[float | None, str]:
    completion = client.complete(
        LEXAM_JUDGE_PROMPT.format(question=question, reference=reference, candidate=candidate),
        system=LEXAM_JUDGE_SYSTEM,
    )
    return parse_score(completion.text), completion.text


def ensemble_min(scores: list[float | None]) -> float | None:
    usable = [s for s in scores if s is not None]
    return min(usable) if usable else None


def judge_answer(clients: list, question: str, reference: str, candidate: str) -> JudgeVerdict:
    per_judge: list[float | None] = []
    explanations: list[str] = []
    for client in clients:
        score, text = judge_once(client, question, reference, candidate)
        per_judge.append(score)
        explanations.append(text)
    return JudgeVerdict(
        score=ensemble_min(per_judge), per_judge=per_judge, explanations=explanations
    )
