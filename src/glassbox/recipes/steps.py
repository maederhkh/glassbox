"""The four legal-reasoning steps, in one place.

Recipe 2 (structured) delivers all four in a single call. The Phase 2 pipeline
will deliver them one per call, one stage each. Recipe 2 is only a valid control
for the pipeline if both ask for exactly the same four things, so the wording
lives here and nowhere else: two copies would eventually drift, and the study
would then be comparing two variables instead of one.

This has already happened once on a neighbouring prompt. The shared JSON
instructions were tightened partway through a run, leaving 20 answers on the old
wording and 2 on the new, detectable only from file timestamps. That is the
failure this module exists to make impossible.

Each constant is a single instruction carrying no numbering of its own, so a
stage can be handed one directly. `numbered_steps()` renders them as the
numbered list a single-call recipe sends.
"""

from __future__ import annotations

STEP_ISSUES = (
    "Identify each legal issue the question raises, and say in one sentence "
    "why it arises."
)

STEP_RULES = (
    "For each issue, state the governing rule, citing specific provisions "
    "where they exist, and break that rule into its required elements."
)

STEP_APPLICATION = (
    "Apply each element to the facts given, saying whether it holds and citing "
    "the specific facts that decide it."
)

STEP_CONCLUSION = (
    "State your conclusion, then write out the full exam-style answer."
)

#: Pipeline order. Stage N of the pipeline receives ORDERED_STEPS[N - 1].
ORDERED_STEPS = (STEP_ISSUES, STEP_RULES, STEP_APPLICATION, STEP_CONCLUSION)


def numbered_steps() -> str:
    """The steps as the numbered list a single-call recipe asks for."""
    return "\n".join(f"{i}. {step}" for i, step in enumerate(ORDERED_STEPS, 1))
