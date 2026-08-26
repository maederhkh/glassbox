"""Recipe 1: one call, one strong open legal-reasoning prompt."""

from __future__ import annotations

import time
from datetime import datetime, timezone

from glassbox.dataset import Question
from glassbox.recipes.base import RecipeResult, prompt_fingerprint

PLAIN_SYSTEM = (
    "You are an expert in {course}, answering a university law examination "
    "question. You address legal issues in a structured, exam-style manner."
)

PLAIN_PROMPT = """Answer the following law examination question.

Use precise legal language. Identify the legal issues raised, state the applicable \
rules and cite specific provisions where they exist, apply those rules to the facts \
given, and reach a reasoned conclusion.

Do not state disclaimers, do not suggest consulting a lawyer, and do not tell the \
reader to research the matter themselves. If the question requires material that has \
not been provided, say so explicitly rather than inventing it.

Answer in English.

Question:
{question}

Answer:"""

# Computed once at import time over the fixed templates only (course/question
# placeholders left unrendered) -- see prompt_fingerprint's docstring. Constant
# across every question and every PlainRecipe call.
PROMPT_HASH = prompt_fingerprint(PLAIN_SYSTEM, PLAIN_PROMPT)


class PlainRecipe:
    name = "plain"

    def run(self, question: Question, client) -> RecipeResult:
        started = time.monotonic()
        completion = client.complete(
            PLAIN_PROMPT.format(question=question.question),
            system=PLAIN_SYSTEM.format(course=question.course),
        )
        return RecipeResult(
            recipe=self.name,
            question_id=question.id,
            final_answer=completion.text,
            sections=None,
            stages=[],
            usage=completion.usage,
            seconds=time.monotonic() - started,
            metadata={
                "model": completion.model,
                "temperature": getattr(client, "temperature", None),
                "reasoning_effort": getattr(client, "reasoning_effort", None),
                "prompt_hash": PROMPT_HASH,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )
