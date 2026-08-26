"""Recipes 2 and 3: one call, four sections.

Recipe 2 (structured) is the study's most important control. It receives the same
instructions and produces the same schema as the four-stage pipeline, in a single
call. If it matches the pipeline, decomposition's value was the instructions rather
than the separate calls - an honest negative result the design is built to detect
(spec section 5, hypothesis H2).

Recipe 3 (think_longer) is identical except the client is constructed with a raised
reasoning effort, so its token spend approaches the pipeline's without decomposing
the task. It answers the "your gain was just more compute" objection.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone

from glassbox.dataset import Question
from glassbox.recipes.base import RecipeResult, prompt_fingerprint
from glassbox.schema import case_file_json_instructions, parse_case_file

STRUCTURED_SYSTEM = (
    "You are an expert in {course}, answering a university law examination "
    "question. You address legal issues in a structured, exam-style manner."
)

STRUCTURED_PROMPT = """Answer the following law examination question by working through \
it in four steps.

1. Identify each legal issue the question raises, and say in one sentence why it arises.
2. For each issue, state the governing rule, citing specific provisions where they \
exist, and break that rule into its required elements.
3. Apply each element to the facts given, saying whether it holds and citing the \
specific facts that decide it.
4. State your conclusion, then write out the full exam-style answer.

Use precise legal language. Do not state disclaimers, do not suggest consulting a \
lawyer, and do not tell the reader to research the matter themselves. If the question \
requires material that has not been provided, say so explicitly rather than inventing \
it. Answer in English.

{json_instructions}

Question:
{question}
"""

# Computed once at import time over the fixed templates and the fully-rendered
# shared JSON instructions -- course/question placeholders left unrendered, see
# prompt_fingerprint's docstring. Constant across every question, and shared
# byte-for-byte between StructuredRecipe and ThinkLongerRecipe (the latter
# inherits run() unchanged), since they differ only in the client's
# reasoning_effort, never in the prompt. If case_file_json_instructions() is
# ever tightened again mid-experiment, this hash changes for every result
# produced afterward, making that change auditable from the data instead of
# from file mtimes.
PROMPT_HASH = prompt_fingerprint(
    STRUCTURED_SYSTEM, STRUCTURED_PROMPT, case_file_json_instructions()
)


class StructuredRecipe:
    name = "structured"
    # Exposed as a class attribute (not just used as a module-level constant
    # above) so runner.py's skip-existing resume can compare a stored
    # result's recorded prompt_hash against "this recipe's prompt, right
    # now" via getattr(recipe, "PROMPT_HASH", None). ThinkLongerRecipe
    # inherits this unchanged, matching its inherited run().
    PROMPT_HASH = PROMPT_HASH

    def run(self, question: Question, client) -> RecipeResult:
        started = time.monotonic()
        completion = client.complete(
            STRUCTURED_PROMPT.format(
                question=question.question,
                json_instructions=case_file_json_instructions(),
            ),
            system=STRUCTURED_SYSTEM.format(course=question.course),
        )

        metadata = {
            "model": completion.model,
            "temperature": getattr(client, "temperature", None),
            "reasoning_effort": getattr(client, "reasoning_effort", None),
            "prompt_hash": PROMPT_HASH,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "parse_failed": False,
        }
        try:
            case_file = parse_case_file(completion.text, question.id)
            sections = case_file.to_sections()
            # Leave this empty rather than substituting the conclusion. normalise()
            # prefers a non-empty final_answer and falls back to joining sections, so an
            # empty value routes a parse-succeeded-but-no-answer result to the sections
            # join. Substituting the conclusion alone would silently grade the recipe on
            # a fragment of what it produced.
            final_answer = case_file.final_answer or ""
        except ValueError:
            metadata["parse_failed"] = True
            sections, final_answer = None, completion.text

        return RecipeResult(
            recipe=self.name,
            question_id=question.id,
            final_answer=final_answer,
            sections=sections,
            stages=[],
            usage=completion.usage,
            seconds=time.monotonic() - started,
            metadata=metadata,
        )


class ThinkLongerRecipe(StructuredRecipe):
    """Recipe 2 with a raised reasoning effort, set on the client by the caller."""

    name = "think_longer"
