"""Recipe 4: the four-stage pipeline. Stage 1 only, so far.

The recipe under study. One question becomes one model call per stage, each
call receiving a shared case file and filling in its own section.

Two properties make it an experiment rather than just a longer prompt, and both
live in run() below:

  * Each stage is asked for its own step and nothing else. The instruction comes
    from glassbox.recipes.steps, the same source Recipe 2 draws its four-part
    prompt from, so the two cannot drift apart and stop being comparable.

  * A stage sees the facts plus every committed section written before it, but
    never an earlier stage's raw model output (spec 6.2). Passing raw text would
    hand the later stage the earlier one's deliberation, which is closer to
    single-pass reasoning split across calls than to decomposition.

Stages 2-4 are not implemented, so this recipe is intentionally absent from
scripts/run_recipe.py's RECIPES. A one-stage result saved into a run directory
would be indistinguishable from a pipeline answer and would be graded as one.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone

from glassbox.dataset import Question
from glassbox.recipes.base import RecipeResult, StageRecord, prompt_fingerprint
from glassbox.recipes.steps import STEP_ISSUES
from glassbox.schema import parse_case_file, stage_issues_json_instructions

PIPELINE_SYSTEM = (
    "You are an expert in {course}, answering a university law examination "
    "question. You address legal issues in a structured, exam-style manner."
)


@dataclass(frozen=True)
class Stage:
    """One call of the pipeline.

    ``section`` names the CaseFile field this stage fills, which is also what
    later stages receive through the relay.
    """

    name: str
    instruction: str
    json_instructions: str
    section: str


STAGE_ISSUES = Stage(
    name="issues",
    instruction=STEP_ISSUES,
    json_instructions=stage_issues_json_instructions(),
    section="issues",
)


def build_stage_prompt(stage: Stage, question: Question, committed: dict[str, str]) -> str:
    """The facts, then whatever earlier stages committed, then this stage's step."""
    parts = ["Question:", question.question, ""]
    if committed:
        parts.append("Work already completed on this question:")
        parts.append("")
        for name, text in committed.items():
            parts.append(f"{name}:")
            parts.append(text)
            parts.append("")
    parts.append("Your task now:")
    parts.append(stage.instruction)
    parts.append("")
    parts.append(stage.json_instructions)
    return "\n".join(parts)


class PipelineRecipe:
    name = "pipeline"
    STAGES = (STAGE_ISSUES,)

    #: Covers every stage's instruction and JSON shape, so tightening any one of
    #: them changes the hash on every result produced afterwards, exactly as it
    #: does for the single-call recipes.
    PROMPT_HASH = prompt_fingerprint(
        PIPELINE_SYSTEM,
        *(s.instruction for s in STAGES),
        *(s.json_instructions for s in STAGES),
    )

    def __init__(self, stages: tuple[Stage, ...] | None = None) -> None:
        # Injectable so the relay can be exercised with a synthetic second
        # stage before the real Stage 2 exists.
        self.stages = self.STAGES if stages is None else stages

    def run(self, question: Question, client) -> RecipeResult:
        started = time.monotonic()
        system = PIPELINE_SYSTEM.format(course=question.course)

        committed: dict[str, str] = {}
        records: list[StageRecord] = []
        case_file = None

        for stage in self.stages:
            prompt = build_stage_prompt(stage, question, committed)
            stage_started = time.monotonic()
            completion = client.complete(prompt, system=system)
            records.append(StageRecord(
                name=stage.name,
                prompt=prompt,
                output=completion.text,
                usage=completion.usage,
                seconds=time.monotonic() - stage_started,
            ))

            # Raises on unparseable output rather than degrading quietly: a
            # stage that produced nothing usable makes every later stage's
            # input wrong, and a half-run saved to disk is worse than none.
            case_file = parse_case_file(completion.text, question.id)
            rendered = case_file.to_sections()
            if stage.section in rendered:
                committed[stage.section] = rendered[stage.section]

        total = records[0].usage
        for record in records[1:]:
            total = total + record.usage

        return RecipeResult(
            recipe=self.name,
            question_id=question.id,
            # Empty until the final stage writes one. normalise() prefers a
            # non-empty final_answer and falls back to joining sections, so a
            # partial run is never passed off as an answer.
            final_answer=(case_file.final_answer or "") if case_file else "",
            sections=dict(committed),
            stages=records,
            usage=total,
            seconds=time.monotonic() - started,
            metadata={
                "model": records[0].usage and client.model,
                "temperature": getattr(client, "temperature", None),
                "reasoning_effort": getattr(client, "reasoning_effort", None),
                "prompt_hash": self.PROMPT_HASH,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "stages_run": [s.name for s in self.stages],
            },
        )
