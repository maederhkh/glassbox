"""Recipe 4: the four-stage pipeline.

The recipe under study. One question becomes one model call per stage, each
call receiving a shared case file and filling in its own section.

Four properties make this decomposition rather than a longer prompt:

  * Each stage is asked for its own step and nothing else. The instructions come
    from glassbox.recipes.steps, the same source Recipe 2 draws its four-part
    prompt from, so the control cannot drift from the thing it controls.

  * A stage sees the facts plus every committed section written before it, but
    never an earlier stage's raw model output (spec 6.2). Passing raw text would
    hand the later stage the earlier one's deliberation, which is single-pass
    reasoning split across calls rather than decomposition.

  * A stage that answers a step which is not its own has that content
    DISCARDED and the lapse counted (spec 6.1). Keeping an early conclusion
    would let it reach the final stage, which would then echo it, and the
    decomposition would collapse without anything failing. How often the model
    cannot hold a partial legal analysis is itself a result.

  * A stage MAY correct an earlier section, but must declare it (spec 6.4). A
    declared amendment is not a violation and does update the earlier section;
    the same content with no declaration is a violation. That is what makes the
    amendment log a measurement of self-correction rather than decoration, and
    the pipeline's rescue rate is a quantity the study reports.

The recipe is intentionally absent from scripts/run_recipe.py's RECIPES until
its behaviour has been checked against the real model. A test asserts that
absence so it cannot be added by accident.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone

from glassbox.dataset import Question
from glassbox.recipes.base import RecipeResult, StageRecord, prompt_fingerprint
from glassbox.recipes.steps import (
    STEP_APPLICATION,
    STEP_CONCLUSION,
    STEP_ISSUES,
    STEP_RULES,
)
from glassbox.schema import (
    CaseFile,
    parse_case_file,
    stage_application_json_instructions,
    stage_conclusion_json_instructions,
    stage_issues_json_instructions,
    stage_rules_json_instructions,
)

PIPELINE_SYSTEM = (
    "You are an expert in {course}, answering a university law examination "
    "question. You address legal issues in a structured, exam-style manner."
)

#: Amendment.section names a rendered section; CaseFile stores findings under
#: its own field name. Declaring an amendment to "application" therefore
#: licenses writing "findings".
SECTION_TO_FIELD = {"issues": "issues", "rules": "rules", "application": "findings"}

#: Never counted as content a stage wrote: bookkeeping, not legal reasoning.
_NOT_CONTENT = ("question_id", "amendments")


@dataclass(frozen=True)
class Stage:
    """One call of the pipeline.

    ``fields`` are the CaseFile fields this stage is permitted to fill.
    Anything else it returns is out of turn.
    """

    name: str
    instruction: str
    json_instructions: str
    fields: tuple[str, ...]


STAGE_ISSUES = Stage(
    name="issues",
    instruction=STEP_ISSUES,
    json_instructions=stage_issues_json_instructions(),
    fields=("issues",),
)

STAGE_RULES = Stage(
    name="rules",
    instruction=STEP_RULES,
    json_instructions=stage_rules_json_instructions(),
    fields=("rules",),
)

# Named for what it does; its output renders under "application" for the grader.
STAGE_APPLICATION = Stage(
    name="application",
    instruction=STEP_APPLICATION,
    json_instructions=stage_application_json_instructions(),
    fields=("findings",),
)

STAGE_CONCLUSION = Stage(
    name="conclusion",
    instruction=STEP_CONCLUSION,
    json_instructions=stage_conclusion_json_instructions(),
    fields=("conclusion", "final_answer"),
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


def classify_output(stage: Stage, parsed: CaseFile) -> tuple[list[str], list[str]]:
    """Split a stage's output into fields it was allowed to fill, and the rest.

    A stage may fill its own fields, plus any field it declared an amendment
    for. Everything else is out of turn.
    """
    amended = {
        SECTION_TO_FIELD[a.section]
        for a in (parsed.amendments or [])
        if a.section in SECTION_TO_FIELD
    }
    allowed = set(stage.fields) | amended
    written = {
        field for field, value in parsed.model_dump().items()
        if value and field not in _NOT_CONTENT
    }
    return sorted(written & allowed), sorted(written - allowed)


def _merge(accumulated: CaseFile | None, kept: dict, question_id: str) -> CaseFile:
    """Fold a stage's permitted output into the case file built so far.

    Each stage returns only its own section, so replacing the case file with
    the latest parse would silently discard every earlier stage.
    """
    base = accumulated.model_dump() if accumulated is not None else {}
    base.update(kept)
    base["question_id"] = question_id
    return CaseFile(**base)


class PipelineRecipe:
    name = "pipeline"
    STAGES = (STAGE_ISSUES, STAGE_RULES, STAGE_APPLICATION, STAGE_CONCLUSION)

    #: Covers every stage's instruction and JSON shape, so tightening any one of
    #: them changes the hash on every result produced afterwards, exactly as it
    #: does for the single-call recipes.
    PROMPT_HASH = prompt_fingerprint(
        PIPELINE_SYSTEM,
        *(s.instruction for s in STAGES),
        *(s.json_instructions for s in STAGES),
    )

    def __init__(self, stages: tuple[Stage, ...] | None = None) -> None:
        # Injectable so a subset of stages can be exercised in isolation.
        self.stages = self.STAGES if stages is None else stages

    def run(self, question: Question, client) -> RecipeResult:
        started = time.monotonic()
        system = PIPELINE_SYSTEM.format(course=question.course)

        committed: dict[str, str] = {}
        records: list[StageRecord] = []
        violations: list[dict] = []
        truncated = False
        amendments: list[dict] = []
        case_file: CaseFile | None = None

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

            # Raises rather than degrading quietly: a stage that produced
            # nothing usable makes every later stage's input wrong, and a
            # half-run saved to disk is worse than none.
            truncated = truncated or completion.finish_reason == "length"

            parsed = parse_case_file(completion.text, question.id)

            allowed, out_of_turn = classify_output(stage, parsed)
            if out_of_turn:
                violations.append({"stage": stage.name, "wrote": out_of_turn})
            for amendment in parsed.amendments or []:
                amendments.append({
                    "stage": stage.name,
                    "section": amendment.section,
                    "change": amendment.change,
                    "reason": amendment.reason,
                })

            kept = {field: getattr(parsed, field) for field in allowed}
            case_file = _merge(case_file, kept, question.id)
            # The accumulated case file is the single source the relay reads
            # from (spec 6.2). A second parallel accumulator could disagree
            # with it, and would make the merge above unobservable.
            committed = case_file.to_sections()

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
                "model": client.model,
                "temperature": getattr(client, "temperature", None),
                "reasoning_effort": getattr(client, "reasoning_effort", None),
                "prompt_hash": self.PROMPT_HASH,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "stages_run": [s.name for s in self.stages],
                "truncated": truncated,
                "stage_violations": violations,
                "amendments": amendments,
            },
        )
