"""The shared four-section answer schema.

Recipe 2 fills this in one call. The Phase 2 pipeline fills the same object stage
by stage. Because the schema is identical, the only difference between them is
whether it is completed in one context or four - which is precisely the
manipulation this study measures (spec section 6.3).

Values are natural-language legal reasoning inside a JSON envelope: not bare
keyword JSON, which flattens reasoning quality, and not free prose, which cannot
be scored or traced across stages.
"""

from __future__ import annotations

import json
from typing import Literal

from pydantic import BaseModel, ValidationError

# A model recording an amendment against the section it just wrote under the
# JSON key `findings` naturally echoes that field name, even though the
# amendment vocabulary calls that section `application`. Without this alias,
# that one-word mismatch fails the whole case file - and since self-correction
# rate is a measured quantity, it preferentially destroys exactly the samples
# where the pipeline did self-correct.
_AMENDMENT_SECTION_ALIASES = {"findings": "application"}
_AMENDMENT_SECTIONS = {"issues", "rules", "application"}


class Issue(BaseModel):
    id: str
    statement: str
    why_it_arises: str


class Rule(BaseModel):
    issue_id: str
    rule: str
    elements: list[str]


class ElementFinding(BaseModel):
    issue_id: str
    element: str
    holds: Literal["yes", "no", "uncertain"]
    reasoning: str


class Amendment(BaseModel):
    section: Literal["issues", "rules", "application"]
    change: str
    reason: str


class CaseFile(BaseModel):
    question_id: str
    issues: list[Issue] | None = None
    rules: list[Rule] | None = None
    findings: list[ElementFinding] | None = None
    conclusion: str | None = None
    final_answer: str | None = None
    amendments: list[Amendment] = []

    def to_sections(self) -> dict[str, str]:
        """Flatten to named prose blocks for grading and inspection."""
        sections: dict[str, str] = {}
        if self.issues:
            sections["issues"] = "\n".join(
                f"{i.statement} {i.why_it_arises}" for i in self.issues)
        if self.rules:
            sections["rules"] = "\n".join(
                f"{r.rule} Elements: {'; '.join(r.elements)}." for r in self.rules)
        if self.findings:
            sections["application"] = "\n".join(
                f"{f.element}: {f.holds}. {f.reasoning}" for f in self.findings)
        if self.conclusion:
            sections["conclusion"] = self.conclusion
        if self.final_answer:
            sections["final_answer"] = self.final_answer
        return sections


def case_file_json_instructions() -> str:
    return """Return only JSON, in exactly this form. Every value is natural-language \
legal writing, not keywords:

{"issues": [{"id": "i1", "statement": "the legal issue", "why_it_arises": "one sentence"}],
 "rules": [{"issue_id": "i1", "rule": "the governing rule, citing provisions where they \
exist", "elements": ["each required element, one per entry"]}],
 "findings": [{"issue_id": "i1", "element": "the element", "holds": "yes|no|uncertain", \
"reasoning": "why, citing the specific facts"}],
 "conclusion": "the overall conclusion",
 "final_answer": "the full exam-style answer, written out in prose",
 "amendments": [{"section": "issues|rules|application", "change": "what you changed", \
"reason": "why"}]}

Every "holds" value must be exactly one of the three literal words yes, no, or \
uncertain - nothing else. Never write a qualifier, parenthetical, or combination \
such as "partially", "yes (as to X)", "not applicable", or "no/uncertain": put any \
such nuance in "reasoning" instead, and choose "uncertain" whenever the element does \
not hold cleanly as yes or no. If an element has multiple parts that could get \
different answers, do not pack more than one answer into a single "holds" value \
(e.g. never write "a1: yes; a2: uncertain") - either give the element one overall \
answer (uncertain if the parts disagree) or split it into separate "findings" \
entries, each naming one sub-part in "element" and carrying its own single "holds".

Every entry in "rules" must include a non-empty "elements" array, even when the \
rule has only one element - never omit this field."""


def _balanced_json_objects(text: str) -> list[str]:
    """Every balanced top-level `{...}` span in `text`, in order of appearance.

    String-aware, so a brace inside a quoted value never mis-counts depth.
    Markdown code fences need no special handling: their backticks just sit
    outside any `{}` span and are skipped like any other prose character.
    """
    spans: list[str] = []
    depth = 0
    start = None
    in_string = False
    escape = False
    for i, ch in enumerate(text):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start is not None:
                spans.append(text[start : i + 1])
                start = None
    return spans


def _extract_payload(text: str) -> dict:
    """Pick the case-file JSON object out of raw model output.

    Prefers the *last* balanced object: a model that drafts then corrects
    itself, or echoes an example before the real payload, leaves the good
    object last. Earlier candidates are only a fallback for when the last
    balanced span is not valid JSON on its own (e.g. an incidental brace pair
    in prose, `"{...}"`, that closes before the real payload even starts).
    """
    candidates = _balanced_json_objects(text)
    last_error: json.JSONDecodeError | None = None
    for candidate in reversed(candidates):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError as exc:
            last_error = exc
    if last_error is not None:
        raise ValueError(
            f"could not parse any JSON object found in model output: {last_error}"
        ) from last_error
    raise ValueError(f"no JSON object found in model output: {text[:200]!r}")


def _normalise_amendments(raw: object) -> list[dict]:
    """Recover amendments recorded under an off-vocabulary section label,
    rather than failing the entire case file over one field.

    An unrecognised label drops just that one amendment; the rest of the case
    file - and every other amendment - survives.
    """
    cleaned: list[dict] = []
    for amendment in raw or []:
        if not isinstance(amendment, dict):
            continue
        section = amendment.get("section")
        section = _AMENDMENT_SECTION_ALIASES.get(section, section)
        if section not in _AMENDMENT_SECTIONS:
            continue
        cleaned.append({**amendment, "section": section})
    return cleaned


def parse_case_file(text: str, question_id: str) -> CaseFile:
    payload = _extract_payload(text or "")
    payload["question_id"] = question_id
    for key in ("issues", "rules", "findings"):
        if key in payload and not payload[key]:
            payload[key] = None
    payload["amendments"] = _normalise_amendments(payload.get("amendments"))
    try:
        return CaseFile(**payload)
    except ValidationError as exc:
        raise ValueError(f"invalid case file: {exc}") from exc
