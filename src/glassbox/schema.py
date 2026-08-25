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
import re
from typing import Literal

from pydantic import BaseModel, ValidationError

_FENCE = re.compile(r"```(?:json)?\s*(\{.*\})\s*```", re.DOTALL)


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
"reason": "why"}]}"""


def parse_case_file(text: str, question_id: str) -> CaseFile:
    fenced = _FENCE.search(text or "")
    payload_text = fenced.group(1) if fenced else (text or "")
    start, end = payload_text.find("{"), payload_text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"no JSON object found in model output: {(text or '')[:200]!r}")

    payload = json.loads(payload_text[start : end + 1])
    payload["question_id"] = question_id
    for key in ("issues", "rules", "findings"):
        if key in payload and not payload[key]:
            payload[key] = None
    try:
        return CaseFile(**payload)
    except ValidationError as exc:
        raise ValueError(f"invalid case file: {exc}") from exc
