"""Decompose each expert reference answer into atomic checkable points.

Spec section 7.2: this is the primary diagnostic metric. Point-level near-binary
judgments are far more reliable than holistic 1-10 ratings, and the same point list
is traced through every pipeline stage to measure error propagation.

No legal content is invented here. The reference answer is only split into its parts.
Every drafted checklist is verified by hand before use.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from glassbox.dataset import Question

BUILD_PROMPT = """Below is a law examination question and the reference answer written \
by the examiner.

Split the reference answer into its atomic checkable points: the separately markable \
propositions a candidate would have to state to earn full marks. Use a grading test, \
not a sentence-counting one: a point is something a grader would award a distinct mark \
for. If splitting a statement into two would not earn a candidate two separate marks, \
it is one point, not two. A rule stated together with its application to the facts is \
normally one point, not two. A single argument elaborated across several sentences, or \
illustrated with an example or a supporting detail, is normally one point, not one per \
sentence or clause. Do not add anything the reference answer does not contain, and do \
not merge two genuinely distinct, separately markable propositions into one point.

Reference answers take one of three forms. Identify which one this is and report it \
as "source":
- "model_answer": prose that answers the question directly.
- "marking_scheme": addressed to the grader, e.g. "the answer should focus on...", \
"at a minimum it should raise...". Each required element is close to a point already, \
but still apply the one-mark-per-point test above -- a marking scheme can list several \
clauses of elaboration under a single required element, and those clauses are one point.
- "mark_annotated": a model answer where per-item mark values survived text \
extraction as bare digits scattered mid-prose and at paragraph ends (for example \
"...unsatisfying. 1 The 'legal families' proposal..." or a lone "2" on its own \
line between paragraphs). Treat this the same as a model answer for extracting \
propositions, but the mark values are exactly the calibration you need for \
granularity: they are the number of marks the examiner actually allocated to that \
stretch of text, so let each value bound how many points you draw from the text it is \
attached to -- a stretch marked "1" is one point, not three. Completely ignore the \
digits themselves as content. A digit sitting alone, or a number glued onto the end or \
start of a sentence with no grammatical role, is a mark value, never a proposition, and \
must never become a point of its own or be folded into a point's text.

Return only JSON, in exactly this form:
{{"source": "model_answer" or "marking_scheme" or "mark_annotated",
  "points": [{{"text": "..."}}, {{"text": "..."}}]}}

### Question
{question}

### Reference answer
{reference}
"""

# A point whose entire text is digits/whitespace/punctuation is never a legal
# proposition -- it is a mark value that leaked out of extraction (see
# BUILD_PROMPT's "mark_annotated" case). Dropped defensively even though the
# prompt already tells the model to ignore these, because a single such point
# reaching data/checklists/ would silently corrupt every score computed from it.
_STRAY_DIGIT_POINT = re.compile(r"^[\d\s.,;:()\-]*$")


@dataclass(frozen=True)
class ChecklistPoint:
    id: str
    text: str


@dataclass(frozen=True)
class Checklist:
    question_id: str
    source: str
    points: list[ChecklistPoint]


def _extract_json(text: str) -> dict:
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    payload = fenced.group(1) if fenced else text
    start, end = payload.find("{"), payload.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"no JSON object found in judge output: {text[:200]!r}")
    return json.loads(payload[start : end + 1])


def build_checklist(question: Question, client) -> Checklist:
    completion = client.complete(
        BUILD_PROMPT.format(question=question.question, reference=question.answer)
    )
    payload = _extract_json(completion.text)
    texts = [p["text"].strip() for p in payload["points"]]
    texts = [t for t in texts if t and not _STRAY_DIGIT_POINT.match(t)]
    points = [
        ChecklistPoint(id=f"{question.id}-p{i}", text=t) for i, t in enumerate(texts, 1)
    ]
    return Checklist(
        question_id=question.id, source=payload.get("source", "unknown"), points=points
    )


def save_checklist(checklist: Checklist, directory: Path) -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{checklist.question_id}.json"
    path.write_text(json.dumps(asdict(checklist), indent=2, ensure_ascii=False),
                    encoding="utf-8")
    return path


def load_checklist(question_id: str, directory: Path) -> Checklist:
    payload = json.loads(
        (Path(directory) / f"{question_id}.json").read_text(encoding="utf-8")
    )
    return Checklist(
        question_id=payload["question_id"],
        source=payload["source"],
        points=[ChecklistPoint(**p) for p in payload["points"]],
    )


def checklist_to_markdown(checklist: Checklist, question: Question) -> str:
    lines = [
        f"# {checklist.question_id}",
        "",
        f"**Course:** {question.course} · **Area:** {question.area} · "
        f"**Reference type:** {checklist.source}",
        "",
        "Correct any point that misreads the reference answer. Delete points the "
        "reference does not actually make. Add points it makes that are missing. "
        "Keep one proposition per line.",
        "",
        "## Points",
        "",
    ]
    lines += [f"{i}. {p.text}" for i, p in enumerate(checklist.points, 1)]
    lines += ["", "## Question", "", question.question, "",
              "## Reference answer", "", question.answer, ""]
    return "\n".join(lines)


SCORE_PROMPT = """You are marking a law examination answer against a list of points \
taken from the examiner's reference answer.

For each point, decide:
  "covered" - the answer makes this point, in substance. Different wording is fine.
  "partial" - the answer gestures at it but is incomplete or imprecise.
  "missed"  - the answer does not make this point.

Judge substance, not style or length. A brief answer that makes the point is "covered".

Separately, list:
  contradictions - statements in the answer that contradict the reference points.
  inventions     - legal rules, cases, statutes or provisions the answer cites that do \
not appear in the reference points and that you have reason to doubt.

Return only JSON, in exactly this form:
{{"verdicts": [{{"point_id": "...", "coverage": "covered|partial|missed", "evidence": "..."}}],
  "contradictions": ["..."], "inventions": ["..."]}}

### Points
{points}

### Answer
{answer}
"""

_COVERAGE_WEIGHT = {"covered": 1.0, "partial": 0.5, "missed": 0.0}


@dataclass(frozen=True)
class PointVerdict:
    point_id: str
    coverage: str
    evidence: str


@dataclass(frozen=True)
class ChecklistVerdict:
    question_id: str
    verdicts: list[PointVerdict]
    contradictions: list[str]
    inventions: list[str]


def score_checklist(checklist: Checklist, answer_text: str, client) -> ChecklistVerdict:
    rendered = "\n".join(f"- {p.id}: {p.text}" for p in checklist.points)
    completion = client.complete(
        SCORE_PROMPT.format(points=rendered, answer=answer_text)
    )
    payload = _extract_json(completion.text)

    returned = {
        v["point_id"]: v for v in payload.get("verdicts", []) if "point_id" in v
    }
    verdicts = []
    for point in checklist.points:
        found = returned.get(point.id)
        coverage = (found or {}).get("coverage", "missed")
        if coverage not in _COVERAGE_WEIGHT:
            coverage = "missed"
        verdicts.append(
            PointVerdict(
                point_id=point.id,
                coverage=coverage,
                evidence=(found or {}).get("evidence", ""),
            )
        )

    return ChecklistVerdict(
        question_id=checklist.question_id,
        verdicts=verdicts,
        contradictions=list(payload.get("contradictions", [])),
        inventions=list(payload.get("inventions", [])),
    )


def coverage_fraction(verdict: ChecklistVerdict) -> float:
    if not verdict.verdicts:
        return 0.0
    return sum(_COVERAGE_WEIGHT[v.coverage] for v in verdict.verdicts) / len(verdict.verdicts)
