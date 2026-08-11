# glassbox Phase 1 — Harness and Grader Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the measuring instrument and the three single-call recipes, so that by the end of Day 6 you can run Plain, Structured and Think-longer across all 20 development questions and get a full score table — before a single line of the multi-stage pipeline exists.

**Architecture:** A thin metered LLM client wraps the OpenAI API and records tokens, cost and latency for every call. Recipes are interchangeable objects that turn a `Question` into a `RecipeResult`; a runner executes any recipe over any question set and persists results with full provenance. Grading is three independent layers over the persisted results — LEXam's official judge, per-criterion sub-scores, and point-level checklist coverage — all operating on condition-stripped text so no grader can tell which recipe produced an answer.

**Tech Stack:** Python 3.12, uv, pydantic v2 (schemas and validation), openai (model and judge calls), pandas + pyarrow + huggingface_hub (dataset), tenacity (retries), pytest (tests). No network access in any unit test.

## Global Constraints

- **All model access goes through OpenRouter**, using the OpenAI SDK with `base_url="https://openrouter.ai/api/v1"` and `OPENROUTER_API_KEY`. Model ids are namespaced (`openai/gpt-5-mini`, not `gpt-5-mini`). Nothing talks to `api.openai.com`.
- Primary system model: **`openai/gpt-5-mini`**, the same model for every recipe and every stage.
- **Temperature is not supported by this model.** Task 1 established it: `temperature` is absent from `supported_parameters` on all four upstream routes and `default_parameters.temperature` is `null`. The call does not error — the parameter is silently dropped. The client must therefore **not send it**, and `SYSTEM_TEMPERATURE` is `None`. Sampling variation across repeated runs comes from the model's own default sampling; that is what the reliability measurement observes, and it must be described that way rather than as a chosen temperature. `seed` **is** supported and must be left unset, or repeated runs would be identical and the reliability measure would read zero by construction.
- Judges: LEXam's full three-judge ensemble — **`openai/gpt-4o`, `deepseek/deepseek-chat`, `qwen/qwen3-32b`** — scored as the **minimum** of the three, matching their September 2025 protocol. Model slugs are unverified guesses until Task 1's probe confirms them; a wrong slug must surface in Task 1, not in Task 6.
- LEXam judge scale is **0.0–1.0 in 0.1 increments**, emitted as `[[0.7]]`, parsed with `r"\[\[(\d\.\d)\]\]"`, clamped to `[0, 1]`, `None` if unparseable.
- Question set for all Phase 1 work: **`data/dev_20.json`** (dev split). The `test` split is never touched in Phase 1.
- Inclusion criterion, already frozen in `scripts/select_questions.py`: English, answer ≥ 150 words, question ≥ 50 words, `area != "Interdisciplinary"`.
- Only ids and metadata are versioned. LEXam question and answer text stays under `data/cache/` (gitignored). **Never commit question or reference-answer text.**
- Every persisted run records: model, temperature, reasoning effort, prompt hash, dataset revision, seed, timestamp, and full token usage.
- No unit test may make a network call. Tests use `FakeLLMClient`.
- Package lives at `src/glassbox/`, installed in editable mode. Scripts in `scripts/` are thin CLI wrappers over library code.

---

## File Structure

| Path | Responsibility |
|---|---|
| `pyproject.toml` | Package metadata, dependencies, pytest config |
| `.env.example` | Documents required environment variables |
| `src/glassbox/config.py` | Settings: models, temperature, pricing table, paths |
| `src/glassbox/usage.py` | `Usage` record, addition, cost computation |
| `src/glassbox/llm.py` | `LLMClient` (real) and `FakeLLMClient` (tests), retries, metering |
| `src/glassbox/dataset.py` | `Question`, loads frozen sample plus cached text |
| `src/glassbox/schema.py` | Pydantic models: `CaseFile`, `Issue`, `Rule`, `ElementFinding`, `Amendment` |
| `src/glassbox/recipes/base.py` | `Recipe` protocol, `RecipeResult`, `StageRecord` |
| `src/glassbox/recipes/plain.py` | Recipe 1 |
| `src/glassbox/recipes/structured.py` | Recipes 2 and 3 (same code, different reasoning effort) |
| `src/glassbox/storage.py` | Run persistence and loading, provenance |
| `src/glassbox/runner.py` | Execute a recipe over a question set |
| `src/glassbox/grading/lexam_judge.py` | LEXam prompts, score parsing, min-ensemble |
| `src/glassbox/grading/normalise.py` | Blind, condition-stripped answer text |
| `src/glassbox/grading/subscores.py` | Per-criterion sub-scores |
| `src/glassbox/grading/checklist.py` | Checklist build and point-level scoring |
| `scripts/probe_api.py` | Day 1 only: records which API parameters work |
| `scripts/run_recipe.py` | CLI: run a recipe over a question set |
| `scripts/calibrate.py` | CLI: the Day 3 calibration gate |
| `scripts/build_checklists.py` | CLI: draft checklists for human verification |
| `scripts/grade_runs.py` | CLI: score persisted runs |
| `tests/` | Mirrors `src/glassbox/` |

---

# Day 1 — Metered model client

**End state:** you can ask GPT-5-mini one development question from the terminal and see the answer, the token counts, the cost and the latency.

### Task 1: Probe the API surface

GPT-5-family parameter names differ from older models and must be verified against the live API rather than assumed. This task produces a recorded fact, not a guess.

**Files:**
- Create: `scripts/probe_api.py`
- Create: `.env.example`
- Create: `pyproject.toml`
- Create: `docs/api-surface.md`

**Interfaces:**
- Consumes: nothing
- Produces: `docs/api-surface.md`, recording the working call shape that Task 2 implements against

- [ ] **Step 1: Create `pyproject.toml`**

```toml
[project]
name = "glassbox"
version = "0.1.0"
description = "Does explicit multi-stage decomposition improve LLM legal reasoning?"
requires-python = ">=3.12"
dependencies = [
    "openai>=1.60",
    "pydantic>=2.9",
    "pandas>=2.2",
    "pyarrow>=17",
    "huggingface-hub>=0.26",
    "python-dotenv>=1.0",
    "tenacity>=9.0",
]

[project.optional-dependencies]
dev = ["pytest>=8.3"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/glassbox"]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["src", "."]
```

`pythonpath` includes `.` as well as `src` because `tests/test_calibrate.py` imports from
`scripts/`, which lives at the repository root rather than inside the package.

- [ ] **Step 2: Create `.env.example`**

```
OPENROUTER_API_KEY=sk-or-v1-your-key-here
```

- [ ] **Step 3: Install**

Run: `uv venv && uv pip install -e ".[dev]"`
Expected: succeeds, `glassbox` installed in editable mode.

Then copy `.env.example` to `.env` and paste your real key into it. `.env` is already gitignored.

- [ ] **Step 4: Write the probe script**

```python
"""Record what OpenRouter actually accepts for the models this study uses.

Run once, on Day 1. Two unknowns are settled here: which call parameters
gpt-5-mini honours, and whether every model slug in the study resolves at all.
A wrong judge slug discovered on Day 3 wastes a day; discovered here it costs
nothing. The result is written to docs/api-surface.md and the rest of the
codebase is built against whatever this reports as working.
"""

import json
import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
)
MODEL = "openai/gpt-5-mini"
JUDGE_SLUGS = ["openai/gpt-4o", "deepseek/deepseek-chat", "qwen/qwen3-32b"]
QUESTION = "In one sentence, what is the doctrine of consideration in contract law?"

results = []


def attempt(label: str, fn):
    try:
        r = fn()
        usage = getattr(r, "usage", None)
        results.append({
            "attempt": label,
            "ok": True,
            "usage": usage.model_dump() if usage else None,
        })
        print(f"OK   {label}")
    except Exception as exc:
        results.append({"attempt": label, "ok": False, "error": f"{type(exc).__name__}: {exc}"})
        print(f"FAIL {label}: {type(exc).__name__}: {exc}")


attempt("chat: plain", lambda: client.chat.completions.create(
    model=MODEL, messages=[{"role": "user", "content": QUESTION}]))

attempt("chat: temperature=0.7", lambda: client.chat.completions.create(
    model=MODEL, messages=[{"role": "user", "content": QUESTION}], temperature=0.7))

attempt("chat: max_completion_tokens=512", lambda: client.chat.completions.create(
    model=MODEL, messages=[{"role": "user", "content": QUESTION}],
    max_completion_tokens=512))

attempt("chat: reasoning_effort=low", lambda: client.chat.completions.create(
    model=MODEL, messages=[{"role": "user", "content": QUESTION}],
    reasoning_effort="low"))

attempt("chat: reasoning_effort=high", lambda: client.chat.completions.create(
    model=MODEL, messages=[{"role": "user", "content": QUESTION}],
    reasoning_effort="high"))

attempt("chat: extra_body reasoning effort=low", lambda: client.chat.completions.create(
    model=MODEL, messages=[{"role": "user", "content": QUESTION}],
    extra_body={"reasoning": {"effort": "low"}}))

attempt("chat: extra_body reasoning effort=high", lambda: client.chat.completions.create(
    model=MODEL, messages=[{"role": "user", "content": QUESTION}],
    extra_body={"reasoning": {"effort": "high"}}))

for slug in JUDGE_SLUGS:
    attempt(f"slug resolves: {slug}", lambda s=slug: client.chat.completions.create(
        model=s, messages=[{"role": "user", "content": "Reply with the digit 1."}]))

print("\n" + json.dumps(results, indent=2))

lines = ["# API surface", "",
         f"Probed model: `{MODEL}`. Regenerate with `python scripts/probe_api.py`.", "",
         "| attempt | works | notes |", "|---|---|---|"]
for r in results:
    note = "" if r["ok"] else r["error"].replace("|", "/")[:160]
    lines.append(f"| `{r['attempt']}` | {'yes' if r['ok'] else 'no'} | {note} |")
lines += ["", "```json", json.dumps(results, indent=2), "```", ""]

with open("docs/api-surface.md", "w", encoding="utf-8") as fh:
    fh.write("\n".join(lines))
print("\nwrote docs/api-surface.md")
```

- [ ] **Step 5: Run the probe**

Run: `python scripts/probe_api.py`
Expected: a table of OK/FAIL lines, and `docs/api-surface.md` written.

**Read the output before continuing.** Four things must be settled:
1. Whether `temperature` is accepted. If gpt-5-mini rejects it, **record that**, set temperature to the model default everywhere, and note in the spec's limitations that sampling variation comes from the model default rather than a chosen temperature. Do not silently drop it.
2. How reasoning effort is passed — top-level `reasoning_effort`, `extra_body={"reasoning": {...}}`, or neither. Recipe 3 needs at least two distinct levels that measurably change token spend. If no form works, Recipe 3 falls back to the approximate length-based variant already flagged in spec §5, and that must be recorded.
3. **Whether all four model slugs resolve.** The three judge slugs are educated guesses. Any that fails must be corrected here — check OpenRouter's model list for the right slug and re-run the probe. Do not leave a broken slug for Task 6 to discover.
4. Whether `reasoning_tokens` appears in the usage payload. OpenRouter's usage shape can differ from OpenAI's; Task 2's accounting depends on knowing where reasoning tokens live, or that they are absent.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml .env.example scripts/probe_api.py docs/api-surface.md
git commit -m "chore: project scaffolding and recorded GPT-5-mini API surface"
```

---

### Task 2: Usage accounting and the metered client

**Files:**
- Create: `src/glassbox/__init__.py`, `src/glassbox/config.py`, `src/glassbox/usage.py`, `src/glassbox/llm.py`
- Create: `tests/test_usage.py`, `tests/test_llm.py`

**Interfaces:**
- Consumes: `docs/api-surface.md` from Task 1
- Produces:
  - `Usage(input_tokens, output_tokens, reasoning_tokens, calls)`, supports `+`, `Usage.zero()`
  - `cost_usd(usage: Usage, model: str) -> float | None`
  - `Completion(text: str, usage: Usage, model: str, seconds: float)`
  - `LLMClient(model, temperature, reasoning_effort=None).complete(prompt, system=None) -> Completion`
  - `FakeLLMClient(responses: list[str])` with the same `.complete()` signature and a `.prompts` list recording every prompt received

- [ ] **Step 1: Write the failing tests**

`tests/test_usage.py`:

```python
from glassbox.usage import Usage, cost_usd


def test_usage_adds_componentwise():
    a = Usage(input_tokens=10, output_tokens=5, reasoning_tokens=2, calls=1)
    b = Usage(input_tokens=3, output_tokens=7, reasoning_tokens=0, calls=1)
    total = a + b
    assert total == Usage(input_tokens=13, output_tokens=12, reasoning_tokens=2, calls=2)


def test_usage_zero_is_additive_identity():
    a = Usage(input_tokens=10, output_tokens=5, reasoning_tokens=2, calls=1)
    assert Usage.zero() + a == a


def test_cost_is_none_for_unpriced_model():
    usage = Usage(input_tokens=1000, output_tokens=1000, reasoning_tokens=0, calls=1)
    assert cost_usd(usage, "some-unpriced-model") is None


def test_cost_counts_reasoning_tokens_as_output():
    usage = Usage(input_tokens=0, output_tokens=1_000_000, reasoning_tokens=1_000_000, calls=1)
    priced = cost_usd(usage, "gpt-5-mini")
    unpriced_half = cost_usd(
        Usage(input_tokens=0, output_tokens=1_000_000, reasoning_tokens=0, calls=1), "gpt-5-mini")
    assert priced is not None and unpriced_half is not None
    assert priced == 2 * unpriced_half
```

`tests/test_llm.py`:

```python
import pytest

from glassbox.llm import FakeLLMClient
from glassbox.usage import Usage


def test_fake_client_returns_queued_responses_in_order():
    client = FakeLLMClient(["first", "second"])
    assert client.complete("a").text == "first"
    assert client.complete("b").text == "second"


def test_fake_client_records_prompts_and_systems():
    client = FakeLLMClient(["x"])
    client.complete("the prompt", system="the system")
    assert client.prompts == ["the prompt"]
    assert client.systems == ["the system"]


def test_fake_client_reports_usage():
    client = FakeLLMClient(["hello"])
    completion = client.complete("a")
    assert isinstance(completion.usage, Usage)
    assert completion.usage.calls == 1


def test_fake_client_raises_when_exhausted():
    client = FakeLLMClient(["only one"])
    client.complete("a")
    with pytest.raises(AssertionError, match="exhausted"):
        client.complete("b")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_usage.py tests/test_llm.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'glassbox.usage'`

- [ ] **Step 3: Write `src/glassbox/__init__.py`**

```python
"""glassbox - does explicit multi-stage decomposition improve LLM legal reasoning?"""

__version__ = "0.1.0"
```

- [ ] **Step 4: Write `src/glassbox/config.py`**

```python
"""Central settings. Everything experiment-affecting lives here, nowhere else."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"
CHECKLIST_DIR = DATA_DIR / "checklists"
OUTPUT_DIR = REPO_ROOT / "output"
RUNS_DIR = OUTPUT_DIR / "runs"

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

SYSTEM_MODEL = "openai/gpt-5-mini"

# None, not 0.7. Task 1 established that gpt-5-mini does not support temperature:
# it is absent from supported_parameters on every upstream route, and the call
# does not error - the parameter is silently dropped. Sending it would create the
# false impression that sampling is under our control. LLMClient omits the
# parameter entirely when this is None.
SYSTEM_TEMPERATURE = None

# Recipe 3 raises this. Confirm the accepted values against docs/api-surface.md
# before relying on them.
EFFORT_BASELINE = "low"
EFFORT_RAISED = "high"

# LEXam's September 2025 protocol: the minimum of three judges. Task 1 confirmed
# all three slugs resolve.
JUDGE_MODELS = ("openai/gpt-4o", "deepseek/deepseek-chat", "qwen/qwen3-32b")

# LEXam grades at temperature 0. Unlike gpt-5-mini these are not reasoning models
# and should honour it - but Task 5 must confirm each judge lists `temperature` in
# its supported_parameters, using the same free metadata endpoint Task 1 used. A
# judge silently sampling at its default would add noise to every score.
JUDGE_TEMPERATURE = 0.0

# USD per million tokens. VERIFY against current OpenRouter pricing before trusting
# any cost figure. A model absent from this table reports cost as None rather than
# zero, so a missing price can never masquerade as a free call.
PRICING_PER_MTOK: dict[str, dict[str, float]] = {
    "openai/gpt-5-mini": {"input": 0.25, "output": 2.00},
    "openai/gpt-4o": {"input": 2.50, "output": 10.00},
    "deepseek/deepseek-chat": {"input": 0.28, "output": 0.88},
    "qwen/qwen3-32b": {"input": 0.10, "output": 0.30},
}

MAX_ATTEMPTS = 5
```

- [ ] **Step 5: Write `src/glassbox/usage.py`**

```python
"""Token and cost accounting.

Reasoning tokens are billed as output tokens, and are tracked separately so the
cost-versus-quality curve in spec section 9 can distinguish visible output from
hidden thinking.
"""

from __future__ import annotations

from dataclasses import dataclass

from glassbox.config import PRICING_PER_MTOK


@dataclass(frozen=True)
class Usage:
    input_tokens: int
    output_tokens: int
    reasoning_tokens: int
    calls: int

    @classmethod
    def zero(cls) -> "Usage":
        return cls(input_tokens=0, output_tokens=0, reasoning_tokens=0, calls=0)

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            reasoning_tokens=self.reasoning_tokens + other.reasoning_tokens,
            calls=self.calls + other.calls,
        )

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens + self.reasoning_tokens


def cost_usd(usage: Usage, model: str) -> float | None:
    """USD cost, or None when the model has no recorded price."""
    price = PRICING_PER_MTOK.get(model)
    if price is None:
        return None
    billed_output = usage.output_tokens + usage.reasoning_tokens
    return (
        usage.input_tokens * price["input"] + billed_output * price["output"]
    ) / 1_000_000
```

- [ ] **Step 6: Write `src/glassbox/llm.py`**

Build `_call` against whichever API `docs/api-surface.md` reports as working. The version below uses Chat Completions; if the probe showed the Responses API is required, replace the body of `_call` and the usage extraction, leaving every signature unchanged.

```python
"""Metered LLM client.

Every call records tokens and latency. Nothing else in the codebase talks to the
OpenAI SDK directly, so all cost accounting flows through one place.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential

from glassbox.config import (
    MAX_ATTEMPTS, OPENROUTER_BASE_URL, SYSTEM_MODEL, SYSTEM_TEMPERATURE,
)
from glassbox.usage import Usage


@dataclass(frozen=True)
class Completion:
    text: str
    usage: Usage
    model: str
    seconds: float


class LLMClient:
    def __init__(
        self,
        model: str = SYSTEM_MODEL,
        temperature: float | None = SYSTEM_TEMPERATURE,
        reasoning_effort: str | None = None,
    ) -> None:
        from openai import OpenAI

        load_dotenv()
        self.model = model
        self.temperature = temperature
        self.reasoning_effort = reasoning_effort
        self._client = OpenAI(
            base_url=OPENROUTER_BASE_URL,
            api_key=os.environ["OPENROUTER_API_KEY"],
        )

    @retry(stop=stop_after_attempt(MAX_ATTEMPTS),
           wait=wait_exponential(multiplier=2, min=2, max=60))
    def _call(self, messages: list[dict[str, str]]):
        kwargs: dict = {"model": self.model, "messages": messages}
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        if self.reasoning_effort is not None:
            kwargs["reasoning_effort"] = self.reasoning_effort
        return self._client.chat.completions.create(**kwargs)

    def complete(self, prompt: str, system: str | None = None) -> Completion:
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        started = time.monotonic()
        response = self._call(messages)
        seconds = time.monotonic() - started

        raw = response.usage
        details = getattr(raw, "completion_tokens_details", None)
        reasoning = getattr(details, "reasoning_tokens", 0) or 0
        visible = max((raw.completion_tokens or 0) - reasoning, 0)

        return Completion(
            text=response.choices[0].message.content or "",
            usage=Usage(
                input_tokens=raw.prompt_tokens or 0,
                output_tokens=visible,
                reasoning_tokens=reasoning,
                calls=1,
            ),
            model=self.model,
            seconds=seconds,
        )


@dataclass
class FakeLLMClient:
    """Test double. Returns queued responses and records what it was asked."""

    responses: list[str]
    model: str = "fake-model"
    # None, mirroring LLMClient. Recipes record getattr(client, "temperature", None)
    # into result metadata, so a default of 0.7 here would write a temperature into
    # test fixtures that the real client never sends.
    temperature: float | None = None
    reasoning_effort: str | None = None
    prompts: list[str] = field(default_factory=list)
    systems: list[str | None] = field(default_factory=list)
    _index: int = 0

    def complete(self, prompt: str, system: str | None = None) -> Completion:
        assert self._index < len(self.responses), (
            f"FakeLLMClient exhausted: {len(self.responses)} responses queued, "
            f"call {self._index + 1} requested"
        )
        text = self.responses[self._index]
        self._index += 1
        self.prompts.append(prompt)
        self.systems.append(system)
        return Completion(
            text=text,
            usage=Usage(
                input_tokens=len(prompt.split()),
                output_tokens=len(text.split()),
                reasoning_tokens=0,
                calls=1,
            ),
            model=self.model,
            seconds=0.0,
        )
```

- [ ] **Step 7: Run tests to verify they pass**

Run: `pytest tests/test_usage.py tests/test_llm.py -v`
Expected: all 8 PASS

- [ ] **Step 8: Verify against the live API**

Run: `python -c "from glassbox.llm import LLMClient; from glassbox.usage import cost_usd; c=LLMClient(); r=c.complete('In one sentence, what is promissory estoppel?'); print(r.text); print(r.usage); print('USD', cost_usd(r.usage, r.model)); print('%.2fs' % r.seconds)"`

Expected: a one-sentence answer, non-zero token counts, a cost figure, and a latency under ~30s. If `temperature` or the reasoning-effort parameter errors here, fix `_call` to match `docs/api-surface.md` — including whether effort is passed top-level or via `extra_body`.

- [ ] **Step 9: Commit**

```bash
git add src/glassbox tests/test_usage.py tests/test_llm.py
git commit -m "feat: metered LLM client with token and cost accounting"
```

---

# Day 2 — Recipe 1 across all 20 questions

**End state:** `python scripts/run_recipe.py --recipe plain --questions dev_20` produces 20 saved answers with complete provenance, and prints total tokens, cost and time.

### Task 3: Dataset loading

**Files:**
- Create: `src/glassbox/dataset.py`, `tests/test_dataset.py`

**Interfaces:**
- Consumes: `data/dev_20.json` and `data/cache/dev_20_full.json`, both written by `scripts/select_questions.py`
- Produces:
  - `Question(id, question, answer, course, area, jurisdiction, year, question_words, answer_words)`
  - `load_sample(name: str) -> list[Question]`
  - `load_manifest(name: str) -> dict`

- [ ] **Step 1: Write the failing test**

```python
import json

import pytest

from glassbox.dataset import Question, load_sample


def test_loads_twenty_dev_questions():
    questions = load_sample("dev_20")
    assert len(questions) == 20
    assert all(isinstance(q, Question) for q in questions)


def test_questions_carry_text_and_metadata():
    q = load_sample("dev_20")[0]
    assert len(q.question) > 100
    assert len(q.answer) > 100
    assert q.area in {"Public", "Private", "Criminal"}
    assert q.answer_words >= 150
    assert q.question_words >= 50


def test_order_is_stable_across_calls():
    assert [q.id for q in load_sample("dev_20")] == [q.id for q in load_sample("dev_20")]


def test_ids_match_the_committed_manifest():
    from glassbox.config import DATA_DIR

    manifest = json.loads((DATA_DIR / "dev_20.json").read_text(encoding="utf-8"))
    expected = [row["id"] for row in manifest["selected"]]
    assert [q.id for q in load_sample("dev_20")] == expected


def test_unknown_sample_raises():
    with pytest.raises(FileNotFoundError):
        load_sample("does_not_exist")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_dataset.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'glassbox.dataset'`

- [ ] **Step 3: Write `src/glassbox/dataset.py`**

```python
"""Load a frozen question sample.

Ids and metadata come from the versioned manifest; question and reference-answer
text comes from the gitignored cache, so the repository never redistributes LEXam
content. Order always follows the manifest, so runs are comparable across sessions.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from glassbox.config import CACHE_DIR, DATA_DIR


@dataclass(frozen=True)
class Question:
    id: str
    question: str
    answer: str
    course: str
    area: str
    jurisdiction: str
    year: str
    question_words: int
    answer_words: int


def _read_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Regenerate it with:\n"
            f"  python scripts/select_questions.py --split dev --n 20 "
            f"--seed 20260810 --name dev_20"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def load_manifest(name: str) -> dict:
    return _read_json(DATA_DIR / f"{name}.json")


def load_sample(name: str) -> list[Question]:
    manifest = load_manifest(name)
    rows = _read_json(CACHE_DIR / f"{name}_full.json")
    by_id = {row["id"]: row for row in rows}

    questions = []
    for entry in manifest["selected"]:
        row = by_id[entry["id"]]
        questions.append(
            Question(
                id=row["id"],
                question=row["question"],
                answer=row["answer"],
                course=row["course"],
                area=row["area"],
                jurisdiction=row["jurisdiction"],
                year=str(row["year"]),
                question_words=int(row["question_words"]),
                answer_words=int(row["answer_words"]),
            )
        )
    return questions
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_dataset.py -v`
Expected: all 5 PASS

- [ ] **Step 5: Commit**

```bash
git add src/glassbox/dataset.py tests/test_dataset.py
git commit -m "feat: load the frozen question sample from manifest plus cache"
```

---

### Task 4: Recipe 1, storage and the runner

**Files:**
- Create: `src/glassbox/recipes/__init__.py`, `src/glassbox/recipes/base.py`, `src/glassbox/recipes/plain.py`
- Create: `src/glassbox/storage.py`, `src/glassbox/runner.py`, `scripts/run_recipe.py`
- Create: `tests/test_plain.py`, `tests/test_storage.py`

**Interfaces:**
- Consumes: `Question`, `LLMClient`, `Usage`
- Produces:
  - `StageRecord(name, prompt, output, usage, seconds)`
  - `RecipeResult(recipe, question_id, final_answer, sections, stages, usage, seconds, metadata)`
  - `PlainRecipe(name="plain").run(q, client) -> RecipeResult`
  - `save_result(result, run_dir) -> Path`, `load_results(run_dir) -> list[RecipeResult]`
  - `run_recipe(recipe, questions, client, run_dir) -> list[RecipeResult]`

- [ ] **Step 1: Write the failing tests**

`tests/test_plain.py`:

```python
from glassbox.dataset import Question
from glassbox.llm import FakeLLMClient
from glassbox.recipes.plain import PlainRecipe

QUESTION = Question(
    id="q1", question="Does the shop owe a duty of care?", answer="Yes, shops owe...",
    course="Tort Law", area="Private", jurisdiction="Generic", year="2022",
    question_words=60, answer_words=200,
)


def test_plain_makes_exactly_one_call():
    client = FakeLLMClient(["The shop owes a duty of care because..."])
    PlainRecipe().run(QUESTION, client)
    assert len(client.prompts) == 1


def test_plain_returns_the_model_text_as_final_answer():
    client = FakeLLMClient(["The shop owes a duty of care because..."])
    result = PlainRecipe().run(QUESTION, client)
    assert result.final_answer == "The shop owes a duty of care because..."
    assert result.recipe == "plain"
    assert result.question_id == "q1"


def test_plain_has_no_sections_and_no_stages():
    result = PlainRecipe().run(QUESTION, FakeLLMClient(["answer"]))
    assert result.sections is None
    assert result.stages == []


def test_prompt_contains_question_and_course_but_not_reference_answer():
    client = FakeLLMClient(["answer"])
    PlainRecipe().run(QUESTION, client)
    prompt = client.prompts[0]
    assert QUESTION.question in prompt
    assert QUESTION.course in prompt
    assert "Yes, shops owe" not in prompt


def test_usage_is_recorded():
    result = PlainRecipe().run(QUESTION, FakeLLMClient(["answer"]))
    assert result.usage.calls == 1
```

`tests/test_storage.py`:

```python
from glassbox.recipes.base import RecipeResult
from glassbox.storage import load_results, save_result
from glassbox.usage import Usage

RESULT = RecipeResult(
    recipe="plain", question_id="q1", final_answer="an answer", sections=None,
    stages=[], usage=Usage(input_tokens=10, output_tokens=20, reasoning_tokens=5, calls=1),
    seconds=1.5, metadata={"model": "gpt-5-mini", "temperature": 0.7},
)


def test_round_trips_a_result(tmp_path):
    save_result(RESULT, tmp_path)
    loaded = load_results(tmp_path)
    assert len(loaded) == 1
    assert loaded[0] == RESULT


def test_filename_includes_recipe_and_question_id(tmp_path):
    path = save_result(RESULT, tmp_path)
    assert "plain" in path.name and "q1" in path.name


def test_results_load_in_a_stable_order(tmp_path):
    for qid in ["q3", "q1", "q2"]:
        save_result(
            RecipeResult(recipe="plain", question_id=qid, final_answer="a", sections=None,
                         stages=[], usage=Usage.zero(), seconds=0.0, metadata={}),
            tmp_path,
        )
    assert [r.question_id for r in load_results(tmp_path)] == ["q1", "q2", "q3"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_plain.py tests/test_storage.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'glassbox.recipes'`

- [ ] **Step 3: Write `src/glassbox/recipes/base.py` and an empty `src/glassbox/recipes/__init__.py`**

```python
"""Shared recipe types.

Every recipe turns one Question into one RecipeResult. Single-call recipes leave
`stages` empty; the pipeline fills it, one StageRecord per call, which is what
makes stage-level error attribution possible later.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from glassbox.dataset import Question
from glassbox.usage import Usage


@dataclass(frozen=True)
class StageRecord:
    name: str
    prompt: str
    output: str
    usage: Usage
    seconds: float


@dataclass(frozen=True)
class RecipeResult:
    recipe: str
    question_id: str
    final_answer: str
    sections: dict[str, str] | None
    stages: list[StageRecord]
    usage: Usage
    seconds: float
    metadata: dict = field(default_factory=dict)


class Recipe(Protocol):
    name: str

    def run(self, question: Question, client) -> RecipeResult: ...
```

- [ ] **Step 4: Write `src/glassbox/recipes/plain.py`**

The prompt below is adapted from LEXam's own `QA_PROMPT`, with the Swiss-law and German-language defaults removed because this study uses only English questions, most of which are international or generic in jurisdiction. It is deliberately strong — spec §5 requires the baseline not be handicapped.

```python
"""Recipe 1: one call, one strong open legal-reasoning prompt."""

from __future__ import annotations

import time

from glassbox.dataset import Question
from glassbox.recipes.base import RecipeResult

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
            },
        )
```

- [ ] **Step 5: Write `src/glassbox/storage.py`**

```python
"""Persist and reload recipe results as JSON, one file per result."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from glassbox.recipes.base import RecipeResult, StageRecord
from glassbox.usage import Usage


def save_result(result: RecipeResult, run_dir: Path) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / f"{result.recipe}__{result.question_id}.json"
    path.write_text(json.dumps(asdict(result), indent=2, ensure_ascii=False),
                    encoding="utf-8")
    return path


def _to_result(payload: dict) -> RecipeResult:
    return RecipeResult(
        recipe=payload["recipe"],
        question_id=payload["question_id"],
        final_answer=payload["final_answer"],
        sections=payload["sections"],
        stages=[
            StageRecord(
                name=s["name"], prompt=s["prompt"], output=s["output"],
                usage=Usage(**s["usage"]), seconds=s["seconds"],
            )
            for s in payload["stages"]
        ],
        usage=Usage(**payload["usage"]),
        seconds=payload["seconds"],
        metadata=payload.get("metadata", {}),
    )


def load_results(run_dir: Path) -> list[RecipeResult]:
    results = [
        _to_result(json.loads(p.read_text(encoding="utf-8")))
        for p in sorted(Path(run_dir).glob("*.json"))
    ]
    return sorted(results, key=lambda r: (r.recipe, r.question_id))
```

- [ ] **Step 6: Write `src/glassbox/runner.py`**

```python
"""Run one recipe over a question set, saving each result as it completes."""

from __future__ import annotations

from pathlib import Path

from glassbox.dataset import Question
from glassbox.recipes.base import RecipeResult
from glassbox.storage import save_result
from glassbox.usage import Usage


def run_recipe(recipe, questions: list[Question], client, run_dir: Path,
               verbose: bool = True) -> list[RecipeResult]:
    results: list[RecipeResult] = []
    total = Usage.zero()

    for i, question in enumerate(questions, 1):
        result = recipe.run(question, client)
        save_result(result, run_dir)
        results.append(result)
        total = total + result.usage
        if verbose:
            print(f"[{i}/{len(questions)}] {question.id[:8]} "
                  f"{result.usage.total_tokens:>7} tok  {result.seconds:>5.1f}s")

    return results
```

- [ ] **Step 7: Write `scripts/run_recipe.py`**

```python
"""Run a recipe over a frozen question set."""

from __future__ import annotations

import argparse

from glassbox.config import EFFORT_BASELINE, RUNS_DIR, SYSTEM_MODEL, SYSTEM_TEMPERATURE
from glassbox.dataset import load_sample
from glassbox.llm import LLMClient
from glassbox.recipes.plain import PlainRecipe
from glassbox.runner import run_recipe
from glassbox.usage import Usage, cost_usd

RECIPES = {"plain": PlainRecipe}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--recipe", required=True, choices=sorted(RECIPES))
    p.add_argument("--questions", default="dev_20")
    p.add_argument("--effort", default=EFFORT_BASELINE)
    p.add_argument("--tag", default="", help="suffix for the output directory")
    a = p.parse_args()

    questions = load_sample(a.questions)
    client = LLMClient(model=SYSTEM_MODEL, temperature=SYSTEM_TEMPERATURE,
                       reasoning_effort=a.effort)
    run_dir = RUNS_DIR / f"{a.questions}__{a.recipe}{a.tag}"

    print(f"{a.recipe} over {len(questions)} questions -> {run_dir}\n")
    results = run_recipe(RECIPES[a.recipe](), questions, client, run_dir)

    total = Usage.zero()
    for r in results:
        total = total + r.usage
    cost = cost_usd(total, SYSTEM_MODEL)

    print(f"\ncalls          {total.calls}")
    print(f"input tokens   {total.input_tokens:,}")
    print(f"output tokens  {total.output_tokens:,}")
    print(f"reasoning tok  {total.reasoning_tokens:,}")
    print(f"total tokens   {total.total_tokens:,}")
    print(f"cost           {'unknown' if cost is None else f'${cost:.4f}'}")
    print(f"time           {sum(r.seconds for r in results):.1f}s")


if __name__ == "__main__":
    main()
```

- [ ] **Step 8: Run tests to verify they pass**

Run: `pytest tests/ -v`
Expected: all PASS

- [ ] **Step 9: Run Recipe 1 for real**

Run: `python scripts/run_recipe.py --recipe plain --questions dev_20`
Expected: 20 progress lines, 20 files in `output/runs/dev_20__plain/`, and a totals block.

Now **read three of the answers**. You are checking for problems that no test catches: refusals, disclaimers slipping in despite the instruction, answers in the wrong language, or truncation mid-sentence. If any appear, fix `PLAIN_PROMPT` and re-run before continuing — this is your baseline and spec §5 requires it not be handicapped.

- [ ] **Step 10: Commit**

```bash
git add src/glassbox tests/ scripts/run_recipe.py
git commit -m "feat: Recipe 1 (plain), run persistence and the recipe runner"
```

---

# Day 3 — LEXam's judge and the calibration gate

**End state:** `python scripts/calibrate.py` proves the judge is wired correctly, and every Recipe 1 answer has an official LEXam score.

### Task 5: The LEXam judge

**Files:**
- Create: `src/glassbox/grading/__init__.py`, `src/glassbox/grading/lexam_judge.py`
- Create: `tests/test_lexam_judge.py`

**Interfaces:**
- Consumes: `LLMClient`
- Produces:
  - `LEXAM_JUDGE_SYSTEM`, `LEXAM_JUDGE_PROMPT`
  - `parse_score(text: str) -> float | None`
  - `judge_once(client, question, reference, candidate) -> tuple[float | None, str]`
  - `ensemble_min(scores: list[float | None]) -> float | None`
  - `judge_answer(clients: list, question, reference, candidate) -> JudgeVerdict`
  - `JudgeVerdict(score, per_judge, explanations)`

- [ ] **Step 1: Write the failing tests**

```python
from glassbox.grading.lexam_judge import (
    JudgeVerdict, ensemble_min, judge_answer, judge_once, parse_score,
)
from glassbox.llm import FakeLLMClient


def test_parses_a_well_formed_score():
    assert parse_score("Good reasoning. The correctness score: [[0.7]]") == 0.7


def test_parses_the_last_score_when_several_appear():
    assert parse_score("maybe [[0.3]] but on reflection [[0.8]]") == 0.8


def test_returns_none_when_no_score_present():
    assert parse_score("This answer is quite good overall.") is None


def test_clamps_out_of_range_scores_to_zero():
    assert parse_score("score: [[9.9]]") == 0.0


def test_ensemble_takes_the_minimum():
    assert ensemble_min([0.8, 0.4, 0.6]) == 0.4


def test_ensemble_ignores_unparseable_judges():
    assert ensemble_min([0.8, None, 0.6]) == 0.6


def test_ensemble_is_none_when_every_judge_failed():
    assert ensemble_min([None, None]) is None


def test_judge_once_returns_score_and_full_text():
    client = FakeLLMClient(["Reasoning here. The correctness score: [[0.6]]"])
    score, text = judge_once(client, "the question", "the reference", "the candidate")
    assert score == 0.6
    assert "correctness score" in text


def test_judge_prompt_contains_all_three_inputs():
    client = FakeLLMClient(["[[0.5]]"])
    judge_once(client, "QQQ", "RRR", "CCC")
    prompt = client.prompts[0]
    assert "QQQ" in prompt and "RRR" in prompt and "CCC" in prompt


def test_judge_answer_across_multiple_judges_takes_the_minimum():
    clients = [FakeLLMClient(["[[0.9]]"]), FakeLLMClient(["[[0.4]]"])]
    verdict = judge_answer(clients, "q", "r", "c")
    assert isinstance(verdict, JudgeVerdict)
    assert verdict.score == 0.4
    assert verdict.per_judge == [0.9, 0.4]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_lexam_judge.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'glassbox.grading'`

- [ ] **Step 3: Write `src/glassbox/grading/lexam_judge.py` and an empty `src/glassbox/grading/__init__.py`**

The system prompt and the `[[score]]` contract are LEXam's, reproduced so scores stay comparable with their published results. The one deliberate change: their system prompt says "Swiss law schools exams"; this study uses only English questions, mostly international or generic in jurisdiction, so the wording is generalised. Record this in results metadata.

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_lexam_judge.py -v`
Expected: all 10 PASS

- [ ] **Step 5: Commit**

```bash
git add src/glassbox/grading tests/test_lexam_judge.py
git commit -m "feat: LEXam official judge with score parsing and min-ensemble"
```

---

### Task 6: The calibration gate

Three checks, cheapest first. **If any fails, stop and fix the wiring — every later number depends on this.**

Full replication of LEXam's published 60.32 is not attempted: that figure covers 2,541 questions of which ~83% are German, so reproducing it would cost thousands of calls and tell you little about your English subset. Instead: two deterministic sanity checks that must pass, plus an optional partial replication that must land in a plausible band.

**Files:**
- Create: `scripts/calibrate.py`
- Create: `tests/test_calibrate.py`

**Interfaces:**
- Consumes: `judge_answer`, `load_sample`, `LLMClient`
- Produces: `check_reference_scores_high`, `check_empty_scores_low`, each returning `(passed: bool, mean: float, detail: str)`

- [ ] **Step 1: Write the failing test**

```python
from glassbox.llm import FakeLLMClient

from scripts.calibrate import check_empty_scores_low, check_reference_scores_high


class _Q:
    def __init__(self, qid, question, answer):
        self.id, self.question, self.answer = qid, question, answer


QUESTIONS = [_Q("q1", "question one", "reference one"),
             _Q("q2", "question two", "reference two")]


def test_reference_check_passes_when_judge_scores_high():
    clients = [FakeLLMClient(["[[1.0]]", "[[0.9]]"])]
    passed, mean, _ = check_reference_scores_high(QUESTIONS, clients, threshold=0.8)
    assert passed is True
    assert mean == 0.95


def test_reference_check_fails_when_judge_scores_low():
    clients = [FakeLLMClient(["[[0.2]]", "[[0.3]]"])]
    passed, mean, _ = check_reference_scores_high(QUESTIONS, clients, threshold=0.8)
    assert passed is False
    assert mean == 0.25


def test_empty_check_passes_when_judge_scores_low():
    clients = [FakeLLMClient(["[[0.0]]", "[[0.1]]"])]
    passed, mean, _ = check_empty_scores_low(QUESTIONS, clients, threshold=0.2)
    assert passed is True


def test_empty_check_fails_when_judge_is_too_generous():
    clients = [FakeLLMClient(["[[0.6]]", "[[0.7]]"])]
    passed, mean, _ = check_empty_scores_low(QUESTIONS, clients, threshold=0.2)
    assert passed is False
```

- [ ] **Step 2: Run test to verify it fails**

First create the empty package marker so `scripts` is importable:

```bash
touch scripts/__init__.py
```

Run: `pytest tests/test_calibrate.py -v`
Expected: FAIL — `ImportError: cannot import name 'check_reference_scores_high'`

- [ ] **Step 3: Write `scripts/calibrate.py`**

```python
"""Calibration gate. Run before trusting any score.

Check 1 - the reference answer, judged against itself, must score high. If it does
not, the prompt, the parsing or the model wiring is broken.

Check 2 - an empty answer must score near zero. If it does not, the judge is
rewarding presence rather than content, and every later comparison is noise.

Check 3 (optional, --replicate N) - run LEXam's own QA_PROMPT over N random test-split
questions and check the mean lands near their published figure for the model.
"""

from __future__ import annotations

import argparse
import statistics

from glassbox.config import JUDGE_MODELS, JUDGE_TEMPERATURE
from glassbox.dataset import load_sample
from glassbox.grading.lexam_judge import judge_answer
from glassbox.llm import LLMClient

EMPTY_ANSWER = "I am not able to answer this question."


def _mean(scores: list[float | None]) -> float:
    usable = [s for s in scores if s is not None]
    return statistics.mean(usable) if usable else 0.0


def check_reference_scores_high(questions, clients, threshold: float = 0.8):
    scores = [
        judge_answer(clients, q.question, q.answer, q.answer).score for q in questions
    ]
    mean = _mean(scores)
    detail = ", ".join(f"{s}" for s in scores)
    return mean >= threshold, mean, detail


def check_empty_scores_low(questions, clients, threshold: float = 0.2):
    scores = [
        judge_answer(clients, q.question, q.answer, EMPTY_ANSWER).score for q in questions
    ]
    mean = _mean(scores)
    detail = ", ".join(f"{s}" for s in scores)
    return mean <= threshold, mean, detail


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--questions", default="dev_20")
    p.add_argument("--n", type=int, default=5, help="questions per sanity check")
    a = p.parse_args()

    questions = load_sample(a.questions)[: a.n]
    clients = [LLMClient(model=m, temperature=JUDGE_TEMPERATURE) for m in JUDGE_MODELS]
    print(f"judges: {', '.join(JUDGE_MODELS)}   questions: {len(questions)}\n")

    ok_ref, mean_ref, detail_ref = check_reference_scores_high(questions, clients)
    print(f"[{'PASS' if ok_ref else 'FAIL'}] reference answer scores high  "
          f"mean={mean_ref:.2f} (need >= 0.80)")
    print(f"        per question: {detail_ref}")

    clients = [LLMClient(model=m, temperature=JUDGE_TEMPERATURE) for m in JUDGE_MODELS]
    ok_empty, mean_empty, detail_empty = check_empty_scores_low(questions, clients)
    print(f"[{'PASS' if ok_empty else 'FAIL'}] empty answer scores low       "
          f"mean={mean_empty:.2f} (need <= 0.20)")
    print(f"        per question: {detail_empty}")

    if ok_ref and ok_empty:
        print("\nGATE PASSED - judge wiring is sound.")
    else:
        raise SystemExit("\nGATE FAILED - fix the judge before grading anything.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_calibrate.py -v`
Expected: all 4 PASS

- [ ] **Step 5: Run the gate for real**

Run: `python scripts/calibrate.py --questions dev_20 --n 5`
Expected: both checks PASS.

If the reference check fails, read one full judge explanation — print it by hand — before changing anything. The usual causes are a malformed prompt, a `[[score]]` the regex misses, or the reference answer being a marking scheme rather than a model answer (spec §3), which a judge may legitimately score lower. If marking schemes are the cause, note it and raise the threshold discussion in the spec rather than weakening the check silently.

- [ ] **Step 6: Add judge scoring to `scripts/grade_runs.py`**

```python
"""Score persisted runs with LEXam's official judge."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from glassbox.config import JUDGE_MODELS, JUDGE_TEMPERATURE, RUNS_DIR
from glassbox.dataset import load_sample
from glassbox.grading.lexam_judge import judge_answer
from glassbox.llm import LLMClient
from glassbox.storage import load_results


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--questions", default="dev_20")
    a = p.parse_args()

    run_dir = Path(a.run_dir) if Path(a.run_dir).is_absolute() else RUNS_DIR / a.run_dir
    results = load_results(run_dir)
    questions = {q.id: q for q in load_sample(a.questions)}
    clients = [LLMClient(model=m, temperature=JUDGE_TEMPERATURE) for m in JUDGE_MODELS]

    rows = []
    for i, r in enumerate(results, 1):
        q = questions[r.question_id]
        verdict = judge_answer(clients, q.question, q.answer, r.final_answer)
        rows.append({
            "question_id": r.question_id, "recipe": r.recipe,
            "lexam_score": verdict.score, "per_judge": verdict.per_judge,
            "judges": list(JUDGE_MODELS),
        })
        print(f"[{i}/{len(results)}] {r.question_id[:8]} {verdict.score}")

    out = run_dir / "lexam_scores.json"
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    scored = [r["lexam_score"] for r in rows if r["lexam_score"] is not None]
    print(f"\nscored {len(scored)}/{len(rows)}")
    if scored:
        print(f"mean {statistics.mean(scored) * 100:.1f} / 100")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 7: Score Recipe 1**

Run: `python scripts/grade_runs.py --run-dir dev_20__plain --questions dev_20`
Expected: 20 scores and a mean. Sanity check: it should be roughly in the 40–75 range. Far outside that, on either side, means something is wrong — investigate before continuing.

- [ ] **Step 8: Commit**

```bash
git add scripts/calibrate.py scripts/grade_runs.py tests/test_calibrate.py
git commit -m "feat: calibration gate and LEXam judge scoring of persisted runs"
```

---

# Day 4 — Checklists for human verification

**End state:** every dev question has a drafted checklist of atomic points, written to a markdown file you can read and correct by hand.

### Task 7: Checklist builder

**Files:**
- Create: `src/glassbox/grading/checklist.py`, `scripts/build_checklists.py`
- Create: `tests/test_checklist_build.py`

**Interfaces:**
- Consumes: `Question`, `LLMClient`
- Produces:
  - `ChecklistPoint(id, text)`
  - `Checklist(question_id, source, points)`
  - `build_checklist(question, client) -> Checklist`
  - `save_checklist(checklist, directory) -> Path`, `load_checklist(question_id, directory) -> Checklist`
  - `checklist_to_markdown(checklist, question) -> str`

- [ ] **Step 1: Write the failing tests**

```python
from glassbox.dataset import Question
from glassbox.grading.checklist import (
    Checklist, build_checklist, load_checklist, save_checklist,
)
from glassbox.llm import FakeLLMClient

QUESTION = Question(
    id="q1", question="Is the shop liable?", answer="Shops owe a duty of care. "
    "Failing to display a warning sign breaches it. The breach caused the injury.",
    course="Tort Law", area="Private", jurisdiction="Generic", year="2022",
    question_words=60, answer_words=200,
)

RESPONSE = """{"source": "model_answer", "points": [
  {"text": "Shops owe customers a duty of care."},
  {"text": "Failing to display a warning sign breaches that duty."},
  {"text": "The breach caused the injury."}]}"""


def test_builds_one_point_per_proposition():
    checklist = build_checklist(QUESTION, FakeLLMClient([RESPONSE]))
    assert isinstance(checklist, Checklist)
    assert len(checklist.points) == 3
    assert checklist.points[0].text == "Shops owe customers a duty of care."


def test_point_ids_are_stable_and_prefixed_by_question():
    checklist = build_checklist(QUESTION, FakeLLMClient([RESPONSE]))
    assert [p.id for p in checklist.points] == ["q1-p1", "q1-p2", "q1-p3"]


def test_records_whether_reference_was_model_answer_or_marking_scheme():
    checklist = build_checklist(QUESTION, FakeLLMClient([RESPONSE]))
    assert checklist.source == "model_answer"


def test_tolerates_json_wrapped_in_a_code_fence():
    fenced = f"Here you go:\n```json\n{RESPONSE}\n```"
    assert len(build_checklist(QUESTION, FakeLLMClient([fenced])).points) == 3


def test_prompt_contains_reference_answer():
    client = FakeLLMClient([RESPONSE])
    build_checklist(QUESTION, client)
    assert "Shops owe a duty of care" in client.prompts[0]


def test_round_trips_through_disk(tmp_path):
    checklist = build_checklist(QUESTION, FakeLLMClient([RESPONSE]))
    save_checklist(checklist, tmp_path)
    assert load_checklist("q1", tmp_path) == checklist
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_checklist_build.py -v`
Expected: FAIL — `ImportError: cannot import name 'build_checklist'`

- [ ] **Step 3: Write `src/glassbox/grading/checklist.py`**

```python
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

Split the reference answer into its atomic checkable points: the individual legal \
propositions a candidate would have to make to earn full marks. One proposition per \
point. Do not add anything the reference answer does not contain, and do not merge two \
propositions into one point.

Some reference answers are model answers written in prose. Others are marking schemes \
addressed to the grader ("the answer should at a minimum raise..."). Handle both: for a \
marking scheme, each required element is already a point. Report which kind it was.

Return only JSON, in exactly this form:
{{"source": "model_answer" or "marking_scheme",
  "points": [{{"text": "..."}}, {{"text": "..."}}]}}

### Question
{question}

### Reference answer
{reference}
"""


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
    points = [
        ChecklistPoint(id=f"{question.id}-p{i}", text=p["text"].strip())
        for i, p in enumerate(payload["points"], 1)
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_checklist_build.py -v`
Expected: all 6 PASS

- [ ] **Step 5: Write `scripts/build_checklists.py`**

```python
"""Draft a checklist for every question in a sample, for human verification."""

from __future__ import annotations

import argparse
import statistics

from glassbox.config import CHECKLIST_DIR, SYSTEM_MODEL
from glassbox.dataset import load_sample
from glassbox.grading.checklist import build_checklist, checklist_to_markdown, save_checklist
from glassbox.llm import LLMClient


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--questions", default="dev_20")
    a = p.parse_args()

    questions = load_sample(a.questions)
    client = LLMClient(model=SYSTEM_MODEL, temperature=0.0)
    review_dir = CHECKLIST_DIR / "review"
    review_dir.mkdir(parents=True, exist_ok=True)

    counts, sources = [], []
    for i, q in enumerate(questions, 1):
        checklist = build_checklist(q, client)
        save_checklist(checklist, CHECKLIST_DIR)
        (review_dir / f"{q.id}.md").write_text(
            checklist_to_markdown(checklist, q), encoding="utf-8"
        )
        counts.append(len(checklist.points))
        sources.append(checklist.source)
        print(f"[{i}/{len(questions)}] {q.id[:8]} {len(checklist.points):>2} points "
              f"({checklist.source})")

    print(f"\ntotal points   {sum(counts)}")
    print(f"median/question {statistics.median(counts):.1f}  "
          f"range {min(counts)}-{max(counts)}")
    print(f"reference types {dict((s, sources.count(s)) for s in set(sources))}")
    print(f"\nreview the markdown in {review_dir} and correct it by hand")


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Build the checklists for real**

Run: `python scripts/build_checklists.py --questions dev_20`
Expected: 20 checklists. Total points should land roughly in the 100–200 range; median per question around 5–10.

If the median is 2 or fewer, the model is merging propositions — tighten `BUILD_PROMPT` and re-run. If it exceeds 20 per question it is splitting hairs, which inflates apparent coverage differences; also tighten and re-run.

- [ ] **Step 7: Verify by hand — this is yours, not the agent's**

Read all 20 files in `data/checklists/review/`. For each, check: does every point actually appear in the reference answer, is anything the reference says missing, and is each point one proposition rather than three.

Correct the markdown directly. Then re-sync the JSON from your corrections — **spec §7.4 requires checklists to be frozen before any answer is scored**, so this must be finished before Day 5.

- [ ] **Step 8: Keep checklist text out of version control**

Checklists are derived from LEXam reference answers, so they fall under the same
constraint as the cached question text. Append to `.gitignore`:

```
# Checklists derived from LEXam reference answers - same restriction as data/cache/
data/checklists/
```

Run: `git status --short`
Expected: no `data/checklists/` entries appear.

- [ ] **Step 9: Commit**

```bash
git add .gitignore src/glassbox/grading/checklist.py scripts/build_checklists.py tests/test_checklist_build.py
git commit -m "feat: build reference checklists for human verification"
```

---

# Day 5 — Blind normalisation and checklist scoring

**End state:** a full score table for Recipe 1 — LEXam score, sub-scores, and coverage of every checklist point — all graded on text stripped of any clue about which recipe produced it.

### Task 8: Blind normalisation

**Files:**
- Create: `src/glassbox/grading/normalise.py`, `tests/test_normalise.py`

**Interfaces:**
- Consumes: `RecipeResult`
- Produces: `normalise(result: RecipeResult) -> str`

- [ ] **Step 1: Write the failing tests**

```python
from glassbox.grading.normalise import normalise
from glassbox.recipes.base import RecipeResult
from glassbox.usage import Usage


def _result(recipe, final_answer, sections=None):
    return RecipeResult(recipe=recipe, question_id="q1", final_answer=final_answer,
                        sections=sections, stages=[], usage=Usage.zero(), seconds=0.0,
                        metadata={})


def test_plain_answer_passes_through_with_whitespace_collapsed():
    assert normalise(_result("plain", "The shop  is\n\n\nliable.")) == "The shop is\n\nliable."


def test_sectioned_answer_is_flattened_to_prose_without_section_headings():
    out = normalise(_result("pipeline", "final text", sections={
        "issues": "Duty of care.", "conclusion": "Liable."}))
    assert "issues" not in out.lower()
    assert "Duty of care." in out and "Liable." in out


def test_recipe_name_never_leaks_into_normalised_text():
    for recipe in ["plain", "structured", "think_longer", "pipeline"]:
        assert recipe not in normalise(_result(recipe, "an answer")).lower()


def test_stage_and_json_artefacts_are_stripped():
    text = '```json\n{"issues": ["a"]}\n```\nStage 1 output: The shop is liable.'
    out = normalise(_result("pipeline", text))
    assert "```" not in out
    assert "stage 1" not in out.lower()


def test_identical_content_normalises_identically_across_recipes():
    a = normalise(_result("plain", "The shop is liable."))
    b = normalise(_result("pipeline", "The shop is liable."))
    assert a == b
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_normalise.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'glassbox.grading.normalise'`

- [ ] **Step 3: Write `src/glassbox/grading/normalise.py`**

```python
"""Strip every clue about which recipe produced an answer, before grading.

Spec section 7.3. The thesis this project follows on from found that LLM legal
output can look structurally complete and professional while being substantively
wrong. Pipeline output will look tidier than single-pass output, and a judge that
can see structure will reward it. So the judge must not see structure.
"""

from __future__ import annotations

import re

from glassbox.recipes.base import RecipeResult

_FENCE = re.compile(r"```[a-zA-Z]*\n?|```")
_STAGE_MARKER = re.compile(
    r"^\s*(stage|step)\s*\d+\s*(output)?\s*[:.\-]\s*", re.IGNORECASE | re.MULTILINE
)
_SECTION_HEADING = re.compile(
    r"^\s*#{1,6}\s*(issues?|rules?|elements?|application|subsumption|conclusion|"
    r"analysis|final answer)\s*:?\s*$",
    re.IGNORECASE | re.MULTILINE,
)
_BLANK_RUN = re.compile(r"\n{3,}")
_SPACE_RUN = re.compile(r"[ \t]{2,}")


def normalise(result: RecipeResult) -> str:
    if result.sections:
        text = "\n\n".join(v.strip() for v in result.sections.values() if v and v.strip())
    else:
        text = result.final_answer or ""

    text = _FENCE.sub("", text)
    text = _SECTION_HEADING.sub("", text)
    text = _STAGE_MARKER.sub("", text)
    text = _SPACE_RUN.sub(" ", text)
    text = _BLANK_RUN.sub("\n\n", text)
    return text.strip()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_normalise.py -v`
Expected: all 5 PASS

- [ ] **Step 5: Commit**

```bash
git add src/glassbox/grading/normalise.py tests/test_normalise.py
git commit -m "feat: blind normalisation so graders cannot see which recipe produced an answer"
```

---

### Task 9: Checklist scoring and sub-scores

**Files:**
- Modify: `src/glassbox/grading/checklist.py` (append scoring)
- Create: `src/glassbox/grading/subscores.py`, `tests/test_checklist_score.py`, `tests/test_subscores.py`
- Modify: `scripts/grade_runs.py`

**Interfaces:**
- Consumes: `Checklist`, `normalise`, `LLMClient`
- Produces:
  - `PointVerdict(point_id, coverage, evidence)` where coverage is `"covered" | "partial" | "missed"`
  - `ChecklistVerdict(question_id, verdicts, contradictions, inventions)`
  - `score_checklist(checklist, answer_text, client) -> ChecklistVerdict`
  - `coverage_fraction(verdict) -> float` — covered counts 1.0, partial 0.5, missed 0.0
  - `SubScores(issue_spotting, rule_recall, rule_application, conclusion)`
  - `score_subscores(question, reference, answer_text, client) -> SubScores`

- [ ] **Step 1: Write the failing tests**

`tests/test_checklist_score.py`:

```python
from glassbox.grading.checklist import (
    Checklist, ChecklistPoint, coverage_fraction, score_checklist,
)
from glassbox.llm import FakeLLMClient

CHECKLIST = Checklist(question_id="q1", source="model_answer", points=[
    ChecklistPoint(id="q1-p1", text="Shops owe a duty of care."),
    ChecklistPoint(id="q1-p2", text="No warning sign breaches that duty."),
    ChecklistPoint(id="q1-p3", text="The breach caused the injury."),
])

RESPONSE = """{"verdicts": [
  {"point_id": "q1-p1", "coverage": "covered", "evidence": "shops owe a duty"},
  {"point_id": "q1-p2", "coverage": "partial", "evidence": "mentions the sign"},
  {"point_id": "q1-p3", "coverage": "missed", "evidence": ""}],
 "contradictions": ["says duty is owed only to employees"],
 "inventions": ["cites a non-existent Occupiers Act 1998"]}"""


def test_returns_one_verdict_per_point():
    v = score_checklist(CHECKLIST, "an answer", FakeLLMClient([RESPONSE]))
    assert [x.point_id for x in v.verdicts] == ["q1-p1", "q1-p2", "q1-p3"]
    assert [x.coverage for x in v.verdicts] == ["covered", "partial", "missed"]


def test_captures_contradictions_and_inventions():
    v = score_checklist(CHECKLIST, "an answer", FakeLLMClient([RESPONSE]))
    assert len(v.contradictions) == 1
    assert len(v.inventions) == 1


def test_coverage_fraction_counts_partial_as_half():
    v = score_checklist(CHECKLIST, "an answer", FakeLLMClient([RESPONSE]))
    assert coverage_fraction(v) == (1.0 + 0.5 + 0.0) / 3


def test_missing_points_in_judge_output_default_to_missed():
    sparse = '{"verdicts": [{"point_id": "q1-p1", "coverage": "covered", "evidence": "x"}], \
"contradictions": [], "inventions": []}'
    v = score_checklist(CHECKLIST, "an answer", FakeLLMClient([sparse]))
    assert len(v.verdicts) == 3
    assert [x.coverage for x in v.verdicts] == ["covered", "missed", "missed"]


def test_prompt_lists_every_point_and_the_answer():
    client = FakeLLMClient([RESPONSE])
    score_checklist(CHECKLIST, "THE ANSWER TEXT", client)
    prompt = client.prompts[0]
    assert "q1-p1" in prompt and "q1-p3" in prompt
    assert "THE ANSWER TEXT" in prompt
```

`tests/test_subscores.py`:

```python
from glassbox.grading.subscores import SubScores, score_subscores
from glassbox.llm import FakeLLMClient

RESPONSE = """{"issue_spotting": 0.8, "rule_recall": 0.6,
"rule_application": 0.4, "conclusion": 0.7}"""


def test_returns_all_four_criteria():
    s = score_subscores("the question", "the reference", "the answer",
                        FakeLLMClient([RESPONSE]))
    assert s == SubScores(issue_spotting=0.8, rule_recall=0.6,
                          rule_application=0.4, conclusion=0.7)


def test_clamps_out_of_range_values():
    client = FakeLLMClient(['{"issue_spotting": 1.9, "rule_recall": -0.3, '
                            '"rule_application": 0.5, "conclusion": 0.5}'])
    s = score_subscores("q", "r", "a", client)
    assert s.issue_spotting == 1.0
    assert s.rule_recall == 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_checklist_score.py tests/test_subscores.py -v`
Expected: FAIL — `ImportError: cannot import name 'score_checklist'`

- [ ] **Step 3: Append scoring to `src/glassbox/grading/checklist.py`**

```python
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
```

- [ ] **Step 4: Write `src/glassbox/grading/subscores.py`**

```python
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
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/ -v`
Expected: all PASS

- [ ] **Step 6: Extend `scripts/grade_runs.py` to all three layers**

Add these to the imports at the **top of the module**, alongside the existing ones:

```python
from glassbox.config import CHECKLIST_DIR
from glassbox.grading.checklist import coverage_fraction, load_checklist, score_checklist
from glassbox.grading.normalise import normalise
from glassbox.grading.subscores import score_subscores
```

Then replace the body of `main()` so each result is graded on normalised text through all three layers, and a summary table is printed:

```python
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--questions", default="dev_20")
    a = p.parse_args()

    run_dir = Path(a.run_dir) if Path(a.run_dir).is_absolute() else RUNS_DIR / a.run_dir
    results = load_results(run_dir)
    questions = {q.id: q for q in load_sample(a.questions)}
    judges = [LLMClient(model=m, temperature=JUDGE_TEMPERATURE) for m in JUDGE_MODELS]
    grader = LLMClient(model=JUDGE_MODELS[0], temperature=JUDGE_TEMPERATURE)

    rows = []
    for i, r in enumerate(results, 1):
        q = questions[r.question_id]
        blind = normalise(r)

        verdict = judge_answer(judges, q.question, q.answer, blind)
        subs = score_subscores(q.question, q.answer, blind, grader)
        checklist_verdict = score_checklist(
            load_checklist(q.id, CHECKLIST_DIR), blind, grader
        )
        coverage = coverage_fraction(checklist_verdict)

        rows.append({
            "question_id": r.question_id,
            "recipe": r.recipe,
            "lexam_score": verdict.score,
            "per_judge": verdict.per_judge,
            "judges": list(JUDGE_MODELS),
            "issue_spotting": subs.issue_spotting,
            "rule_recall": subs.rule_recall,
            "rule_application": subs.rule_application,
            "conclusion": subs.conclusion,
            "coverage": coverage,
            "points_total": len(checklist_verdict.verdicts),
            "points_covered": sum(
                1 for v in checklist_verdict.verdicts if v.coverage == "covered"),
            "contradictions": len(checklist_verdict.contradictions),
            "inventions": len(checklist_verdict.inventions),
            "total_tokens": r.usage.total_tokens,
        })
        print(f"[{i}/{len(results)}] {r.question_id[:8]} "
              f"lexam={verdict.score} coverage={coverage:.2f}")

    out = run_dir / "scores.json"
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    def mean_of(key):
        vals = [r[key] for r in rows if r[key] is not None]
        return statistics.mean(vals) if vals else float("nan")

    print(f"\n{'lexam score':<18}{mean_of('lexam_score') * 100:>8.1f} / 100")
    print(f"{'coverage':<18}{mean_of('coverage') * 100:>8.1f} %")
    print(f"{'issue spotting':<18}{mean_of('issue_spotting'):>8.2f}")
    print(f"{'rule recall':<18}{mean_of('rule_recall'):>8.2f}")
    print(f"{'rule application':<18}{mean_of('rule_application'):>8.2f}")
    print(f"{'conclusion':<18}{mean_of('conclusion'):>8.2f}")
    print(f"{'contradictions':<18}{sum(r['contradictions'] for r in rows):>8}")
    print(f"{'inventions':<18}{sum(r['inventions'] for r in rows):>8}")
    print(f"\nwrote {out}")
```

- [ ] **Step 7: Grade Recipe 1 fully**

Run: `python scripts/grade_runs.py --run-dir dev_20__plain --questions dev_20`
Expected: 20 rows and a summary. Coverage should be well below 100% — if Recipe 1 covers nearly every point, the questions are too easy or the checklists too coarse, and the study has a ceiling problem worth raising before Day 6.

- [ ] **Step 8: Commit**

```bash
git add src/glassbox/grading scripts/grade_runs.py tests/
git commit -m "feat: checklist scoring and per-criterion sub-scores on blinded text"
```

---

# Day 6 — Recipes 2 and 3

**End state:** all three single-call recipes scored side by side. **This is the deliverable that makes Phase 1 stand alone**, and the baseline every pipeline result is later compared against.

### Task 10: The shared answer schema

**Files:**
- Create: `src/glassbox/schema.py`, `tests/test_schema.py`

**Interfaces:**
- Consumes: nothing
- Produces: `Issue`, `Rule`, `ElementFinding`, `Amendment`, `CaseFile`, `case_file_json_instructions() -> str`, `parse_case_file(text, question_id) -> CaseFile`, `CaseFile.to_sections() -> dict[str, str]`

The pipeline in Phase 2 fills this **same** schema stage by stage. Recipe 2 fills it in one call. That is what isolates call boundaries as the only difference between them (spec §6.3), so the schema must be defined once, here, and shared.

- [ ] **Step 1: Write the failing tests**

```python
import pytest

from glassbox.schema import CaseFile, parse_case_file

PAYLOAD = """{"issues": [{"id": "i1", "statement": "Duty of care owed?",
   "why_it_arises": "The claimant was a customer."}],
 "rules": [{"issue_id": "i1", "rule": "Occupiers owe a duty of care.",
   "elements": ["occupier", "lawful visitor", "reasonable care"]}],
 "findings": [{"issue_id": "i1", "element": "occupier", "holds": "yes",
   "reasoning": "The shop controls the premises."}],
 "conclusion": "The shop is liable.",
 "final_answer": "The shop is liable because...",
 "amendments": []}"""


def test_parses_a_complete_case_file():
    cf = parse_case_file(PAYLOAD, "q1")
    assert isinstance(cf, CaseFile)
    assert cf.question_id == "q1"
    assert cf.issues[0].statement == "Duty of care owed?"
    assert cf.rules[0].elements == ["occupier", "lawful visitor", "reasonable care"]
    assert cf.findings[0].holds == "yes"


def test_parses_a_partial_case_file_with_only_issues():
    cf = parse_case_file('{"issues": [{"id": "i1", "statement": "s", "why_it_arises": "w"}]}', "q1")
    assert len(cf.issues) == 1
    assert cf.rules is None
    assert cf.conclusion is None


def test_tolerates_a_code_fence():
    cf = parse_case_file(f"```json\n{PAYLOAD}\n```", "q1")
    assert cf.conclusion == "The shop is liable."


def test_rejects_an_invalid_holds_value():
    with pytest.raises(ValueError):
        parse_case_file('{"findings": [{"issue_id": "i1", "element": "e", '
                        '"holds": "maybe", "reasoning": "r"}]}', "q1")


def test_to_sections_omits_empty_sections():
    sections = parse_case_file(PAYLOAD, "q1").to_sections()
    assert set(sections) == {"issues", "rules", "application", "conclusion", "final_answer"}
    empty = parse_case_file('{"issues": []}', "q1").to_sections()
    assert empty == {}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_schema.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'glassbox.schema'`

- [ ] **Step 3: Write `src/glassbox/schema.py`**

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_schema.py -v`
Expected: all 5 PASS

- [ ] **Step 5: Commit**

```bash
git add src/glassbox/schema.py tests/test_schema.py
git commit -m "feat: shared four-section case file schema"
```

---

### Task 11: Recipes 2 and 3

**Files:**
- Create: `src/glassbox/recipes/structured.py`, `tests/test_structured.py`
- Modify: `scripts/run_recipe.py` (register both recipes)

**Interfaces:**
- Consumes: `CaseFile`, `parse_case_file`, `case_file_json_instructions`, `Question`
- Produces: `StructuredRecipe(name="structured")`, `ThinkLongerRecipe(name="think_longer")` — both single-call, both `.run(q, client) -> RecipeResult` with `sections` populated

Recipes 2 and 3 share all code. They differ only in the `reasoning_effort` the client is constructed with, which is why the recipe records it in metadata rather than setting it itself.

- [ ] **Step 1: Write the failing tests**

```python
from glassbox.dataset import Question
from glassbox.llm import FakeLLMClient
from glassbox.recipes.structured import StructuredRecipe, ThinkLongerRecipe

QUESTION = Question(
    id="q1", question="Is the shop liable?", answer="reference text here",
    course="Tort Law", area="Private", jurisdiction="Generic", year="2022",
    question_words=60, answer_words=200,
)

PAYLOAD = """{"issues": [{"id": "i1", "statement": "Duty owed?", "why_it_arises": "customer"}],
 "rules": [{"issue_id": "i1", "rule": "Occupiers owe a duty.", "elements": ["occupier"]}],
 "findings": [{"issue_id": "i1", "element": "occupier", "holds": "yes", "reasoning": "controls"}],
 "conclusion": "Liable.", "final_answer": "The shop is liable because it controls the premises.",
 "amendments": []}"""


def test_structured_makes_exactly_one_call():
    client = FakeLLMClient([PAYLOAD])
    StructuredRecipe().run(QUESTION, client)
    assert len(client.prompts) == 1


def test_structured_populates_sections_and_final_answer():
    result = StructuredRecipe().run(QUESTION, FakeLLMClient([PAYLOAD]))
    assert result.recipe == "structured"
    assert result.final_answer.startswith("The shop is liable")
    assert set(result.sections) == {
        "issues", "rules", "application", "conclusion", "final_answer"}


def test_structured_records_no_stages():
    result = StructuredRecipe().run(QUESTION, FakeLLMClient([PAYLOAD]))
    assert result.stages == []


def test_prompt_asks_for_all_four_sections_and_omits_the_reference():
    client = FakeLLMClient([PAYLOAD])
    StructuredRecipe().run(QUESTION, client)
    prompt = client.prompts[0].lower()
    for word in ["issue", "rule", "element", "appl", "conclusion"]:
        assert word in prompt
    assert "reference text here" not in client.prompts[0]


def test_think_longer_uses_the_same_prompt_as_structured():
    a, b = FakeLLMClient([PAYLOAD]), FakeLLMClient([PAYLOAD])
    StructuredRecipe().run(QUESTION, a)
    ThinkLongerRecipe().run(QUESTION, b)
    assert a.prompts[0] == b.prompts[0]


def test_think_longer_is_named_distinctly():
    result = ThinkLongerRecipe().run(QUESTION, FakeLLMClient([PAYLOAD]))
    assert result.recipe == "think_longer"


def test_unparseable_output_falls_back_to_raw_text():
    result = StructuredRecipe().run(QUESTION, FakeLLMClient(["not json at all"]))
    assert result.final_answer == "not json at all"
    assert result.sections is None
    assert result.metadata["parse_failed"] is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_structured.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'glassbox.recipes.structured'`

- [ ] **Step 3: Write `src/glassbox/recipes/structured.py`**

```python
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

from glassbox.dataset import Question
from glassbox.recipes.base import RecipeResult
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


class StructuredRecipe:
    name = "structured"

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
            "parse_failed": False,
        }
        try:
            case_file = parse_case_file(completion.text, question.id)
            sections = case_file.to_sections()
            final_answer = case_file.final_answer or sections.get("conclusion", "")
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
```

- [ ] **Step 4: Register both recipes in `scripts/run_recipe.py`**

```python
from glassbox.recipes.structured import StructuredRecipe, ThinkLongerRecipe

RECIPES = {
    "plain": PlainRecipe,
    "structured": StructuredRecipe,
    "think_longer": ThinkLongerRecipe,
}
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/ -v`
Expected: all PASS

- [ ] **Step 6: Run both recipes for real**

```bash
python scripts/run_recipe.py --recipe structured   --questions dev_20 --effort low
python scripts/run_recipe.py --recipe think_longer --questions dev_20 --effort high
```

Expected: 20 results each. **Check the parse-failure rate** — count `"parse_failed": true` across the saved files. More than one or two means `STRUCTURED_PROMPT` needs tightening; a recipe that fails to parse is a recipe scoring zero for the wrong reason.

Also confirm Recipe 3 actually spent more tokens than Recipe 2. If the totals are the same, `reasoning_effort` had no effect, and Recipe 3 is not doing its job — go back to `docs/api-surface.md` and fix it before Day 7.

- [ ] **Step 7: Grade both**

```bash
python scripts/grade_runs.py --run-dir dev_20__structured   --questions dev_20
python scripts/grade_runs.py --run-dir dev_20__think_longer --questions dev_20
```

- [ ] **Step 8: Commit**

```bash
git add src/glassbox/recipes/structured.py scripts/run_recipe.py tests/test_structured.py
git commit -m "feat: Recipes 2 and 3, sharing the case file schema with the pipeline"
```

---

## Phase 1 complete

You now have three recipes scored on 20 questions across three grading layers, with full token and cost accounting. **Look at the three coverage numbers before starting Phase 2.**

| Observation | What it means for Phase 2 |
|---|---|
| Structured ≫ Plain | The instructions alone are doing a lot of work. H2 becomes the live hypothesis and the pipeline has a high bar to clear. |
| Structured ≈ Plain | Structure alone buys nothing; any pipeline gain would be attributable to the separate calls. |
| Think-longer ≫ Structured | Compute matters a lot on this task. The pipeline must beat Think-longer, not just Plain, to claim anything. |
| Coverage above ~90% anywhere | Ceiling problem. Raise it before Phase 2 — the model tier or the inclusion criterion may need revisiting. |

---

## Out of scope for this plan

Phases 2 and 3 are deliberately not specified in detail here. The pipeline's stage prompts and the fault-injection targets depend on what the Phase 1 checklists and answers actually look like — writing TDD steps for them today would be guesswork. They get their own plans, written once Phase 1 has run.

**Phase 2 — the pipeline (Days 7–11).** One stage per day: issues, rules and elements, application, conclusion plus the amendment log and violation counting, then a full dev-set run scored against the three Phase 1 recipes. `CaseFile` and `RecipeResult.stages` already exist for it.

**Phase 3 — reliability, damage, analysis (Days 12–14).** Three samples plus two rewordings with flip-rate scoring; the three Stage 1 damage types; then the analysis, cost curves and the blind human validation of 20 answers.

## Open items carried forward

1. **Judge slugs.** `deepseek/deepseek-chat` and `qwen/qwen3-32b` are educated guesses at OpenRouter's naming, and `deepseek-chat` must be the V3 generation LEXam used, not a later one. Task 1's probe confirms resolution; confirm the *generation* against OpenRouter's model page and record what was used.
2. **Pricing.** Verify `PRICING_PER_MTOK` against current OpenRouter pricing before quoting any cost figure. OpenRouter also returns actual cost per request, which is worth preferring over a static table if the field is present.
3. **Reasoning effort.** If the probe showed GPT-5-mini has no usable effect from `reasoning_effort`, Recipe 3 needs the fallback from spec §5 and the approximation must be reported.
4. **Contamination.** Unknown whether LEXam is in training data. Report per-question baseline scores and note any ceiling saturation as a finding rather than filtering questions.
5. **Pre-registration.** Register the inclusion criterion and the six hypotheses on OSF before drawing the evaluation set from the `test` split.
