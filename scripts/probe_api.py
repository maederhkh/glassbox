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
