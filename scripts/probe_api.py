"""Record what OpenRouter actually accepts for the models this study uses.

Run once, on Day 1. Two unknowns are settled here: which call parameters
gpt-5-mini honours, and whether every model slug in the study resolves at all.
A wrong judge slug discovered on Day 3 wastes a day; discovered here it costs
nothing. The result is written to docs/api-surface.md and the rest of the
codebase is built against whatever this reports as working.

A plain "the call didn't error" is not proof a parameter took effect --
OpenRouter (and some upstream providers) silently accept and drop parameters
a model doesn't support rather than rejecting the request. `temperature` is
checked separately from the other attempts for exactly this reason: we first
consult OpenRouter's public model metadata (no inference spend) for whether
`temperature` is a *supported* parameter for this model/route, and only fall
back to an empirical before/after comparison (a handful of cheap inference
calls) if that metadata is missing or ambiguous.
"""

import json
import os
import urllib.error
import urllib.request

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


def fetch_json(url: str):
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.load(resp)


def check_temperature_support(model: str) -> dict:
    """Settle whether `model` honours `temperature`, preferring free metadata
    over paid inference. Returns a dict describing the method used, the
    conclusion, and the evidence behind it.
    """
    try:
        listing = fetch_json("https://openrouter.ai/api/v1/models")
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, ValueError) as exc:
        print(f"FAIL metadata fetch (models list): {type(exc).__name__}: {exc}")
        listing = None

    entry = None
    if listing is not None:
        for m in listing.get("data", []):
            if m.get("id") == model:
                entry = m
                break

    endpoints = None
    if entry is not None:
        canonical_slug = entry.get("canonical_slug")
        if canonical_slug:
            try:
                ep_payload = fetch_json(
                    f"https://openrouter.ai/api/v1/models/{canonical_slug}/endpoints")
                endpoints = ep_payload.get("data", {}).get("endpoints")
            except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, ValueError) as exc:
                print(f"FAIL metadata fetch (endpoints): {type(exc).__name__}: {exc}")

    supported_parameters = entry.get("supported_parameters") if entry else None
    metadata_is_conclusive = isinstance(supported_parameters, list)

    if metadata_is_conclusive:
        default_temp = (entry.get("default_parameters") or {}).get("temperature")
        honoured = "temperature" in supported_parameters
        per_endpoint = [
            {"provider": ep.get("provider_name"),
             "supported_parameters": ep.get("supported_parameters")}
            for ep in (endpoints or [])
        ]
        print(f"metadata: model-level supported_parameters={supported_parameters}")
        print(f"metadata: model-level default_parameters.temperature={default_temp!r}")
        for pe in per_endpoint:
            print(f"metadata: endpoint {pe['provider']} supported_parameters="
                  f"{pe['supported_parameters']}")
        return {
            "method": "metadata",
            "temperature_honoured": honoured,
            "evidence": {
                "model_supported_parameters": supported_parameters,
                "model_default_parameters_temperature": default_temp,
                "per_endpoint_supported_parameters": per_endpoint,
            },
        }

    # Metadata missing or ambiguous: settle it empirically. Two completions
    # at temperature=0 (should be near-identical if honoured) and two at
    # temperature=2 (should visibly diverge if honoured). Four short calls,
    # no more.
    print("metadata absent/ambiguous -- falling back to empirical temperature probe "
          "(4 short calls)")

    def sample(temp: float) -> str:
        r = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Name one animal. Reply with one word."}],
            temperature=temp,
        )
        return (r.choices[0].message.content or "").strip()

    temp0 = [sample(0), sample(0)]
    temp2 = [sample(2), sample(2)]
    print(f"empirical: temperature=0 outputs={temp0}")
    print(f"empirical: temperature=2 outputs={temp2}")
    low_identical = temp0[0] == temp0[1]
    high_diverges = temp2[0] != temp2[1]
    honoured = low_identical and high_diverges
    return {
        "method": "empirical",
        "temperature_honoured": honoured,
        "evidence": {"temperature_0_outputs": temp0, "temperature_2_outputs": temp2},
    }


temperature_check = check_temperature_support(MODEL)

print("\n" + json.dumps(results, indent=2))
print("\n" + json.dumps(temperature_check, indent=2))

lines = ["# API surface", "",
         f"Probed model: `{MODEL}`. Regenerate with `python scripts/probe_api.py`.", "",
         "| attempt | works | notes |", "|---|---|---|"]
for r in results:
    if r["attempt"] == "chat: temperature=0.7" and r["ok"]:
        note = "call did not error -- see 'Temperature support' section below before trusting this"
    else:
        note = "" if r["ok"] else r["error"].replace("|", "/")[:160]
    lines.append(f"| `{r['attempt']}` | {'yes' if r['ok'] else 'no'} | {note} |")

lines += ["", "## Temperature support", ""]
if temperature_check["temperature_honoured"]:
    lines.append(
        f"**Conclusion: HONOURED**, per {temperature_check['method']} evidence below. "
        f"`temperature` can be set explicitly (e.g. 0.7) and is expected to take effect.")
else:
    lines.append(
        f"**Conclusion: NOT HONOURED**, per {temperature_check['method']} evidence below. "
        "The `chat: temperature=0.7` attempt in the table above returns `ok: true`, but "
        "that only shows OpenRouter/the upstream provider *accepted* the request -- not "
        "that the parameter had any effect. `temperature` is silently dropped before it "
        "reaches the model. Downstream code must not treat 0.7 as a controlled variable: "
        "any sampling variation observed in the reliability recipe comes from the model's "
        "own default (undocumented) temperature, not from a value this codebase sets. This "
        "must be stated as a limitation in the study write-up.")
lines += ["", "Evidence:", "", "```json",
          json.dumps(temperature_check, indent=2), "```", ""]

lines += ["## Raw attempt results", "", "```json", json.dumps(results, indent=2), "```", ""]

with open("docs/api-surface.md", "w", encoding="utf-8") as fh:
    fh.write("\n".join(lines))
print("\nwrote docs/api-surface.md")
