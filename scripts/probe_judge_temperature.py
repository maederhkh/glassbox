"""Confirm the LEXam judge models honour `temperature`.

Task 5 grades every judge at `JUDGE_TEMPERATURE = 0.0` (config.py). That is only
meaningful if each judge model actually *honours* the parameter -- Task 1 found
that gpt-5-mini silently drops it. This script settles the question for the
three judge models using the same free, no-inference metadata endpoint Task 1
used (`GET https://openrouter.ai/api/v1/models` plus the per-model endpoints
listing), and appends the result to docs/api-surface.md as a distinct,
regenerable section.

Deliberately metadata-only: unlike probe_api.py's fallback for ambiguous
models, this script makes *no* inference calls. Judge-model inference costs
real money and would prove nothing about whether OpenRouter/the upstream
provider honours a parameter -- a call can silently accept and drop `temperature`
without erroring (exactly the gpt-5-mini finding). If metadata is missing or
ambiguous for a judge, that is reported as inconclusive rather than resolved
empirically.

Run any time to refresh: `python scripts/probe_judge_temperature.py`.
"""

import json
import urllib.error
import urllib.request

from glassbox.config import JUDGE_MODELS, JUDGE_TEMPERATURE

API_SURFACE_PATH = "docs/api-surface.md"
SECTION_MARKER = "## Judge temperature support"


def fetch_json(url: str):
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.load(resp)


def check_model(model: str, listing: dict) -> dict:
    """Metadata-only temperature check for one judge model. No inference calls."""
    entry = None
    for m in listing.get("data", []):
        if m.get("id") == model:
            entry = m
            break

    if entry is None:
        print(f"FAIL metadata: model {model!r} not found in models listing")
        return {
            "model": model,
            "method": "metadata",
            "conclusive": False,
            "temperature_honoured": None,
            "evidence": {"error": "model not found in /api/v1/models listing"},
        }

    supported_parameters = entry.get("supported_parameters")
    default_temp = (entry.get("default_parameters") or {}).get("temperature")
    metadata_is_conclusive = isinstance(supported_parameters, list)

    per_endpoint: list[dict] = []
    canonical_slug = entry.get("canonical_slug")
    if canonical_slug:
        try:
            ep_payload = fetch_json(
                f"https://openrouter.ai/api/v1/models/{canonical_slug}/endpoints")
            endpoints = ep_payload.get("data", {}).get("endpoints") or []
            per_endpoint = [
                {"provider": ep.get("provider_name"),
                 "supported_parameters": ep.get("supported_parameters")}
                for ep in endpoints
            ]
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, ValueError) as exc:
            print(f"FAIL metadata fetch (endpoints) for {model}: {type(exc).__name__}: {exc}")

    if not metadata_is_conclusive:
        print(f"metadata: {model} -- supported_parameters missing/ambiguous, "
              f"inconclusive (no inference fallback per Task 5 constraint)")
        return {
            "model": model,
            "method": "metadata",
            "conclusive": False,
            "temperature_honoured": None,
            "evidence": {
                "model_supported_parameters": supported_parameters,
                "model_default_parameters_temperature": default_temp,
                "per_endpoint_supported_parameters": per_endpoint,
            },
        }

    honoured = "temperature" in supported_parameters
    print(f"metadata: {model} -- model-level supported_parameters={supported_parameters}")
    print(f"metadata: {model} -- model-level default_parameters.temperature={default_temp!r}")
    for pe in per_endpoint:
        print(f"metadata: {model} -- endpoint {pe['provider']} "
              f"supported_parameters={pe['supported_parameters']}")
    return {
        "model": model,
        "method": "metadata",
        "conclusive": True,
        "temperature_honoured": honoured,
        "evidence": {
            "model_supported_parameters": supported_parameters,
            "model_default_parameters_temperature": default_temp,
            "per_endpoint_supported_parameters": per_endpoint,
        },
    }


try:
    listing = fetch_json("https://openrouter.ai/api/v1/models")
except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, ValueError) as exc:
    print(f"FAIL metadata fetch (models list): {type(exc).__name__}: {exc}")
    raise SystemExit(1)

results = [check_model(model, listing) for model in JUDGE_MODELS]

print("\n" + json.dumps(results, indent=2))

lines = [SECTION_MARKER, "",
         f"`JUDGE_TEMPERATURE = {JUDGE_TEMPERATURE}` (config.py) is only meaningful if "
         "each judge honours the parameter. Checked via the free "
         "`https://openrouter.ai/api/v1/models` metadata endpoint (plus each model's "
         "`/endpoints` listing) -- no inference calls, matching Task 1's method for "
         "gpt-5-mini. Regenerate with `python scripts/probe_judge_temperature.py`.", ""]

for r in results:
    if r["conclusive"]:
        verdict = "HONOURED" if r["temperature_honoured"] else "NOT HONOURED"
    else:
        verdict = "INCONCLUSIVE (metadata missing/ambiguous, no inference fallback taken)"
    lines.append(f"### `{r['model']}`")
    lines.append("")
    lines.append(f"**{verdict}**")
    lines.append("")

lines += ["Evidence:", "", "```json", json.dumps(results, indent=2), "```", ""]

with open(API_SURFACE_PATH, encoding="utf-8") as fh:
    existing = fh.read()

if SECTION_MARKER in existing:
    existing = existing.split(SECTION_MARKER, 1)[0]

new_content = existing.rstrip("\n") + "\n\n" + "\n".join(lines)

with open(API_SURFACE_PATH, "w", encoding="utf-8") as fh:
    fh.write(new_content)

print(f"\nwrote {API_SURFACE_PATH}")
