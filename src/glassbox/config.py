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

# Output caps, derived from the 60 real answers in output/runs rather than
# guessed. Observed single-call maxima, counting reasoning tokens (which are
# billed and capped as output): 4,633 for plain, 9,775 for structured, 30,792
# for think_longer. These give roughly 1.6x headroom over the relevant maximum.
#
# The cap is keyed to reasoning effort, not to the recipe: output volume tracks
# how hard the model thinks, and this stays correct if another recipe later uses
# raised effort.
#
# Without a cap the client requests the model's maximum (65,536), so OpenRouter
# reserves against that on every call. That makes a small balance unusable and
# leaves per-call cost unbounded in a study that will make thousands of calls.
MAX_OUTPUT_TOKENS_BASELINE = 16_000
MAX_OUTPUT_TOKENS_RAISED = 48_000

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
