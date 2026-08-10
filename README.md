# glassbox

**Does breaking legal reasoning into explicit steps actually make it better?**

A controlled study of whether explicit multi-stage decomposition improves large language
model legal reasoning, compared with single-pass end-to-end generation.

Single-pass generation is a black box: when the reasoning fails, you cannot see where.
A staged pipeline is a glass box — every intermediate step is inspectable. This project
tests whether that transparency comes with any actual gain in quality, reliability, or
error recovery, and what it costs.

## Status

**Design phase.** No implementation yet.

The full design is in
[`docs/superpowers/specs/2026-08-10-glassbox-design.md`](docs/superpowers/specs/2026-08-10-glassbox-design.md).

## The experiment in short

The same 30 English open-ended law-exam questions from
[LEXam](https://lexam-benchmark.github.io/), the same model (GPT-5-mini), four ways of
asking:

| Recipe | Calls | Purpose |
|---|---|---|
| **Plain** | 1 | Honest baseline — a strong open legal-reasoning prompt |
| **Structured** | 1 | Same four-part instructions and output schema as the pipeline, one call |
| **Think longer** | 1 | Structured, with the thinking budget raised to match pipeline token spend |
| **Pipeline** | 4 | Issues → rules and elements → application → conclusion |

The two middle recipes are controls. They exist to rule out the two explanations that
would otherwise make a positive result uninterpretable: *"the gain was just the
instructions"* and *"the gain was just more compute."*

## What gets measured

- **Quality** — LEXam's official grader, per-skill sub-scores, and a frozen checklist of
  the atomic points in each expert reference answer
- **Reliability** — repeated runs and reworded prompts, scored by *flip rate*
- **Error propagation** — Stage 1 outputs are deliberately damaged three ways, then
  Stages 2–4 re-run, to measure how often the pipeline repairs its own mistakes
- **Cost** — calls, tokens, money and time, reported as a quality-versus-cost curve

A negative result is a valid result. The design is built so that every possible outcome
is informative — see the outcome table in the spec.

## Related work by the same author

This is an independent project. It follows on from, but shares no code with,
`prompt2GDPR-v2` (an agentic GDPR Article 5(1)(b) compliance pipeline) and an earlier
master's thesis on single-prompt GDPR Article 5 assessment.
