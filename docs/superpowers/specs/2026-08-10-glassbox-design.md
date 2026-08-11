# glassbox — Design

**Date:** 2026-08-10
**Status:** Design approved in outline; build scope not yet chosen.

---

## 1. Research question

> How does explicit multi-stage decomposition affect the quality, reliability, failure
> modes and computational cost of LLM legal reasoning, compared with single-pass
> end-to-end generation?

The study does not assume decomposition helps. It is designed so that every possible
outcome is informative.

### Why this is not already answered

Multi-stage "agent" pipelines are widely built and widely assumed to help. Two
explanations are almost never ruled out:

1. **The instructions, not the pipeline.** Telling a model to follow IRAC in one prompt
   costs nothing. If that captures the whole benefit, the pipeline is unnecessary.
2. **The compute, not the structure.** Four calls spend more tokens than one. Extra
   inference-time compute alone raises quality on hard reasoning tasks.

A two-arm design (single-pass vs pipeline) cannot separate these from decomposition
itself. This design adds one control arm for each.

A third, more current framing: modern reasoning models already decompose internally via
reasoning tokens. The live question is therefore not *"do stages help?"* but *"is
**externalised, inspectable** decomposition still worth anything now that models
decompose internally?"*

---

## 2. Design decisions

| # | Decision | Rationale |
|---|---|---|
| 1 | Independent repository, no shared code with `prompt2GDPR-v2` | Clean provenance for a controlled study |
| 2 | 30 English open questions from LEXam; expert answers as reference | No self-authored legal ground truth |
| 3 | GPT-5-mini for every recipe and every stage | The manipulation is structure, not model capability |
| 4 | Four recipes: Plain, Structured, Think-longer, Pipeline | Two controls rule out the two rival explanations |
| 5 | Pipeline = 4 stages (issues / rules+elements / application / conclusion) | Minimum decomposition that supports stage-level error attribution |
| 6 | Case-file relay for information flow | Every stage has what it needs; no shared reasoning traces |
| 7 | Amendments permitted but logged | Turns self-correction into measured data, not an assumption |
| 8 | JSON envelope, natural-language values; identical schema in Structured and Pipeline | Isolates call boundaries as the only difference |
| 9 | Three grading layers, blind to condition | Structure must not be able to impress the grader |
| 10 | Human validation of the grader on 20 answers | The grader's trustworthiness is not assumed |
| 11 | 3 repeated samples + 2 reworded prompts | Measures both kinds of instability |
| 12 | Deliberate damage to Stage 1, three types | Causal, not correlational, propagation evidence |
| 13 | Hypotheses registered before running | Separates a study from a demonstration |

---

## 3. Dataset

**Source:** LEXam open-question subset ([paper](https://arxiv.org/abs/2505.12864),
[leaderboard](https://lexam-benchmark.github.io/),
[code](https://github.com/LEXam-Benchmark/LEXam),
[data](https://huggingface.co/datasets/LEXam-Benchmark/LEXam)).

**Fields, as actually released:** `question`, `answer` (expert reference), `course`,
`language` (`en`/`de`), `area` (Criminal / Public / Private / Interdisciplinary),
`jurisdiction` (Swiss / International / Generic), `year`, `id`.

There is **no `guidance` column** in the Hugging Face release, despite the paper
describing one. Nothing therefore needs to be withheld from the model, and the earlier
concern about leaking the reasoning plan to the single-pass recipes does not arise.

**Splits.** `open_question` ships `dev` (300 rows, 80 English) and `test` (2,541 rows, 436
English). The **development set is drawn from `dev`** and the **evaluation set from
`test`**, so the two are disjoint by construction and the pipeline cannot be tuned on the
questions it is later measured on.

### Inclusion criterion

LEXam's English open questions are heterogeneous. Of 436 in the test split, 99 have
reference answers under 50 words and are factual recall — *"What are the requirements to
take the bar exam?"* → *"J.D."* A staged legal-reasoning pipeline is not applicable to
such questions; including them would let question-type mismatch drive the result.

The study's population is therefore defined, and pre-registered, as:

| Filter | Reason |
|---|---|
| `language == "en"` | The prototype is English-only |
| `answer` ≥ 150 words | Proxy for "requires analysis rather than recall" |
| `question` ≥ 50 words | Proxy for "has a fact pattern or substantive prompt" |
| `area != "Interdisciplinary"` | Holds Legal Theory and Legal Sociology — moral philosophy and social theory essays, which have no issues, rules or elements to decompose |

This yields **22 eligible in `dev`** and **143 in `test`**.

This is population definition, not cherry-picking: the criterion is objective, fixed
before the draw, applied identically to every condition, and the number of questions it
keeps is reported. Sampling is simple random **within** the criterion, with a fixed seed;
the resulting area, jurisdiction and course mix is reported rather than forced.

**Replacement rule.** A drawn question that is still recall or essay on inspection is
replaced by the next question in the shuffled order, and the number of replacements is
reported.

**Two residual heterogeneities**, deliberately *labelled rather than filtered*, because
both serve hypothesis H3:

- Some questions are **fact-pattern subsumption problems** ("A ordered 20,000 civilians
  killed; assess A's liability under the Rome Statute"), others are **doctrinal discussion
  questions** ("to whom does the board of directors owe a duty?"). Each drawn question is
  labelled, and the label is a recorded covariate.
- Some reference answers are **model answers**, others are **marking schemes** ("the answer
  should at a minimum raise the issue of…"). The checklist builder must handle both. Marking
  schemes are in fact already close to checklists.

**Provenance.** `scripts/select_questions.py` records the dataset revision, criterion,
seed and drawn ids in `data/<name>.json`. Only ids and metadata are versioned; question and
answer text is cached under `data/cache/` and gitignored, so the repository never
redistributes LEXam's content.

---

## 4. Model configuration

| Setting | Value |
|---|---|
| Provider | OpenRouter, for both the system model and all three judges |
| Model | GPT-5-mini (LEXam open-question score 60.3) |
| Temperature | **Not settable.** See below. |
| Reasoning effort | Held constant across Plain / Structured / Pipeline; raised only in Think-longer |
| Seeds | Supported by the model, but deliberately left unset — see below |

**Temperature cannot be controlled on this model.** Confirmed 2026-08-10 against
OpenRouter's model metadata: `temperature` is absent from `supported_parameters` on all
four upstream routes for `openai/gpt-5-mini`, and `default_parameters.temperature` is
`null`. A call passing `temperature` does **not** error — the parameter is silently
dropped. The client therefore omits it.

Two consequences for the study, both of which must be stated rather than glossed:

- Repeated-run variation (§8.1) comes from the model's **own default sampling**, not from
  a temperature we chose. The measurement is still valid — it observes how stable the
  system is in the configuration anyone would actually deploy — but the claim is "variation
  under default sampling", not "variation at temperature 0.7".
- `seed` **is** supported, and is deliberately left unset. Fixing it would make repeated
  runs identical and drive the reliability measure to zero by construction.

**Why mid-tier.** Top models score ~70 under LEXam's current grading, so a top-tier model
leaves little headroom; a null result would be uninterpretable — indistinguishable from
"there was no room to show an effect." A weak model risks failing to hold the four-stage
format at all, which would turn the study into a measure of instruction-following.

**Why not GPT-4o as the system model.** GPT-4o is one of LEXam's three graders. GPT-5-mini
shares a vendor family with one of the three, which the minimum-score ensemble rule dampens
and the independent checklist metric bypasses. Stated as a limitation, not hidden.

---

## 5. The four recipes

All four receive the same question and facts and must answer that same question.

Recipes 2, 3 and 4 additionally share an identical output schema, so the manipulation
between them is only the number of calls and the amount of thinking. Recipe 1 is
deliberately open-format — that is what makes it the natural baseline. Blind normalisation
(§7.3) converts every recipe's output into a common flat format before grading, so no
recipe gains or loses from its output shape.

### Recipe 1 — Plain
One call, one strong open legal-reasoning prompt. Deliberately **not** weakened. This is
what the model does when asked well.

### Recipe 2 — Structured
One call. Contains the **same four-part instructions** and requires the **same output
schema** as the pipeline.

This is the study's most important control. If Recipes 2 and 4 score alike, the finding is
that decomposition's value is the instructions, not the separate calls — an honest and
publishable negative result.

### Recipe 3 — Think longer
Recipe 2 with the reasoning-effort setting raised until total token spend approximately
matches the pipeline's.

This answers the compute objection without crippling either side. Exact token matching is
not attempted; instead the three single-call recipes trace a quality-versus-cost frontier
and the pipeline is checked against it (§9).

*Caveat:* this depends on an adjustable thinking budget. If the parameter proves too
coarse, the fallback is a cruder length-based approximation, which must be reported as
approximate.

### Recipe 4 — Pipeline
Four sequential calls. Detailed in §6.

---

## 6. Pipeline design

### 6.1 Stages

| Stage | Receives | Produces | Must not |
|---|---|---|---|
| **1 Issues** | question + facts | each legal issue, with one line on why it arises | name rules; reach any conclusion |
| **2 Rules** | case file | per issue: governing rule(s), split into required elements | apply anything to the facts; conclude |
| **3 Application** | case file | per element: whether it holds here, citing the specific facts | state the overall answer |
| **4 Conclusion** | case file | the conclusion and the full exam-style answer | add issues without logging the amendment |

The "must not" column is load-bearing. Without it there are not four stages, only four
opportunities to answer the whole question. **Violations are counted and reported** — how
often a stage reaches for a conclusion early is itself a finding about whether the model
can hold a partial legal analysis.

### 6.2 Information flow — case-file relay

There is one structured artifact. Each stage receives:

- the original question and facts, and
- every **committed section** written by earlier stages,

but **never** an earlier stage's internal reasoning or thinking tokens. Each stage writes
only its own section.

Rejected alternatives:

- **Strict chain** (previous stage's output only): Stage 4 would lose the rule statements,
  forcing Stage 3 to copy them forward; copying errors would then be indistinguishable
  from reasoning errors.
- **Full transcript** (everything, including reasoning): converges on single-pass with
  extra steps, washing out the contrast, and cost grows per stage.

Running case-file relay **and** strict chain as two arms — making information isolation a
measured variable — is a good second experiment, deliberately postponed.

### 6.3 Output schema

A small JSON envelope per stage whose **values are natural-language legal reasoning**.
Not bare keyword JSON (which flattens reasoning quality), not free prose (which cannot be
scored automatically or traced across stages).

Recipe 2 and Recipe 4 emit the **same schema**, so the only manipulation between them is
whether that schema is filled in one context or four.

### 6.4 Amendments

Any stage may add to or correct an earlier section, but must record each change in an
`amendments` list (what changed, where, why).

This is a *measurement*, not an extra component: it yields a **rescue rate** (later stage
fixes an earlier miss) and a **damage rate** (later stage breaks something that was
correct), and it is what makes the fault-injection study in §8 meaningful.

Rejected: a frozen case file (unrealistic, and invites the objection that propagation was
forced by design); mandatory review of earlier stages by every stage (smuggles a reflection
step into the first experiment); silent amendment (destroys error localisation, the
project's main contribution).

---

## 7. Evaluation

### 7.1 Three layers

1. **LEXam's official grader** — minimum-score ensemble of GPT-4o + DeepSeek-V3 +
   Qwen3-32B, as of their September 2025 update. Keeps results comparable to published
   work. All three are reached through OpenRouter, so the published protocol is used in
   full rather than approximated by a single judge.
2. **Per-criterion sub-scores** — issue spotting, rule recall, rule application,
   conclusion. These are LEXam's own validated criteria and they map directly onto the four
   pipeline stages, giving stage-level attribution without inventing a rubric.
3. **Checklist (primary diagnostic metric)** — each expert reference answer is split into
   atomic checkable points. Every answer is scored per point as *covered / partial /
   missed*, plus two flags: **contradicts the reference** and **invented rule or
   authority**.

### 7.2 Why the checklist matters

- **Statistical power.** 30 questions × ~5–8 points ≈ 150–250 scoring units instead of 30
  holistic scores. Item-level near-binary judgments are far more reliable than 1–10 quality
  ratings.
- **Propagation, measured.** The same point list is traced through all four stages: a point
  absent at Stage 1 and never recovered is propagation as a number, not a narrative.
- **Not self-authored ground truth.** No legal content is invented — the experts' answer is
  only split into its parts.

### 7.3 Blind grading

Before grading, every answer is normalised to the same flat format, stripped of anything
revealing which recipe produced it, and presented in shuffled order.

The justification comes from the author's own thesis finding: outputs can *look*
structurally complete and professional while containing substantive legal errors. Pipeline
output will look tidier. A grader that can see structure will reward it. Therefore the
grader must not see it.

### 7.4 Checklists are frozen before any system runs

An LLM drafts each point list from the expert answer; **the author verifies all 30 by
hand**; the lists are then committed and never modified. Freezing first removes any
possibility of tuning the rubric toward a recipe.

### 7.5 Human validation of the grader

LEXam's own validation (single-GPT-4o era) reported roughly *r* = 0.70, quadratic-weighted
κ ≈ 0.49 and mean absolute error ≈ 2 points on a 10-point scale against human experts.
That is moderate agreement, and no published agreement figure is known for the newer
three-judge ensemble.

Therefore: **10 questions × 2 recipes = 20 answers** — Recipe 1 (Plain) and Recipe 4
(Pipeline), the two ends of the comparison, so the validation set spans the widest range of
answer quality. Hand-scored by the author using the same checklist, **blind** (shuffled,
condition-stripped) and **before** looking at the automatic grades. Agreement is reported.

Human scores never replace or mix with automatic scores. They calibrate them. If agreement
is poor, the grader prompt or judge configuration is revised and the same 20 answers are
re-checked. Discovering a broken grader on 20 answers is cheap; discovering it after
analysis is not.

Effort estimate: roughly half a day.

---

## 8. Reliability and error propagation

### 8.1 Two kinds of instability

- **Sampling instability** — same prompt, 3 runs, under the model's default sampling
  (temperature is not settable on GPT-5-mini; see §4, and `seed` is left unset so runs can
  differ). The **mean is the headline score** (less noisy than a single run); the spread is
  the stability measure.
- **Prompt-wording sensitivity** — 2 reworded prompts of identical meaning. For the
  pipeline all four stage prompts are reworded together, so the manipulation is comparable.

The second is the more novel angle and follows directly from the thesis finding that labels
changed with prompt wording. **Hypothesis: fixed narrow stages are less sensitive to
wording, so decomposition may improve reliability without improving accuracy.**

**Metric — flip rate.** A checklist point covered in one run and missed in another. Lower is
more reliable. Sharper than "the answers looked different."

5 runs × 30 questions × 4 recipes ≈ **1,050 model calls**.

### 8.2 Deliberate damage (fault injection)

For each question, take a pipeline run that scored well — **a run the pipeline handled
correctly** — and produce three damaged Stage 1 outputs, then re-run Stages 2–4:

| Damage type | Edit to Stage 1 | Ability tested |
|---|---|---|
| **Take away** | delete a correct issue | noticing something is *missing* |
| **Add junk** | insert an irrelevant issue | *ignoring* nonsense |
| **Wrong law** | replace a correct rule with a plausible wrong one | catching the *wrong rule* |

~270 additional calls.

Recorded per run: recovered / partly recovered / propagated; whether the amendment log
registered it; and the change in score and checklist coverage.

**Why this is the strongest part of the study.** Pipelines routinely *claim* that later
stages catch earlier mistakes; almost nobody tests it. Observing that "Stage 1 missed an
issue and so did the final answer" is only a correlation — those may simply have been the
hardest questions, where every stage fails independently. Planting the error in a question
the pipeline handled *well* removes that doubt entirely.

It adds no architectural component and no new variable to the comparison. It is
instrumentation, not engineering — the "don't overengineer" principle governs architecture,
not measurement.

---

## 9. Cost accounting

Recorded per run: number of calls, input tokens, output tokens, reasoning tokens, cost,
wall-clock time.

Reported as a **quality-versus-cost curve**, not a single matched comparison. The three
single-call recipes trace the frontier obtainable by simply spending more tokens without
decomposing; the question is whether the pipeline sits **above** that frontier. If it sits
on it, the honest finding is that decomposition bought nothing that extra tokens in one
call would not have bought.

---

## 10. Hypotheses, registered before running

| | Prediction |
|---|---|
| **H1** | Pipeline beats Plain on checklist coverage |
| **H2** | Pipeline ≈ Structured — most of the gain is the instructions, not the separate calls |
| **H3** | Pipeline beats Structured on multi-issue questions but not on single-issue ones (overcomplication) |
| **H4** | Pipeline has a lower flip rate than Plain under rewording — reliability gain without accuracy gain |
| **H5** | Pipeline rejects added junk more often than it notices a deleted issue |
| **H6** | Pipeline costs 3–5× Plain per question |

H2 predicts against the author's own architecture. Hits and misses are reported equally.

**Analysis plan.** Paired comparisons at the question level (each question contributes one
observation per recipe, so grader bias on a difficult question affects all recipes alike).
Checklist coverage analysed at point level. H3 examined descriptively only — 30 questions
cannot support a moderation test, and this is stated rather than glossed.

**Framing.** 30 questions is a pilot. One deliberate deliverable is an **effect-size
estimate for powering a full study**, which is a stronger and more honest contribution than
claiming 30 questions settle the question.

---

## 11. Build order

The pipeline is roughly a sixth of the code. The bulk is the harness and the measuring
instrument.

| # | Component | Note |
|---|---|---|
| 1 | Data loading + frozen sample (`scripts/select_questions.py`) | **done** — `data/dev_20.json` |
| 2 | Runner skeleton + Recipe 1 (Plain), one question end to end | |
| 3 | Grader with LEXam's official score only → **calibration gate** | see below |
| 4 | Checklist builder + checklist grading | author verifies all 30 by hand |
| 5 | Recipes 2 and 3 | |
| 6 | Recipe 4 — the pipeline | |
| 7 | Reliability runs (3 samples + 2 rewordings) | |
| 8 | Damage tool | |
| 9 | Analysis and report | |

**Build the measuring instrument before the thing being measured.** If the pipeline is
built first there is no way to tell whether it is any good, and stage prompts get tuned by
intuition. By step 6 the pipeline can be evaluated honestly on its first run.

**The calibration gate (step 3).** Run LEXam's official prompt with their official grader on
a defined slice of their data and check that their published score is approximately
reproduced. If it is not, stop and find the bug — everything downstream is worthless until
it reproduces.

This is a genuine replication run, **not** a comparison of this study's 30-question average
against the leaderboard. Their published figure covers a test set that is ~83% German, and
models score better on English, so the two numbers are not comparable.

**Author-owned tasks** (cannot be delegated): verifying the 30 checklists, and the 20 blind
hand-scores.

---

## 12. Open items to resolve at build time

1. ~~Does LEXam's protocol feed `guidance` to the model or only to the grader?~~
   **Resolved 2026-08-10:** there is no `guidance` column in the released dataset, so
   nothing needs withholding. If their evaluation code sources guidance elsewhere, note it.
2. **Grading scale.** The paper refers to both 0–10 and 0–100 in different places. Pin down
   the actual scale and the exact minimum-score ensemble rule from their code.
3. ~~English question distribution across `area` and `jurisdiction`.~~
   **Resolved 2026-08-10:** 143 eligible English questions in `test`
   (Private 76 / Public 56 / Criminal 11; International 129 / Generic 16 / Swiss 6, across
   19 courses). Ample for a random draw of 30.
4. **GPT-5-mini reasoning-effort values and pricing** — confirm the available settings support
   Recipe 3, and the total cost of ~1,300 calls.
5. **Contamination.** Unknown whether LEXam is in training data. Report baseline per-question
   scores and note any ceiling saturation as a finding rather than filtering questions, which
   would be cherry-picking.
6. **Build scope** — not yet chosen by the author.

---

## 13. Deliberately postponed

Retrieval of legal sources · reflection or critic stages · retries · different models per
stage · per-issue fan-out (decomposition by *content* rather than by *operation*) · the
3-stage variant (which would answer "why four stages?") · case-file relay vs strict chain as
two arms · German data and a possible BenGER collaboration · additional model families.

Stating these explicitly signals that the first version's simplicity is a design decision.
Each is a clean follow-up experiment; none belongs in the first, where every added component
becomes another variable competing to explain the result.

---

## 14. Why every outcome is informative

| If the result is… | What it establishes |
|---|---|
| Pipeline beats all three single-call recipes | Decomposition genuinely helps. Strong positive result. |
| Pipeline ≈ Structured | The benefit was the **instructions**, not the separate calls. Agent pipelines are overkill here — a valuable negative result. |
| Pipeline ≈ Think-longer | The benefit was **extra compute**, not structure. Rarely tested. |
| Pipeline scores worse | Decomposition overcomplicates legal questions. Not previously shown cleanly. |
| Same score, fewer flips | Decomposition improves **reliability** rather than accuracy — arguably the most interesting outcome, and it speaks directly to the thesis findings. |

Independently of which occurs, the fault-injection study yields numbers not currently
available for any legal-AI pipeline: how often a multi-step legal reasoning system repairs
its own mistakes, broken down by the kind of mistake.

---

## 15. Limitations to state up front

- 30 questions is a pilot; an effect-size estimate for a properly powered study is an
  intended deliverable.
- One model, one dataset, English subset only.
- The system model shares a vendor family with one of the three graders.
- Possible training-data contamination is unknown.
- Judge-versus-human agreement rests on 20 hand-scored answers.
- Recipe 3's compute matching is approximate, not exact.
- Temperature is not settable on the system model, so repeated-run variation reflects the
  model's default sampling rather than a chosen setting.
- The four stages mirror IRAC and the German *Gutachtenstil*; results may not transfer to
  legal traditions structured differently.

---

## 16. Longer-term direction

A clean English result becomes a case study to bring to the BenGER team: here is the
architecture, here is the measurement framework, here is what happened in English — does the
same architectural effect hold for German subsumption-based reasoning? Because the four
stages mirror the *Gutachtenstil* structure, the architecture transfers substantially
unchanged. LEXam's own German subset also offers a nearer-term path to the same question
without a new dataset.
