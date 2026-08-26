# Phase 1 outcomes

What Phase 1 built, what it measured about the measuring instrument itself, and what
Phase 2 must do before the pipeline is built on top.

Phase 1 deliberately built **the measuring instrument and the control conditions**, not
the four-stage pipeline. 129 tests, 40 commits, 60 real answers on disk.

---

## 1. Findings of record

These are properties of the model, the benchmark and the graders, established by
measurement rather than assumption. Several contradict what the design spec originally
assumed; the spec has been corrected in place with the retractions visible.

### Temperature is not settable on `openai/gpt-5-mini`

`temperature` is absent from `supported_parameters` on all four upstream OpenRouter routes
and `default_parameters.temperature` is `null`. **A call passing it does not error — the
parameter is silently dropped.**

Consequence: repeated-run variation comes from the model's own default sampling, not from a
chosen temperature. The reliability measurement is still valid, but the claim is "variation
under default sampling". `seed` **is** supported and is deliberately left unset, or repeated
runs would be identical and the reliability measure would read zero by construction.

*This was nearly missed. The first probe only checked that the call did not error, which
cannot distinguish "accepted" from "silently dropped".*

### The holistic LEXam judge saturates

Recipe 1 scored **0.850** across the 20 dev questions: median 0.90, range 0.60–0.90,
distribution {0.6: 1, 0.8: 7, 0.9: 12}. Nothing reached 1.0. On the same questions the
**expert reference answers self-score 0.89** — a four-point gap against an observed judge
maximum of 0.9.

The holistic score therefore has very little power to separate a competent model answer
from an examiner's own answer, and correspondingly little to separate two recipes. It is
retained for comparability with published work; it cannot carry a result.

For context, 0.850 sits far above LEXam's published 60.3 for this model and above the 70.2
top score on their whole leaderboard. Part of that gap is expected — their figure covers a
test set that is ~83% German, and their prompt instructs Swiss law and German-style
citation, which fits English international-law questions poorly.

### The checklist metric discriminates; the sub-score layer does not

Checklist coverage took **18 distinct values across 20 answers**. That is the instrument the
study runs on.

The four sub-scores did not: `issue_spotting` and `conclusion` returned **one** distinct
value each across all 20 answers; `rule_recall` and `rule_application` two each. Ruled out
as a code fault — the same client, same run and same code path produced 18 distinct coverage
values, and the constant is not the parse-failure sentinel. The cause is the prompt: the four
criteria were written as unanchored yes/no questions, which invite a ceiling answer for any
competent answer.

**This layer is unusable as built and must be fixed before recipes are compared** — spec
§7.1 layer 2 exists to give the stage-level attribution that makes error localisation
possible. Two candidate fixes are in §4 below.

### The minimum-score ensemble is a lower bound, not a consensus

Wherever the three judges disagreed, the ensemble equalled the harshest, almost always
`qwen3-32b`. A reported score means "no judge scored this above X", not "the judges agreed
on X". This is LEXam's protocol, reproduced rather than second-guessed — and it earned its
keep: taking the minimum is what surfaced the mismatched reference pair below, where a mean
would have muted it.

### LEXam's dev split contains a mismatched question/answer pair

`0f6dd9e7` asks whether an RRP clause and a five-year exclusive-purchasing obligation
violate **Art. 101 TFEU**. Its reference answer analyses **Art. 102 TFEU** abuse of
dominance and refusal to supply, cites *United Brands*, and names a party "Y" absent from
the question. The documents are unrelated.

Worth reporting to the LEXam authors, and worth stating in the write-up as a benchmark
data-quality observation. Replaced from the pre-recorded reserve list; the substitution is
recorded inside `data/dev_20.json`.

Two further questions scored low against themselves for benign reasons and were **kept**:
`74b136af` is a marking scheme rather than a model answer, and `f164c0eb` carries the
examiner's own mark allocations as bare digits mid-prose.

### Reference answers come in three forms, not two

Model answers; marking schemes addressed to the grader; and **mark-annotated model answers**
where per-item point values survived text extraction as bare digits. Any component reading
reference text must ignore those digits rather than treat them as content.

### The drafted checklists are finer than a real examiner's marking

370 points across 20 questions, median 18.5, range 6–37. The calibration came from
`f164c0eb`, whose reference carries the examiner's own marks: **11 marks across 9 items** for
a 581-word answer, against 18 drafted points — roughly 1.6× finer, with the excess
concentrated in descriptive material.

Two consequences: it does **not** bias the comparison between recipes (correlated points move
together), but it **does** weight questions unevenly, so **coverage is computed per question
first, then averaged — never pooled over raw points**.

Operating rule for verification: *a point should correspond to something a grader would award
a mark for.* Merging over-split points is part of verification, not a separate step.

### Points are a resolution gain, not extra sample size

An earlier draft of the spec claimed the checklist turns 30 questions into "150–250 scoring
units, which is where your statistical power comes from". **That was wrong and is retracted.**
Points within a question are heavily correlated. **The unit of analysis is the question.**
What the checklist buys is a far more resolved per-question measurement — decisive given the
holistic judge's saturation, but not extra n.

### The compute control works, and better than the design assumed

| Recipe | Input | Output | Reasoning | Total | Cost |
|---|---|---|---|---|---|
| 1 Plain | 9,215 | 46,071 | 14,400 | 69,686 | $0.1232 |
| 2 Structured | 18,155 | 99,723 | 11,392 | 129,270 | $0.2268 |
| 3 Think-longer | 18,155 | 132,779 | 288,448 | 439,382 | $0.8470 |

Reasoning tokens rose **11,392 → 288,448 (25×)** on a byte-identical prompt, for a **3.40×**
total-token ratio. Input tokens are identical between Recipes 2 and 3, independently
corroborating their shared prompt hash.

Two things follow. Recipe 3 will likely **exceed** the pipeline's eventual spend, which
permits a stronger claim than the spec anticipated — *single-pass with more compute than the
pipeline still did not beat it* — rather than approximate matching. And the extra spend is
almost entirely hidden thinking: visible output rose only 33%, so Recipe 3 thinks harder
without writing longer, keeping answer length out of the comparison as a confound.

Also note **structure is not free in tokens**: Recipe 2 costs roughly double Recipe 1 in a
single call. If the pipeline ties Recipe 2, the honest statement is not "the instructions were
free" but "the instructions cost half as much as the pipeline and bought the same thing".

### Recipe 1 replicated itself within 0.8%

Re-run two weeks apart on the same prompt: 69,121 → 69,686 tokens, $0.1222 → $0.1232. Useful
incidental evidence that token differences between recipes reflect the recipe rather than
run-to-run noise.

### Blinding regexes written against imagined output did not match real output

Both blinding patterns failed on data already on disk:

- `_STAGE_MARKER` required ASCII punctuation after the number. GPT-5-mini at high effort
  writes `Step 1 — Issues identified` with an **em dash**, so all four step headings reached
  the grader in **15 of 20** Recipe 3 answers and **none** in the other arms — an asymmetric
  leak in the pair that must differ only in effort.
- `_SECTION_HEADING` required a markdown `#`, which the model used in **0 of 60** answers. It
  was dead code. Bare heading lines were leaking in plain 14/20, structured 10/20,
  think_longer 3/20 — symmetric, so not a validity threat, but the grader was seeing structure
  across the board.

Both fixed; residual leak 0/60. **General lesson: derive any output-parsing pattern from real
samples, and take its fixtures verbatim from them.**

---

## 2. Two known gaps

**The sections-bearing grading path has never met real model output.** Substantially closed
by local analysis at zero cost: the `final_answer`-preferred branch fires on 40/40
schema-recipe results, the sections-join fallback was never needed, and blinded text equals
`final_answer.strip()`. What remains — how the three judges behave on that text — is a
machinery check for the funded grading run, not a design risk.

**Judge spend is unmetered.** Score rows record the *answer's* tokens, not the judges'. Every
judge call sends question + full reference + candidate to three models — the most expensive
calls in the project — and none is counted. This is why the credit wall arrived unseen. The
usage data is already captured at the `Completion` boundary and discarded one frame later, so
the fix is mechanical. **It should land before any further grading, including Phase 1's own
remaining grading.**

---

## 3. Open decisions for the author

**Fix the sub-score layer.** Either rewrite the criteria as countable quantities, or tag each
checklist point with its criterion (issue / rule / application / conclusion) and derive each
sub-score as coverage within that tag. The second inherits the discrimination that
demonstrably works, costs no extra API calls, and maps exactly onto the four pipeline stages.

**Decide it before hand-verifying the checklists**, not merely before the comparison — the
tagged option rebuilds the checklists with a category per point, so verifying first means
verifying twice.

**Verify the 20 checklists.** `data/checklists/review/` holds one markdown file per question.
Confirm each point is in the reference, and merge over-split points. Until the lists are
frozen, every score is provisional.

**Pre-register** the inclusion criterion and the six hypotheses before drawing the evaluation
set from the `test` split. Git timestamps are not credible pre-registration; OSF or AsPredicted
are.

---

## 4. Phase 2 backlog

### Before any further grading

1. **Meter judge spend** — thread `Usage` through `JudgeVerdict` / `SubScores` /
   `ChecklistVerdict`, add judge token and cost columns per row, add a budget stop. Prices for
   all three judges are already in `config.PRICING_PER_MTOK`.
2. **Enforce the §7.4 freeze.** `data/checklists/` is gitignored for licensing, so "committed
   and never modified" is currently impossible. Commit a manifest of per-checklist SHA-256
   hashes — hashes redistribute no LEXam content — and verify them before scoring. Without
   this the no-tuning guarantee rests on nothing.
3. **Link score rows to what they scored.** Resume keys on `(question_id, recipe)` only, so
   after `--rerun` replaces an answer, or after verification changes a checklist, re-grading
   skips the row and re-serves stale numbers as if final. Record the answer's `prompt_hash`
   and a checklist content hash per row and refuse on mismatch. Add `--out` instead of a
   hard-coded filename.
4. **Restrict the retry policy** to retryable errors. It currently retries all exceptions five
   times with up to 60s backoff, so a 402 wall multiplies into hours across a batch.
5. **Per-row fault tolerance in grading**, and consolidate the duplicated JSON extractor —
   `subscores.py` re-implements `checklist._extract_json` without its no-brace guard, so a
   grader reply lacking `{` kills the batch *after* three judge calls were paid.

### Before the pipeline

6. **Guard the manipulated variable.** `--effort` defaults to baseline for every recipe and
   `EFFORT_RAISED` is dead code, so `--recipe think_longer` without `--effort high` silently
   produces Recipe 2 wearing Recipe 3's label. Resume does not check effort either. Map
   recipe→default effort and extend the resume check to `reasoning_effort`.
7. **Extract the four step instructions as shared constants.** The pipeline's stage prompts
   must match Recipe 2's wording exactly, but that wording is inline in `STRUCTURED_PROMPT`.
   Copy-paste is how drift starts.
8. **Harden `parse_case_file`** — the pipeline puts model JSON through it four times per
   question instead of once. An unpaired quote in preamble prose loses a case file; a
   non-list `amendments` raises an uncaught `TypeError` *before the result is saved*, losing a
   paid completion; the string-awareness logic is unpinned.
9. **Decide amendment-drop visibility.** An unrecognised section label now drops that one
   amendment instead of destroying the case file — the right trade — but silently, which
   shrinks the measured self-correction rate invisibly.
10. **Add a `stage` field to `Amendment`** if per-stage attribution is wanted directly rather
    than reconstructed by diffing stage outputs. Cheaper to decide now than after the format
    is persisted.
11. **Commit a round-trip test for non-empty `stages`** — currently only `stages=[]` is
    covered, and the pipeline is what fills it.
12. **Build the shuffled, condition-stripped export** for §7.5's human validation. Immaterial
    for stateless per-call LLM judging, required for the human pass.

### When `eval_30` lands

13. Parameterise the `dev_20`-hardcoded error hint; give a stale cache an actionable message
    rather than a bare `KeyError`; make the question-replacement write atomic (a mid-failure
    currently leaves committed files against a stale cache **and the tool cannot re-run to
    repair**); persist `--full-validate`'s output as a machine-readable artefact rather than
    console text.

---

## 5. Spend

Recipe answers: **$1.20** for all 60, auditable from the persisted results. Total project
spend reached the account's $11 — the balance went on grading, the calibration gate, sample
validation, and two discarded recipe runs. Phase 2 and 3 involve roughly 1,000 recipe calls
plus several thousand grading calls: **budget $30–60**, and land the judge metering first.
