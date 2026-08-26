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
# The trailing punctuation class started as ASCII-only `[:.\-]`. Running normalise()
# over all 60 on-disk dev_20 results showed GPT-5-mini at high effort writes
# "Step 1 — Issues identified" ... "Step 4 — Conclusion" with an em dash
# (U+2014), which that class never matched -- so 15/20 think_longer answers carried
# the stage scaffolding straight through to the grader (0/20 in plain/structured).
# Widened to the whole general-punctuation dash block (U+2010 HYPHEN through U+2015
# HORIZONTAL BAR, which includes U+2014 EM DASH and U+2013 EN DASH) so a differently
# generated dash doesn't reopen the same leak. The punctuation is still mandatory
# (not `?`), so prose that merely begins "Step 1 of the analysis is straightforward"
# -- no separator at all -- is left untouched.
_STAGE_MARKER = re.compile(
    r"^\s*(stage|step)\s*\d+\s*(output)?\s*[:.\-‐-―]\s*",
    re.IGNORECASE | re.MULTILINE,
)
# Originally required a markdown '#' heading marker. Checking against the same 60
# on-disk results found it never fired once: the model never writes '#' headings --
# it writes bare ("Issue"), bulleted ("- Application:"), numbered ("1. Issues",
# "4. Conclusion"), or all-caps ("OVERALL CONCLUSION") section names instead. Widened
# to accept an optional bullet/number/hash marker (or none at all) and an optional
# "overall " prefix before "conclusion". The `$` anchor is unchanged, so a heading
# word is only stripped when it is the *entire* line -- "- Rule/principle: ZGB
# requires ..." or "2. Applicable rules and theories (...)" (heading word followed by
# real content on the same line) still survive untouched.
_SECTION_HEADING = re.compile(
    r"^\s*(?:#{1,6}|[-*•]|\d+[.)])?\s*(?:overall\s+)?"
    r"(issues?|rules?|elements?|application|subsumption|conclusion|"
    r"analysis|final answer)\s*:?\s*$",
    re.IGNORECASE | re.MULTILINE,
)
_BLANK_RUN = re.compile(r"\n{3,}")
_SPACE_RUN = re.compile(r"[ \t]{2,}")


def normalise(result: RecipeResult) -> str:
    # `final_answer` first, sections only as a fallback. Every recipe is graded on
    # the answer an examiner would actually read.
    #
    # The reverse precedence is a validity bug, and a scheduled one: CaseFile.to_sections()
    # puts the full prose answer INSIDE the sections dict alongside issues, rules,
    # application and conclusion. Joining the values would hand the grader the analysis
    # AND an answer restating it - roughly double Recipe 1's content volume, for exactly
    # the recipes under test, awarded by the component whose whole purpose is to stop
    # recipe-identifying differences reaching the grader.
    #
    # The sections fallback still matters: a schema-following recipe whose output fails
    # to parse sets sections=None and puts raw text in final_answer, so that path already
    # takes the first branch. The fallback covers a parse that succeeded without a
    # final_answer.
    text = (result.final_answer or "").strip()
    if not text and result.sections:
        text = "\n\n".join(v.strip() for v in result.sections.values() if v and v.strip())

    # Stage marker before section heading, not the reverse: a real "Step N — Issues"
    # line has the heading word as the *entire* remainder after the dash, so
    # stripping the stage prefix first can unmask a now-bare "Issues" line
    # underneath -- which the heading pass must still get a chance to catch.
    text = _FENCE.sub("", text)
    text = _STAGE_MARKER.sub("", text)
    text = _SECTION_HEADING.sub("", text)
    text = _SPACE_RUN.sub(" ", text)
    text = _BLANK_RUN.sub("\n\n", text)
    return text.strip()
