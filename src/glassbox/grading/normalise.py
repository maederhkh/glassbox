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
