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


# Minor: nothing above pins the empty-list -> None coercion on its own field;
# deleting it from parse_case_file leaves every test above still green.
def test_coerces_an_empty_issues_list_to_none():
    cf = parse_case_file('{"issues": []}', "q1")
    assert cf.issues is None


CORRECTED_PAYLOAD = PAYLOAD.replace(
    '"conclusion": "The shop is liable."',
    '"conclusion": "The shop is not liable."',
)


def test_prefers_the_last_of_two_json_blocks():
    # A model that drafts then corrects itself, or echoes an example before the
    # real payload, leaves the good object last.
    text = (f"```json\n{PAYLOAD}\n```\n"
            f"On reflection, that is wrong:\n"
            f"```json\n{CORRECTED_PAYLOAD}\n```")
    cf = parse_case_file(text, "q1")
    assert cf.conclusion == "The shop is not liable."


def test_tolerates_a_stray_brace_before_the_payload():
    text = "The schema uses {} for empty objects.\n" + PAYLOAD
    cf = parse_case_file(text, "q1")
    assert cf.conclusion == "The shop is liable."


def test_normalises_the_findings_section_alias_to_application():
    # The case file's own third field is named `findings`; a model recording
    # an amendment against the section it just wrote there naturally echoes
    # that field name even though the amendment vocabulary calls it
    # `application`.
    text = ('{"conclusion": "c", "amendments": [{"section": "findings", '
            '"change": "revised an element", "reason": "clarified facts"}]}')
    cf = parse_case_file(text, "q1")
    assert cf.amendments[0].section == "application"


def test_drops_an_unrecognised_amendment_section_without_failing_the_case_file():
    text = ('{"conclusion": "c", "amendments": ['
            '{"section": "bogus", "change": "x", "reason": "y"}, '
            '{"section": "rules", "change": "a", "reason": "b"}]}')
    cf = parse_case_file(text, "q1")
    assert cf.conclusion == "c"
    assert len(cf.amendments) == 1
    assert cf.amendments[0].section == "rules"


def test_coerces_null_amendments_to_an_empty_list():
    cf = parse_case_file('{"conclusion": "c", "amendments": null}', "q1")
    assert cf.amendments == []
