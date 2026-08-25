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
