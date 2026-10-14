import pytest
from pydantic import ValidationError

from src.contracts import Claim, RunRecord, Signal
from src.examples import EXAMPLES, example_claim, example_signal


@pytest.mark.parametrize("name", list(EXAMPLES))
def test_example_round_trips_through_json(name):
    obj = EXAMPLES[name]()
    again = type(obj).model_validate_json(obj.model_dump_json())
    assert again == obj


def test_confidence_must_be_between_0_and_1():
    data = example_claim().model_dump()
    data["confidence"] = 1.5
    with pytest.raises(ValidationError):
        Claim(**data)


def test_unknown_signal_name_is_rejected():
    data = example_signal().model_dump()
    data["name"] = "astrology"
    with pytest.raises(ValidationError):
        Signal(**data)


def test_run_record_nests_everything():
    run = EXAMPLES["run"]()
    assert isinstance(run, RunRecord)
    assert run.efce.status == "conflict_unresolved"
    assert run.findings[0].claims[0].evidence_ids
