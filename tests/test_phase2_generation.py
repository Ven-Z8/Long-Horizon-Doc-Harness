from pathlib import Path

import pytest
from pydantic import BaseModel, ConfigDict

from doc_harness.core.contracts import ModelRequest, StageFailure
from doc_harness.models.runner import build_labeled_content
from doc_harness.models.structured import RawAttempt, generate_structured


class Payload(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    answer: str


def test_nonconsecutive_page_ids_label_their_images():
    content = build_labeled_content([4, 17], ["img-a", "img-b"], "question")
    assert content[0] == {"type": "text", "text": "Document page_id=4"}
    assert content[1] == {"type": "image", "image": "img-a"}
    assert content[2] == {"type": "text", "text": "Document page_id=17"}
    assert content[3] == {"type": "image", "image": "img-b"}
    assert content[-1] == {"type": "text", "text": "question"}


def test_labeled_content_rejects_duplicate_page_ids():
    with pytest.raises(ValueError, match="unique"):
        build_labeled_content([1, 1], ["a", "b"], "question")


def test_structured_generation_repairs_once_and_preserves_attempts():
    request = ModelRequest(
        document_id="d.pdf",
        question="Q",
        page_ids=[0],
        image_paths=["p0.png"],
        prompt="answer",
        config_hash="config",
    )
    responses = iter(
        [
            RawAttempt(raw_response="not json", finish_reason="eos", output_tokens=2),
            RawAttempt(raw_response='{"answer":"yes"}', finish_reason="eos", output_tokens=4),
        ]
    )
    result = generate_structured(
        request,
        Payload,
        lambda _: next(responses),
        lambda error, draft: f"repair {error}: {draft}",
        max_repairs=1,
    )
    assert result.payload == {"answer": "yes"}
    assert len(result.attempts) == 2
    assert result.attempts[0].failure is not None


def test_structured_generation_does_not_repair_twice():
    request = ModelRequest(
        document_id="d.pdf", question="Q", page_ids=[], image_paths=[], prompt="answer", config_hash="config"
    )
    result = generate_structured(
        request,
        Payload,
        lambda _: RawAttempt(raw_response="no", finish_reason="eos"),
        lambda error, draft: "repair",
        max_repairs=1,
    )
    assert result.payload is None
    assert len(result.attempts) == 2
    assert result.failure is not None


def test_length_terminated_json_is_not_accepted():
    request = ModelRequest(
        document_id="d.pdf", question="Q", page_ids=[], image_paths=[], prompt="answer", config_hash="config"
    )
    result = generate_structured(
        request,
        Payload,
        lambda _: RawAttempt(raw_response='{"answer":"yes"}', finish_reason="length"),
        lambda error, draft: "repair",
        max_repairs=0,
    )
    assert result.payload is None
    assert result.failure is not None
