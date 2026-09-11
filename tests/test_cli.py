import json
from pathlib import Path

from doc_harness.cli import run_smoke


def test_smoke_writes_record_and_prediction(tmp_path: Path):
    output = tmp_path / "run.jsonl"
    prediction = tmp_path / "prediction.json"
    run_smoke(
        question="What is shown?",
        document_id="doc.pdf",
        image_paths=[str(tmp_path / "page.png")],
        output_path=output,
        prediction_path=prediction,
    )
    assert output.exists()
    assert json.loads(prediction.read_text())[0]["response"] == "stub answer"
