from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd

from src.components.evaluation.judge import Judge


def _client_returning(content: str):
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
    )
    client = Mock()
    client.chat.completions.create.return_value = response
    return client


def _comparison_frame():
    return pd.DataFrame(
        {
            "instruction": ["Question"],
            "reference": ["Reference"],
            "response": ["Fine-tuned answer"],
            "base_response": ["Base answer"],
        }
    )


def test_chat_json_returns_none_after_malformed_responses(monkeypatch):
    client = _client_returning("not valid JSON")
    monkeypatch.setattr("time.sleep", lambda _seconds: None)

    result = Judge()._chat_json(client, "system", "user")

    assert result is None
    assert client.chat.completions.create.call_count == 3


def test_pairwise_genuine_tie_is_counted_as_tie(monkeypatch):
    judge = Judge()
    monkeypatch.setattr(
        judge,
        "_chat_json",
        Mock(side_effect=[{"winner": "tie"}, {"winner": "tie"}]),
    )

    outcomes, summary = judge.judge_pairwise(Mock(), _comparison_frame())

    assert outcomes == ["tie"]
    assert summary["ties"] == 1
    assert summary["errors"] == 0
    assert summary["judged"] == 1


def test_pairwise_failed_call_is_not_counted_as_tie(monkeypatch):
    judge = Judge()
    monkeypatch.setattr(
        judge,
        "_chat_json",
        Mock(side_effect=[None, {"winner": "tie"}]),
    )

    outcomes, summary = judge.judge_pairwise(Mock(), _comparison_frame())

    assert outcomes == ["error"]
    assert summary["ties"] == 0
    assert summary["errors"] == 1
    assert summary["judged"] == 0


def test_judge_json_payload_validation():
    judge = Judge()

    assert judge.valid_absolute_payload({"score": 5})
    assert not judge.valid_absolute_payload({"score": 6})
    assert not judge.valid_absolute_payload([])
    assert judge.valid_pairwise_payload({"winner": "tie"})
    assert not judge.valid_pairwise_payload({"winner": "fine-tuned"})
