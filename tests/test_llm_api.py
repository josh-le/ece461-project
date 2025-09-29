from ece461.API import llm_api
from typing import Any, Dict

import pytest

from ece461.API import llm_api
from ece461.metricCalcs import metrics as met
from ece461.metricCalcs.net_score import calculate_net_score

def test_query_llm() -> None:
    response = llm_api.query_llm("What is the best software engineering tool?")
    assert isinstance(response, str)
    assert len(response) > 0

def test_llm_api_missing_key(monkeypatch: pytest.MonkeyPatch) -> None:
    # GEN_AI_STUDIO_API_KEY not set -> ValueError
    monkeypatch.delenv("GEN_AI_STUDIO_API_KEY", raising=False)
    with pytest.raises(ValueError):
        llm_api.query_llm("hi")


def test_llm_api_non_200(monkeypatch: pytest.MonkeyPatch) -> None:
    class R:
        status_code = 401
        text = "nope"
        def json(self) -> Dict[str, Any]: return {}
    monkeypatch.setenv("GEN_AI_STUDIO_API_KEY", "x")
    monkeypatch.setattr(llm_api.requests, "post", lambda *a, **k: R())
    with pytest.raises(Exception):
        llm_api.query_llm("hi")


def test_llm_api_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    class R:
        status_code = 200
        def json(self) -> Dict[str, Any]:
            return {"choices":[{"message":{"content":"{\"k\":1}"}}]}
    monkeypatch.setenv("GEN_AI_STUDIO_API_KEY", "x")
    monkeypatch.setattr(llm_api.requests, "post", lambda *a, **k: R())
    out = llm_api.query_llm("prompt")
    assert out == "{\"k\":1}"
