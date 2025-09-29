from typing import Any, Callable, ContextManager, Dict, List, Tuple
from pathlib import Path
import types
import pytest
from ece461.url_file_parser import ModelLinks

from ece461.metricCalcs import metrics as met
from huggingface_hub.errors import (
    EntryNotFoundError,
    RepositoryNotFoundError,
    HfHubHTTPError,

)

# ----------------- Helpers to safely patch & restore the registry -----------------
def swap_registry(temp: dict[str, Callable[..., Any]]) -> ContextManager[None]:
    """Context manager to temporarily replace REGISTRY."""
    class _Ctx:
        _orig: Dict[str, Callable[..., Any]]

        def __enter__(self) -> None:
            _Ctx._orig = dict(met.REGISTRY)
            met.REGISTRY.clear()
            met.REGISTRY.update(temp)
        def __exit__(self, *exc: Any) -> None:
            met.REGISTRY.clear()
            met.REGISTRY.update(_Ctx._orig)
    return _Ctx()

# -------------------------------- normalize() / run_metrics() ---------------------

def test_normalize_tuple_and_dict_cases() -> None:
    # (score, latency)
    mv1 = met.normalize((0.5, 12.0))
    assert mv1["ok"] is True and mv1["score"] == 0.5 and mv1["latency_ms"] == 12.0

    # ({details}, latency) — e.g., size metric that returns a details dict
    mv2 = met.normalize(({"extra": "x"}, 7.0))
    assert mv2["ok"] is True and mv2["score"] is None
    assert mv2["details"]["extra"] == "x" and mv2["latency_ms"] == 7.0

    # Unsupported type -> ok=False
    mv3 = met.normalize(0.9)  # type: ignore[arg-type]
    assert mv3["ok"] is False and mv3["score"] is None

def test_run_metrics_include_exclude_and_empty() -> None:
    # Metrics now receive a model object via keyword arg (model=...)
    def m1(model: Any) -> tuple[float, float]:
        return (0.2, 4.0)
    def m2(model: Any) -> tuple[dict[str, Any], float]:
        return ({"k": "v"}, 3.0)

    dummy_model = types.SimpleNamespace(model_id="dummy", code=None)

    with swap_registry({"m1": m1, "m2": m2}):
        out = met.run_metrics(dummy_model, include=["m1"])
        assert set(out.keys()) == {"m1"} and out["m1"]["score"] == 0.2

        out2 = met.run_metrics(dummy_model, exclude=["m2"])
        assert set(out2.keys()) == {"m1"}

        # empty selection -> {}
        out3 = met.run_metrics(dummy_model, include=["does_not_exist"])
        assert out3 == {}

# ----------------------- README / Model Card fetch & prompts ----------------------

def test_fetch_readme_content_success(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    readme = tmp_path / "README.md"
    readme.write_text("hello world", encoding="utf-8")

    monkeypatch.setattr(met, "hf_hub_download", lambda **_: str(readme))
    assert "hello" in met.fetch_readme_content("x/y")

def test_fetch_readme_content_entry_not_found(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(met, "hf_hub_download", lambda **_: (_ for _ in ()).throw(EntryNotFoundError()))
    assert met.fetch_readme_content("x/y") == ""

def test_fetch_readme_content_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(met, "hf_hub_download", lambda **_: (_ for _ in ()).throw(HfHubHTTPError("boom")))
    assert met.fetch_readme_content("x/y") == ""

def test_fetch_model_card_content_success(monkeypatch: pytest.MonkeyPatch) -> None:

    class MC:
        def __init__(self) -> None: self.content = "card text"
    monkeypatch.setattr(met, "ModelCard", types.SimpleNamespace(load=lambda *_a, **_k: MC()))
    assert met.fetch_model_card_content("x/y") == "card text"


def test_fetch_model_card_content_repo_not_found(monkeypatch: pytest.MonkeyPatch) -> None:
    def _load(*_a: Any, **_k: Any) -> Any:
        raise RepositoryNotFoundError()
    monkeypatch.setattr(met, "ModelCard", types.SimpleNamespace(load=_load))
    assert met.fetch_model_card_content("x/y") == ""

def test_prompt_builders() -> None:
    assert len(met.build_ramp_up_prompt("x")) > 10
    assert len(met.build_performance_prompt("x")) > 10
    assert len(met.build_license_prompt("x")) > 10

# ----------------------------- Ramp-up / Performance ------------------------------
def test_ramp_up_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(met, "fetch_readme_content", lambda _m: "README")
    monkeypatch.setattr(met.llm_api, "query_llm", lambda _p: '{"ramp_up_score":0.7}')
    model = types.SimpleNamespace(model_id="x/y", code=None)
    score, lat = met.calculate_ramp_up_metric(model)
    assert 0.69 <= score <= 0.71 and lat >= 0.0

def test_ramp_up_empty_readme(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(met, "fetch_readme_content", lambda _m: "")
    model = types.SimpleNamespace(model_id="x/y", code=None)
    score, _ = met.calculate_ramp_up_metric(model)
    assert score == 0.0

def test_ramp_up_bad_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(met, "fetch_readme_content", lambda _m: "README")
    monkeypatch.setattr(met.llm_api, "query_llm", lambda _p: "nonsense")
    model = types.SimpleNamespace(model_id="x/y", code=None)
    score, _ = met.calculate_ramp_up_metric(model)
    assert score == 0.0

def test_ramp_up_clamp(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(met, "fetch_readme_content", lambda _m: "README")
    model = types.SimpleNamespace(model_id="x/y", code=None)
    monkeypatch.setattr(met.llm_api, "query_llm", lambda _p: '{"ramp_up_score": 9.9}')
    score_hi, _ = met.calculate_ramp_up_metric(model)
    assert score_hi == 1.0
    monkeypatch.setattr(met.llm_api, "query_llm", lambda _p: '{"ramp_up_score": -2}')
    score_lo, _ = met.calculate_ramp_up_metric(model)
    assert score_lo == 0.0

def test_performance_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(met, "fetch_model_card_content", lambda _m: "MODEL CARD")
    monkeypatch.setattr(met.llm_api, "query_llm", lambda _p: '{"performance_score":0.45}')
    model = types.SimpleNamespace(model_id="x/y", code=None)
    score, lat = met.calculate_performance_metric(model)
    assert 0.44 <= score <= 0.46 and lat >= 0.0

def test_performance_bad_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(met, "fetch_model_card_content", lambda _m: "MODEL CARD")
    monkeypatch.setattr(met.llm_api, "query_llm", lambda _p: 'not-json')
    model = types.SimpleNamespace(model_id="x/y", code=None)
    score, _ = met.calculate_performance_metric(model)
    assert score == 0.0

# ------------------------------ Size / Hardware scores ---------------------------

def test_get_model_weight_size_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    class R:
        status_code = 200
        def raise_for_status(self) -> None: ...
        def json(self) -> list[dict[str, Any]]:
            return [
                {"path": "a.bin", "type": "file", "size": 1_000_000},
                {"path": "dir", "type": "directory", "size": 0},
            ]
    monkeypatch.setattr(met.requests, "get", lambda *_a, **_k: R())
    total_mb = met.get_model_weight_size("x/y")
    assert 0.99 < total_mb < 1.01


def test_get_model_weight_size_bad_shapes(monkeypatch: pytest.MonkeyPatch) -> None:
    class R1:
        status_code = 200
        def raise_for_status(self) -> None: ...
        def json(self) -> dict[str, Any]:
            return {}  # not a list
    monkeypatch.setattr(met.requests, "get", lambda *_a, **_k: R1())
    with pytest.raises(ValueError):
        met.get_model_weight_size("x/y")

    class R2:
        status_code = 200
        def raise_for_status(self) -> None: ...
        def json(self) -> list[dict[str, Any]]:
            return [{"path": "dir", "type": "directory", "size": 0}]
    monkeypatch.setattr(met.requests, "get", lambda *_a, **_k: R2())
    with pytest.raises(ValueError):
        met.get_model_weight_size("x/y")

def test_calculate_size_metric_ok_and_error(monkeypatch: pytest.MonkeyPatch) -> None:
    # OK path: function returns ({scores...}, latency_ms)
    monkeypatch.setattr(met, "get_model_weight_size", lambda _m: 150.0)
    model = types.SimpleNamespace(model_id="x/y", code=None)
    details, lat = met.calculate_size_metric(model)
    assert isinstance(details, dict)
    assert set(details.keys()) == {"raspberry_pi", "jetson_nano", "desktop_pc", "aws_server"}
    assert all(0.0 <= v <= 1.0 for v in details.values())
    assert lat >= 0.0

    # Error path: get_model_weight_size raises -> ValueError from wrapper
    def _boom(_m: str) -> float: raise RuntimeError("fail")
    monkeypatch.setattr(met, "get_model_weight_size", _boom)
    with pytest.raises(ValueError):
        met.calculate_size_metric(model)

def test_hardware_thresholds() -> None:
    # hit different branches
    s1 = met.calculate_hardware_compatibility_scores(25.0)     # tiny
    s2 = met.calculate_hardware_compatibility_scores(75.0)     # mid
    s3 = met.calculate_hardware_compatibility_scores(12000.0)  # huge
    for s in (s1, s2, s3):
        assert set(s.keys()) == {"raspberry_pi", "jetson_nano", "desktop_pc", "aws_server"}
        assert all(0.0 <= v <= 1.0 for v in s.values())

# -------------------------------- Code quality -----------------------------------


def test_calculate_code_quality_hit(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    class P:
        def __init__(self) -> None:
            self.stdout = "Your code has been rated at 7.50/10"
    monkeypatch.setattr(met.subprocess, "run", lambda *a, **k: P())
    model = types.SimpleNamespace(model_id="x/y", code=str(tmp_path))
    score, lat = met.calculate_code_quality(model)
    assert 0.74 <= score <= 0.76 and lat >= 0.0

def test_calculate_code_quality_miss(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    class P:
        def __init__(self) -> None:
            self.stdout = "no rating here"
    monkeypatch.setattr(met.subprocess, "run", lambda *a, **k: P())
    model = types.SimpleNamespace(model_id="x/y", code=str(tmp_path))
    score, _ = met.calculate_code_quality(model)
    assert score == 0.0

# -------------------------------- License metric ---------------------------------


def test_calculate_license_metric_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    class MC:
        def __init__(self) -> None: self.content = "MIT License"
    monkeypatch.setattr(met, "ModelCard", types.SimpleNamespace(load=lambda *_a, **_k: MC()))
    monkeypatch.setattr(
        met.llm_api, "query_llm",
        lambda _p: '{"license_score":0.8,"detected_license":"MIT","compatible_with_lgpl_2_1":true,"confidence_0to1":0.9,"rationale":"ok"}',
    )
    model = types.SimpleNamespace(model_id="x/y", code=None)
    score, lat = met.calculate_license_metric(model)
    assert 0.79 <= score <= 0.81 and lat >= 0.0

def test_calculate_license_metric_bad_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    class MC:
        def __init__(self) -> None: self.content = "Custom"
    monkeypatch.setattr(met, "ModelCard", types.SimpleNamespace(load=lambda *_a, **_k: MC()))
    monkeypatch.setattr(met.llm_api, "query_llm", lambda _p: "nonsense")
    model = types.SimpleNamespace(model_id="x/y", code=None)
    score, _ = met.calculate_license_metric(model)
    assert score == 0.0

def test_dataset_and_code_score_with_both_urls():
    """Test the dataset_and_code_quality metric with both URLs"""
    dataset_url = "https://huggingface.co/datasets/bookcorpus"
    code_url = "https://github.com/google-research/bert"
   
    model = ModelLinks("hi", dataset_url, code_url, "google-research/bert")
    score, latency_ms = met.calculate_dataset_and_code_score(model)
   
    # Basic validation
    assert 0.0 <= score <= 1.0, f"Score {score} should be between 0 and 1"
    assert latency_ms >= 0.0, f"Latency {latency_ms} should be non-negative"
    assert isinstance(score, float), "Score should be a float"
    assert isinstance(latency_ms, (int, float)), "Latency should be numeric"


def test_dataset_and_code_score_with_no_urls():
    """Test the dataset_and_code_quality metric with no URLs"""
    model = ModelLinks("hi", None, None, "google-research/bert")
    score, latency_ms = met.calculate_dataset_and_code_score(model)
   
    assert score == 0.0, "Score should be 0.0 when no URLs provided"
    assert latency_ms >= 0.0, f"Latency {latency_ms} should be non-negative"


def test_dataset_and_code_score_partial_urls():
    """Test with only one URL provided"""
    # Only dataset URL
    model = ModelLinks("hi", "https://huggingface.co/datasets/squad", None, "google-research/bert")
    score1, _ = met.calculate_dataset_and_code_score(model)
    assert 0.0 <= score1 <= 1.0, "Should get partial score with only dataset"
   
    # Only code URL  
    model = ModelLinks("hi", None, "https://github.com/google-research/bert", "google-research/bert")
    score2, _ = met.calculate_dataset_and_code_score(model)
    assert 0.0 <= score2 <= 1.0, "Should get partial score with only code"


def test_dataset_quality_with_url():
    """Test the dataset_quality metric with a HuggingFace dataset"""
    model = ModelLinks("hi", "https://huggingface.co/datasets/squad", None, "google-research/bert")
    score, latency_ms = met.calculate_dataset_quality(model)
   
    # Basic validation
    assert 0.0 <= score <= 1.0, f"Score {score} should be between 0 and 1"
    assert latency_ms >= 0.0, f"Latency {latency_ms} should be non-negative"
    assert isinstance(score, float), "Score should be a float"
    assert isinstance(latency_ms, (int, float)), "Latency should be numeric"


def test_dataset_quality_with_no_url():
    """Test the dataset_quality metric with no URL"""
    model = ModelLinks("hi", None, None, "google-research/bert")
    score, latency_ms = met.calculate_dataset_quality(model)
   
    assert score == 0.0, "Score should be 0.0 when no URL provided"
    assert latency_ms >= 0.0, f"Latency {latency_ms} should be non-negative"


def test_dataset_quality_different_platforms():
    """Test dataset quality with different platform URLs"""
    # HuggingFace dataset
    model = ModelLinks("hi", "https://huggingface.co/datasets/glue", None, "google-research/bert")
    hf_score, _ = met.calculate_dataset_quality(model)
   
    # GitHub dataset  
    model = ModelLinks("hi", "https://github.com/google-research/text-to-text-transfer-transformer", None, "google-research/bert")
    gh_score, _ = met.calculate_dataset_quality(model)
   
    # Both should be valid scores
    assert 0.0 <= hf_score <= 1.0
    assert 0.0 <= gh_score <= 1.0


def test_normalize_error_path() -> None:
    bad = met.normalize(None)  # type: ignore[arg-type]
    assert bad["ok"] is False and bad["score"] is None


def test_run_metrics_filters(monkeypatch: pytest.MonkeyPatch) -> None:
    # register two dummy metrics taking model kwarg
    def m1(*, model: Any) -> Tuple[float, float]: return (0.1, 2.0)
    def m2(*, model: Any) -> Tuple[Dict[str, Any], float]: return ({"x": 1}, 3.0)
    model = types.SimpleNamespace(model_id="org/name", code=None)

    # swap the registry
    orig = dict(met.REGISTRY)
    met.REGISTRY.clear(); met.REGISTRY.update({"m1": m1, "m2": m2})
    try:
        only = met.run_metrics(model, include=["m1"])
        assert set(only) == {"m1"} and only["m1"]["score"] == 0.1

        not_m2 = met.run_metrics(model, exclude=["m2"])
        assert set(not_m2) == {"m1"}

        empty = met.run_metrics(model, include=["nope"])
        assert empty == {}
    finally:
        met.REGISTRY.clear(); met.REGISTRY.update(orig)


# ------------------------------ metrics: size / hardware -----------------

def test_get_model_weight_size_variants(monkeypatch: pytest.MonkeyPatch) -> None:
    # OK list
    class R1:
        def raise_for_status(self) -> None: ...
        def json(self) -> List[Dict[str, Any]]:
            return [
                {"path":"a.bin", "type":"file", "size": 1_000_000},
                {"path":"b", "type":"directory", "size": 0}
            ]
    monkeypatch.setattr(met.requests, "get", lambda *a, **k: R1())
    mb = met.get_model_weight_size("x/y")
    assert 0.9 < mb < 1.1

    # not a list -> ValueError
    class R2:
        def raise_for_status(self) -> None: ...
        def json(self) -> Dict[str, Any]:
            return {"weird": True}
    monkeypatch.setattr(met.requests, "get", lambda *a, **k: R2())
    with pytest.raises(ValueError):
        met.get_model_weight_size("x/y")

    # list but no files -> ValueError
    class R3:
        def raise_for_status(self) -> None: ...
        def json(self) -> List[Dict[str, Any]]:
            return [{"path":"dir", "type":"directory", "size": 0}]
    monkeypatch.setattr(met.requests, "get", lambda *a, **k: R3())
    with pytest.raises(ValueError):
        met.get_model_weight_size("x/y")


def test_calculate_size_metric_ok_and_error(monkeypatch: pytest.MonkeyPatch) -> None:
    model = types.SimpleNamespace(model_id="x/y", code=None)
    monkeypatch.setattr(met, "get_model_weight_size", lambda _m: 150.0)
    details, lat = met.calculate_size_metric(model)
    assert isinstance(details, dict) and lat >= 0
    # error propagation -> ValueError
    def boom(_m: str) -> float: raise RuntimeError("x")
    monkeypatch.setattr(met, "get_model_weight_size", boom)
    with pytest.raises(ValueError):
        met.calculate_size_metric(model)


def test_hardware_threshold_edges() -> None:
    # hit boundaries
    for val in (50.0, 100.0, 200.0, 500.0, 1000.0, 2000.0, 5000.0, 10000.0, 12345.0):
        s = met.calculate_hardware_compatibility_scores(val)
        assert set(s) == {"raspberry_pi","jetson_nano","desktop_pc","aws_server"}
        assert all(0.0 <= v <= 1.0 for v in s.values())

# ------------------------------ metrics: dataset/code/dataset_quality ----

def test_dataset_and_code_score_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    # none provided -> score 0
    model_none = types.SimpleNamespace(model_id="x/y", code=None, dataset=None)
    s0, _ = met.calculate_dataset_and_code_score(model_none)
    assert s0 == 0.0

    # ok json
    model = types.SimpleNamespace(model_id="x/y", code="https://gh", dataset="https://hf")
    monkeypatch.setattr(met.llm_api, "query_llm", lambda p: '{"dataset_code_score":0.73}')
    s1, _ = met.calculate_dataset_and_code_score(model)
    assert 0.72 <= s1 <= 0.74

    # bad json / extraction -> 0
    monkeypatch.setattr(met.llm_api, "query_llm", lambda p: 'not-json')
    s2, _ = met.calculate_dataset_and_code_score(model)
    assert s2 == 0.0


def test_dataset_quality_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    # none -> 0
    model_none = types.SimpleNamespace(model_id="x/y", dataset=None, code=None)
    s0, _ = met.calculate_dataset_quality(model_none)
    assert s0 == 0.0

    # ok json
    model = types.SimpleNamespace(model_id="x/y", dataset="https://datasets", code=None)
    monkeypatch.setattr(met.llm_api, "query_llm", lambda p: '{"dataset_quality_score":0.66}')
    s1, _ = met.calculate_dataset_quality(model)
    assert 0.65 <= s1 <= 0.67

    # exception from LLM -> 0
    def bad(_p: str) -> str: raise RuntimeError("boom")
    monkeypatch.setattr(met.llm_api, "query_llm", bad)
    s2, _ = met.calculate_dataset_quality(model)
    assert s2 == 0.0


# ------------------------------ metrics: code_quality no-code --------------

def test_code_quality_no_code() -> None:
    model = types.SimpleNamespace(model_id="x/y", code=None)
    score, lat = met.calculate_code_quality(model)
    assert score == 0.0 and lat >= 0.0

