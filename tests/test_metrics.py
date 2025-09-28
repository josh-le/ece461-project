from typing import Any, Callable, ContextManager, Dict
from pathlib import Path
import types
import pytest

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