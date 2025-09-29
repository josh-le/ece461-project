from typing import Any
import json
import types
import pytest

def test_main_emits_valid_ndjson_for_two_models(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path
) -> None:
    # Import here so monkeypatching lands on the same module object
    from ece461 import main as mainmod

    # Ensure logging setup doesn't raise (your setup() requires LOG_FILE)
    monkeypatch.setenv("LOG_FILE", str(tmp_path / "cli.log"))
    monkeypatch.setenv("LOG_LEVEL", "0")  # off, creates/touches file only

    # Fake two models returned by the parser (bypass file IO)
    fake_models = [
        types.SimpleNamespace(model_id="org/modelA", code=None, dataset=None),
        types.SimpleNamespace(model_id="org/modelB", code=None, dataset=None),
    ]
    monkeypatch.setattr(mainmod, "parse_url_file", lambda _p: fake_models)

    # Stub metrics.run_metrics to return deterministic results
    def fake_run_metrics(m: Any) -> dict:
        if m.model_id.endswith("A"):
            return {
                "license": {"ok": True, "score": 0.9, "latency_ms": 12.3, "details": {}},
                "ramp_up": {"ok": True, "score": 0.7, "latency_ms": 4.4, "details": {}},
                "size": {
                    "ok": True, "score": None, "latency_ms": 9.9,
                    "details": {"raspberry_pi": 0.1, "jetson_nano": 0.2, "desktop_pc": 1.0, "aws_server": 1.0}
                },
                "code_quality": {"ok": True, "score": 0.8, "latency_ms": 1.1, "details": {}},
            }
        return {
            "license": {"ok": True, "score": 0.5, "latency_ms": 2.0, "details": {}},
            "ramp_up": {"ok": True, "score": 0.3, "latency_ms": 1.0, "details": {}},
            "size": {
                "ok": True, "score": None, "latency_ms": 2.2,
                "details": {"raspberry_pi": 0.0, "jetson_nano": 0.0, "desktop_pc": 0.6, "aws_server": 1.0}
            },
            "code_quality": {"ok": True, "score": 1.0, "latency_ms": 0.5, "details": {}},
        }

    monkeypatch.setattr(mainmod.met, "run_metrics", fake_run_metrics)

    # Make net score deterministic and fast
    monkeypatch.setattr(mainmod, "calculate_net_score", lambda d: (0.42, 7))

    # Provide CLI arg (your main reads sys.argv[1])
    monkeypatch.setattr(mainmod.sys, "argv", ["prog", "dummy.csv"])

    # Run
    rc = mainmod.main()
    assert rc == 0

    out = capsys.readouterr().out.strip().splitlines()
    assert len(out) == 2  # one JSON object per model

    # Validate each line is valid JSON with expected keys
    for line in out:
        obj = json.loads(line)
        assert obj["category"] == "MODEL"
        assert "name" in obj and "net_score" in obj and "net_score_latency" in obj
        # latency fields should be integers (your main rounds them)
        for k, v in obj.items():
            if "latency" in k:
                assert isinstance(v, int)
        # size_scores must be a dict with 4 keys (even if zeros)
        assert set(obj["size_score"].keys()) == {
            "raspberry_pi", "jetson_nano", "desktop_pc", "aws_server"
        }
