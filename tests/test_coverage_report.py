from typing import Tuple
import importlib
import types
import pytest

# ⬇️ adjust this import if your file lives elsewhere.
# For example, if the script is at src/ece461/test_summary.py,
# keep it as below. If it's tools/collect_summary.py, import that instead.
import ece461.test_summary as ts


# ---------- unit tests for helpers ----------

def test_last_int_basic() -> None:
    text = "x 12 passed y 3 failed z"
    assert ts.last_int(text, r"(\d+)\s+passed") == 12
    assert ts.last_int(text, r"(\d+)\s+failed") == 3
    assert ts.last_int(text, r"(\d+)\s+skipped") == 0  # no match -> 0


def test_extract_counts_parses_typical_summary() -> None:
    text = "29 passed, 1 failed, 0 errors, 3 skipped in 4.56s"
    p, f, e, s = ts.extract_counts(text)
    assert (p, f, e, s) == (29, 1, 0, 3)


def test_extract_counts_handles_missing_fields() -> None:
    text = "10 passed in 0.12s"  # no failed/errors/skipped
    p, f, e, s = ts.extract_counts(text)
    assert (p, f, e, s) == (10, 0, 0, 0)


def test_extract_coverage_from_total_line() -> None:
    text = """
---------- coverage: platform linux, python 3.11 ----------
Name                            Stmts   Miss  Cover
---------------------------------------------------
src/ece461/foo.py                  10      1    90%
TOTAL                              10      1    90%
"""
    assert ts.extract_coverage(text) == "90%"


def test_extract_coverage_fallback_when_no_total() -> None:
    # No line starting with TOTAL; fallback to last % at line end
    text = "some log...\noverall coverage: 78%\n"
    assert ts.extract_coverage(text) == "78%"


def test_extract_coverage_none_returns_na() -> None:
    assert ts.extract_coverage("no percents here") == "N/A"


# ---------- integration: main() formatting (without running real pytest) ----------

def test_main_formats_output_happy_path(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    fake_out = """
collected 33 items

... test output ...

=========================== short test summary info ===========================
29 passed, 1 failed, 0 errors, 3 skipped in 1.23s

---------- coverage: platform linux, python 3.11 ----------
Name                            Stmts   Miss  Cover
---------------------------------------------------
src/ece461/bar.py                  20      5    75%
TOTAL                              20      5    75%
"""
    # Monkeypatch ts.run to avoid executing pytest; feed fake output
    monkeypatch.setattr(ts, "run", lambda cmd: (0, fake_out))

    # Call main and capture print
    rc = ts.main()
    captured = capsys.readouterr().out.strip()
    assert rc == 0
    assert captured == "29/33 test cases passed. 75% line coverage achieved."


def test_main_formats_output_with_na_coverage(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    fake_out = "10 passed in 0.12s\n(no coverage table)"
    monkeypatch.setattr(ts, "run", lambda cmd: (0, fake_out))
    _ = ts.main()
    captured = capsys.readouterr().out.strip()
    # 10 of 10 (no fails/errors/skips), coverage N/A
    assert captured == "10/10 test cases passed. N/A line coverage achieved."