from __future__ import annotations

from pathlib import Path
import builtins
import logging
import os
import pytest

from ece461.logging_setup import setup as setup_logging


# ---------- helpers ----------

@pytest.fixture(autouse=True)
def _clean_logging() -> None:
    """
    Reset logging between tests to avoid handler bleed-through.
    """
    logging.disable(logging.NOTSET)
    root = logging.getLogger()
    for h in list(root.handlers):
        try:
            h.flush()  # type: ignore[attr-defined]
        except Exception:
            pass
        try:
            h.close()  # type: ignore[attr-defined]
        except Exception:
            pass
        root.removeHandler(h)


def _flush_all_handlers() -> None:
    root = logging.getLogger()
    for h in list(root.handlers):
        try:
            h.flush()  # type: ignore[attr-defined]
        except Exception:
            pass


def _first_file_handler_path() -> Path:
    root = logging.getLogger()
    for h in root.handlers:
        base = getattr(h, "baseFilename", None)
        if isinstance(base, str):
            return Path(base)
    raise AssertionError("Expected a FileHandler configured by setup_logging().")


# ---------- tests ----------

def test_missing_LOG_FILE_env_exits(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    When LOG_FILE is not set, setup() should log an error and sys.exit(1).
    """
    monkeypatch.delenv("LOG_FILE", raising=False)
    monkeypatch.setenv("LOG_LEVEL", "1")
    with pytest.raises(SystemExit) as e:
        setup_logging()
    assert int(e.value.code) == 1


def test_LOG_FILE_points_to_directory_raises(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """
    If LOG_FILE points to a directory, setup() raises ValueError.
    """
    monkeypatch.setenv("LOG_FILE", str(tmp_path))  # directory, not file
    monkeypatch.setenv("LOG_LEVEL", "1")
    with pytest.raises(ValueError):
        setup_logging()


def test_open_append_failure_exits(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """
    If opening LOG_FILE for append fails, setup() logs error and sys.exit(1).
    """
    target = tmp_path / "blocked.log"
    monkeypatch.setenv("LOG_FILE", str(target))
    monkeypatch.setenv("LOG_LEVEL", "1")

    real_open = builtins.open

    def fake_open(file, mode="r", *a, **k):  # type: ignore[no-redef]
        if str(file) == str(target) and "a" in mode:
            raise OSError("permission denied")
        return real_open(file, mode, *a, **k)

    monkeypatch.setattr(builtins, "open", fake_open)
    with pytest.raises(SystemExit) as exc:
        setup_logging()
    assert int(exc.value.code) == 1


def test_level_off_touches_file_and_suppresses_output(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """
    LOG_LEVEL not in {'1','2'} (e.g., '0') should:
      - create/touch the file,
      - globally disable logging (no records written),
      - and return without configuring a handler.
    """
    log_path = tmp_path / "off.log"
    monkeypatch.setenv("LOG_FILE", str(log_path))
    monkeypatch.setenv("LOG_LEVEL", "0")

    setup_logging()

    assert log_path.exists()
    size_before = log_path.stat().st_size

    logging.getLogger("ece461.test.off").critical("boom")
    _flush_all_handlers()

    size_after = log_path.stat().st_size
    assert size_before == 0 and size_after == 0


def test_level_info_writes_info_not_debug(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """
    LOG_LEVEL=1 configures INFO level: INFO messages written; DEBUG filtered.
    """
    log_path = tmp_path / "info.log"
    monkeypatch.setenv("LOG_FILE", str(log_path))
    monkeypatch.setenv("LOG_LEVEL", "1")

    setup_logging()

    lg = logging.getLogger("ece461.test.info")
    lg.debug("debug-msg")
    lg.info("info-msg")

    _flush_all_handlers()
    configured = _first_file_handler_path()
    assert configured == log_path
    text = configured.read_text(encoding="utf-8")
    assert "INFO" in text and "info-msg" in text
    assert "debug-msg" not in text


def test_level_debug_writes_debug(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """
    LOG_LEVEL=2 configures DEBUG level: both DEBUG and INFO are written.
    """
    log_path = tmp_path / "dbg.log"
    monkeypatch.setenv("LOG_FILE", str(log_path))
    monkeypatch.setenv("LOG_LEVEL", "2")

    setup_logging()

    lg = logging.getLogger("ece461.test.dbg")
    lg.debug("debug-here")
    lg.info("info-here")

    _flush_all_handlers()
    text = log_path.read_text(encoding="utf-8")
    assert "DEBUG" in text and "debug-here" in text
    assert "INFO" in text and "info-here" in text


def test_reenable_after_off_then_debug(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """
    After calling setup() with OFF, a subsequent call with DEBUG should re-enable
    logging (thanks to logging.disable(logging.NOTSET)) and write records.
    """
    # First: OFF
    off_path = tmp_path / "off.log"
    monkeypatch.setenv("LOG_FILE", str(off_path))
    monkeypatch.setenv("LOG_LEVEL", "0")
    setup_logging()

    # Then: DEBUG
    dbg_path = tmp_path / "dbg2.log"
    monkeypatch.setenv("LOG_FILE", str(dbg_path))
    monkeypatch.setenv("LOG_LEVEL", "2")
    setup_logging()

    lg = logging.getLogger("ece461.test.reenable")
    lg.debug("hello-debug")
    _flush_all_handlers()
    assert "hello-debug" in dbg_path.read_text(encoding="utf-8")


def test_relative_LOG_FILE_resolves_against_module_root(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """
    When LOG_FILE is a relative path, setup() should resolve it against
    Path(__file__).resolve().parents[2] (module 'project root').
    We monkeypatch __file__ for the module to simulate that root.
    """
    # Pretend the module lives under <tmp>/src/ece461/logging_setup.py
    fake_module_file = tmp_path / "src" / "ece461" / "logging_setup.py"
    fake_module_file.parent.mkdir(parents=True, exist_ok=True)
    fake_module_file.write_text("# dummy", encoding="utf-8")

    # This is the relative LOG_FILE the code will join to parents[2] (== tmp_path)
    rel_path = Path("logs") / "relative.log"
    expected_abs = tmp_path / rel_path  # parents[2] is tmp_path in our fake layout

    monkeypatch.setenv("LOG_FILE", str(rel_path))  # relative!
    monkeypatch.setenv("LOG_LEVEL", "1")

    # Patch the module's __file__ so parents[2] → tmp_path
    import ece461.logging_setup as logging_setup_mod
    monkeypatch.setattr(logging_setup_mod, "__file__", str(fake_module_file), raising=True)

    setup_logging()

    configured = _first_file_handler_path()
    assert configured == expected_abs
    assert expected_abs.exists()
