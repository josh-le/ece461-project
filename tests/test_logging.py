from pathlib import Path
import logging
import pytest

from ece461.logging_setup import setup as setup_logging


def _flush_all_handlers() -> None:
    """Ensure log records are written to disk before assertions."""
    root = logging.getLogger()
    for h in list(root.handlers):
        try:
            h.flush()  # type: ignore[attr-defined]
        except Exception:
            pass


def _first_file_handler_path() -> Path:
    """Return the path of the first FileHandler configured on the root logger."""
    root = logging.getLogger()
    for h in root.handlers:
        # FileHandler exposes .baseFilename
        base = getattr(h, "baseFilename", None)
        if isinstance(base, str):
            return Path(base)
    raise AssertionError("Expected a FileHandler to be configured by setup_logging()")


def test_logging_setup_with_custom_file_debug(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """
    If LOG_FILE is set, logging should write to that exact path.
    LOG_LEVEL=2 (DEBUG) should emit DEBUG and INFO.
    """
    log_path = tmp_path / "myapp.log"
    monkeypatch.setenv("LOG_LEVEL", "2")           # DEBUG
    monkeypatch.setenv("LOG_FILE", str(log_path))  # explicit file

    setup_logging()

    logger = logging.getLogger("ece461.test.custom")
    logger.debug("debug-msg")
    logger.info("info-msg")

    _flush_all_handlers()
    # Get the actual file the logger uses (should match our custom path)
    configured = _first_file_handler_path()
    assert configured == log_path
    text = configured.read_text(encoding="utf-8")
    assert "DEBUG" in text and "debug-msg" in text
    assert "INFO" in text and "info-msg" in text


def test_logging_setup_default_file_info(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """
    If LOG_FILE is not set, setup() uses its own default path (in your code,
    that’s project-root relative). We discover it by reading the FileHandler.
    LOG_LEVEL=1 (INFO) should emit INFO lines.
    """
    monkeypatch.delenv("LOG_FILE", raising=False)
    monkeypatch.setenv("LOG_LEVEL", "1")  # INFO
    # CWD is irrelevant if your implementation computes project-root; we keep this
    # chdir to ensure the test doesn't accidentally write into the repo root.
    monkeypatch.chdir(tmp_path)

    setup_logging()

    logging.getLogger("ece461.test.default").info("hello-default")
    _flush_all_handlers()

    configured = _first_file_handler_path()
    assert configured.exists(), f"Expected log file to be created at: {configured}"
    text = configured.read_text(encoding="utf-8")
    assert "INFO" in text and "hello-default" in text