# ece461/logging_setup.py
import logging, os, sys
from pathlib import Path

def setup() -> None:
    raw_log_file = os.getenv("LOG_FILE")
    if raw_log_file:
        log_file = Path(raw_log_file).expanduser()
        if not log_file.is_absolute():
            log_file = Path(__file__).resolve().parents[2] / log_file
    else:
        logging.error("Invalid LOG_FILE Path")
        sys.exit(1)

    # --- NEW: validate target is a writable file path (not a directory) ---
    log_file.parent.mkdir(parents=True, exist_ok=True)
    if log_file.exists() and log_file.is_dir():
        raise ValueError(f"LOG_FILE points to a directory: {log_file}")
    try:
        # this surfaces permission / invalid-path issues early
        with open(log_file, "a", encoding="utf-8"):
            pass
    except Exception as e:
        logging.error(f"Invalid LOG_FILE: {e}")
        sys.exit(1)
    # ----------------------------------------------------------------------

    lvl = os.getenv("LOG_LEVEL", "0").strip()
    if lvl == "2":
        level = logging.DEBUG
    elif lvl == "1":
        level = logging.INFO
    else:
        log_file.touch(exist_ok=True)
        logging.disable(logging.CRITICAL)
        return

    logging.disable(logging.NOTSET)
    logging.basicConfig(
        filename=str(log_file),
        filemode="w",
        level=level,
        format="%(asctime)s | %(levelname)s | %(funcName)s:%(lineno)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
        encoding="utf-8",
    )
