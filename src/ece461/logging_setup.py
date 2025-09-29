import logging, os, re
from pathlib import Path

def setup() -> None:
    """
    LOG_LEVEL (int): 0=off (default), 1=INFO, 2=DEBUG
    LOG_FILE (str, optional): file path (absolute or relative to project root); default is ./log/ece461.log
    """
    
    raw_log_file = os.getenv("LOG_FILE")
    if raw_log_file:
        log_file = Path(raw_log_file).expanduser()
        if not log_file.is_absolute():
            log_file = Path(__file__).resolve().parents[2] / log_file
    else:
        raise ValueError("Invalid LOG_FILE Path")
    
    # Validate: parent dir + file writability, and not a directory
    try:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        if log_file.exists() and log_file.is_dir():
            raise IsADirectoryError(f"{log_file} is a directory, not a file")
        # Try opening it (this surfaces permission and invalid-path issues now)
        with open(log_file, "a", encoding="utf-8"):
            pass
    except Exception as e:
        # If invalid, safest is to disable logging cleanly (or fallback to default_path)
        logging.disable(logging.CRITICAL)
        return
    
    lvl = os.getenv("LOG_LEVEL", "0").strip()
    if lvl == "2":
        level = logging.DEBUG
    elif lvl == "1":
        level = logging.INFO
    else:
        log_file.touch(exist_ok=True)
        logging.disable(logging.CRITICAL)
        return

    # Re-enable in case logging was disabled earlier in this process
    logging.disable(logging.NOTSET)

    logging.basicConfig(
        filename=str(log_file),
        filemode="w",
        level=level,
        format="%(asctime)s | %(levelname)s | %(funcName)s:%(lineno)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,  # replace any previous config
    )
    return
