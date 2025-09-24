from __future__ import annotations
import os, sys, subprocess, logging
from ece461.logging_setup import setup as setup_logging
from ece461.emit_ndjson import main as emit_ndjson_main
from url_file_parser import parse_url_file

def _install() -> int:
    #Installs dependencies from requirements.txt
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    req = os.path.join(repo_root, "requirements.txt")
    if not os.path.isfile(req):
        print(f"error: requirements.txt not found at {req}", file=sys.stderr)
        return 2
    return subprocess.call([sys.executable, "-m", "pip", "install", "-r", req])

def _test() -> int:
    #Runs tests with pytest and coverage
    return subprocess.call([
        sys.executable, "-m", "pytest",
        "-q", "--disable-warnings",
        "--cov=ece461", "--cov-report=term"
    ])

def _score(url_file: str) -> int:
    #Makes the emitter think it was called directly with that file
    sys.argv = ["emit_ndjson", url_file]
    return emit_ndjson_main()

def main() -> int:
    # 1) Turn logging on/off based on env vars (LOG_LEVEL / LOG_FILE)
    setup_logging()
    log = logging.getLogger("ece461.main")
    log.info("Logging is enabled")
    log.debug("Debug logging is enabled")

    #CLI modes: install, test, <URL_FILE>
    if len(sys.argv) < 2:
        print("usage: python -m ece461.main install|test|<URL_FILE>", file=sys.stderr)
        return 2

    cmd = sys.argv[1].strip().lower()
    if cmd == "install":
        log.info("Running install…")
        return _install()

    if cmd == "test":
        log.info("Running tests…")
        return _test()

    url_file = sys.argv[1]
    if not os.path.isfile(url_file):
        print(f"error: URL file not found: {url_file}", file=sys.stderr)
        return 2

    log.info("Scoring from URL file: %s", url_file)
    return _score(url_file)

if __name__ == "__main__":
    sys.exit(main())
