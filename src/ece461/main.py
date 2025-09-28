import sys
import logging
from ece461.logging_setup import setup as setup_logging
from ece461.url_file_parser import parse_url_file
from ece461.metricCalcs import metrics as met

def main() -> int:
    # 1) Turn logging on/off based on env vars (LOG_LEVEL / LOG_FILE)
    setup_logging()
    log = logging.getLogger("ece461.main")
    log.info("Logging is enabled")
    log.debug("Debug logging is enabled")
    met.run_metrics("openai/whisper-tiny")

    return 0

if __name__ == "__main__":
    sys.exit(main())