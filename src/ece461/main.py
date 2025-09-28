import sys
import logging
from ece461.logging_setup import setup as setup_logging
from ece461.url_file_parser import parse_url_file, ModelLinks
from ece461.metricCalcs import metrics as met
from typing import List

def main() -> int:
    # 1) Turn logging on/off based on env vars (LOG_LEVEL / LOG_FILE)
    setup_logging()
    log = logging.getLogger("ece461.main")
    log.info("Logging is enabled")
    log.debug("Debug logging is enabled")

    models: List[ModelLinks] = parse_url_file(sys.argv[1])
      
    for m in models:
        met.run_metrics(m)

    return 0

if __name__ == "__main__":
    sys.exit(main())
