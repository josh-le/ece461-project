import sys
import logging
from logging_setup import setup as setup_logging
from url_file_parser import parse_url_file
from API.hf_api import process_hf_links
from API.github_api import process_github_links

def main() -> int:
    # 1) Turn logging on/off based on env vars (LOG_LEVEL / LOG_FILE)
    setup_logging()
    log = logging.getLogger("ece461.main")
    log.info("Logging is enabled")
    log.debug("Debug logging is enabled")

    return 0

if __name__ == "__main__":

    # TODO: this just shows how the url file parser works, delete later
    links = parse_url_file("samples/sample-input.txt")
    for link in links:
        print(link.__dict__)
    
    # Process links with APIs
    process_hf_links(links)
    process_github_links(links)

    sys.exit(main())
