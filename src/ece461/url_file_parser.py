from typing import List
import validators, logging, os, sys

class ModelLinks:
    def __init__(self, model: str, dataset: str | None = None, code: str | None = None) -> None:
        self.model = model
        self.dataset = dataset
        self.code = code

def parse_url_file(path: str) -> List[ModelLinks]:
    links = []

    if not os.path.exists(path):
        logging.exception("Error: URL file path doesn't exist")
        sys.exit(1)

    with open(path, 'r') as file:
        lines = file.readlines()

    for line in lines:
        sp = line.strip().split(",")

        for s in sp:
            try:
                validators.url(s)
                links.append(ModelLinks(sp[2], sp[1] if sp[1] else None, sp[0] if sp[0] else None))
            except Exception as e:
                logging.exception("Error: " + str(e))


    return links

