from typing import List
import validators, logging, os, sys

class ModelLinks:
    def __init__(self, model: str, dataset: str | None = None, code: str | None = None, model_id: str = "") -> None:
        self.model = model
        self.dataset = dataset
        self.code = code
        self.model_id = model_id

def parse_url_file(path: str) -> List[ModelLinks]:
    links: List[ModelLinks] = []

    if not os.path.exists(path):
        logging.exception("Error: URL file path doesn't exist")
        sys.exit(1)

    with open(path, 'r') as file:
        lines: List[str] = file.readlines()

    for line in lines:
        code, dataset, model = line.strip().split(",")

        if not model:
            logging.exception("Error: model link not found")
            sys.exit(1)
        else:
            validators.url(model)

        if code:
            validators.url(code)

        if dataset:
            validators.url(dataset)

        spl: List[str] = model.split(".co/")[1].split("/")
        model_id: str = spl[0] + "/" + spl[1]
        links.append(ModelLinks(model, dataset if dataset else None, code if code else None, model_id))

    return links

