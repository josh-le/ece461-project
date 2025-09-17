from typing import List
import validators

class ModelLinks:
    def __init__(self, model: str, dataset: str | None = None, code: str | None = None) -> None:
        self.model = model
        self.dataset = dataset
        self.code = code

def validate_url(url: str) -> bool:
    """
    Validates if the given string is a properly formatted URL.
    
    Args:
        url (str): The URL string to validate.
    """
    return validators.url(url)

# TODO: i am assuming the path exists here
def parse_url_file(path: str) -> List[ModelLinks]:
    links = []

    with open(path, 'r') as file:
        lines = file.readlines()

    for line in lines:
        sp = line.strip().split(",")

        for _ in sp:
            # TODO: check link validity
            continue

        links.append(ModelLinks(sp[2], sp[1] if sp[1] else None, sp[0] if sp[0] else None))

    return links

