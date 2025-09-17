import validators

def validate_url(url: str) -> bool:
    """
    Validates if the given string is a properly formatted URL.
    
    Args:
        url (str): The URL string to validate.
    """
    return validators.url(url)

def parse_URL_file(file_path: str) -> list[str]:
    """
    Reads a file containing URLs (one per line) and returns a list of valid URLs.
    
    Args:
        file_path (str): The path to the file containing URLs.
    """