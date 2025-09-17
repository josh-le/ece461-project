import sys
from url_file_parser import parse_url_file

def main() -> int:

    return 0

if __name__ == "__main__":

    # TODO: this just shows how the url file parser works, delete later
    links = parse_url_file("samples/sample-input.txt")
    for link in links:
        print(link.__dict__)

    sys.exit(main())
