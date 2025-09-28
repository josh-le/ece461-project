import re
import subprocess
import sys
from typing import List, Tuple


CMD: List[str] = [
    "python3", "-m", "pytest", "-q", "--disable-warnings",
    "--cov=ece461", "--cov-report=term",
]


def run(cmd: List[str]) -> Tuple[int, str]:
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    return p.returncode, p.stdout


def last_int(text: str, pattern: str) -> int:
    m = re.findall(pattern, text, flags=re.IGNORECASE)
    return int(m[-1]) if m else 0


def extract_counts(text: str) -> Tuple[int, int, int, int]:
    passed = last_int(text, r"(\d+)\s+passed")
    failed = last_int(text, r"(\d+)\s+failed")
    errors = last_int(text, r"(\d+)\s+errors?")  # matches 'error' or 'errors'
    skipped = last_int(text, r"(\d+)\s+skipped")
    return passed, failed, errors, skipped


def extract_coverage(text: str) -> str:
    # Prefer the TOTAL row percentage from coverage table
    total_lines = [ln for ln in text.splitlines() if ln.strip().startswith("TOTAL")]
    for ln in total_lines:
        m = re.search(r"(\d+)%", ln)
        if m:
            return m.group(1) + "%"
    # Fallback: last percentage-looking number in output
    m2 = re.findall(r"(\d+)%", text)
    return (m2[-1] + "%") if m2 else "N/A"


def main() -> int:
    _, out = run(CMD)
    passed, failed, errors, skipped = extract_counts(out)
    cov = extract_coverage(out)
    print(f"{passed}/{passed + failed + errors + skipped} test cases passed. {cov} line coverage achieved.")
    return 0


if __name__ == "__main__":
    sys.exit(main())