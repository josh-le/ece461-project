import subprocess, re, sys

def score_code_quality(path: str) -> dict[str, object]:
    """Run pylint on Python files in 'path' and return normalized score."""
    proc = subprocess.run(
        [sys.executable, "-m", "pylint", "--exit-zero", "--score=y", path],
        capture_output=True, text=True
    )
    m = re.search(r"rated at\s+([0-9.]+)/10", proc.stdout)
    score10 = float(m.group(1)) if m else 0.0
    return {
        "name": "code_quality",
        "score": max(0.0, min(1.0, score10/10)),
        "raw": {"pylint_score": score10}
    }
