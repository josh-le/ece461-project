import os, sys, json
from typing import Any, Dict
from ece461.metricCalcs.metrics import run_metrics

def read_urls(path: str) -> list[str]:
    text = open(path, "r", encoding="utf-8").read()
    return [u.strip() for u in text.replace("\n", ",").split(",") if u.strip()]

def is_hf_model(u: str) -> bool:
    return "huggingface.co/" in u and "/datasets/" not in u

def model_id(u: str) -> str:
    if "://" in u:
        u = u.split("://", 1)[1]
    parts = u.split("/")
    try:
        i = parts.index("huggingface.co")
    except ValueError:
        return u
    owner = parts[i + 1] if i + 1 < len(parts) else ""
    name  = parts[i + 2] if i + 2 < len(parts) else ""
    return f"{owner}/{name}".strip("/")

def model_name(u: str) -> str:
    mid = model_id(u)
    return mid.split("/", 1)[1] if "/" in mid else mid

def two_decimals(x: Any) -> Any:
    if isinstance(x, float): return float(f"{x:.2f}")
    if isinstance(x, dict):  return {k: two_decimals(v) for k, v in x.items()}
    if isinstance(x, list):  return [two_decimals(v) for v in x]
    return x

def emit(obj: Dict[str, Any]) -> None:
    line = json.dumps(two_decimals(obj), separators=(",", ":"), ensure_ascii=False)
    sys.stdout.write(line + "\n")
    sys.stdout.flush()
    log = os.getenv("LOG_FILE")
    if log:
        with open(log, "a", encoding="utf-8") as f:
            f.write(line + "\n")

def evaluate_model(url: str) -> Dict[str, Any]:
    mid = model_id(url)
    name = model_name(url)

    #Run all metrics in parallel and get results as a dict
    results = run_metrics(mid)

    def score_of(key: str) -> float:
        v = results.get(key, {})
        return float(v.get("score") or 0.0) if isinstance(v, dict) else 0.0

    def latency_of(key: str) -> int:
        v = results.get(key, {})
        try:
            return int(v.get("latency_ms") or 0)
        except Exception:
            return 0

    #Size metric details (expected in results["size"]["score"] as a dict)
    size_details = {}
    size_latency = 0
    size_entry = results.get("size")
    if isinstance(size_entry, dict):
        size_latency = int(size_entry.get("latency_ms") or 0)
        #If score is a dict, use it for details
        if isinstance(size_entry.get("score"), dict):
            size_details = size_entry.get("score")
        else:
            size_details = {}

    #Not implemented metrics (set to 0 if missing)
    net_score, net_lat = score_of("net_score"), latency_of("net_score")
    bus_score, bus_lat = score_of("bus_factor"), latency_of("bus_factor")
    avail_score, avail_lat = score_of("dataset_and_code_score"), latency_of("dataset_and_code_score")
    dsq_score, dsq_lat = score_of("dataset_quality"), latency_of("dataset_quality")

    return {
        "name": name,
        "category": "MODEL",

        "net_score": net_score,
        "net_score_latency": net_lat,

        "ramp_up_time": score_of("ramp_up"),
        "ramp_up_time_latency": latency_of("ramp_up"),

        "bus_factor": bus_score,
        "bus_factor_latency": bus_lat,

        "performance_claims": score_of("performance"),
        "performance_claims_latency": latency_of("performance"),

        "license": score_of("license"),
        "license_latency": latency_of("license"),

        "size_score": {
            "raspberry_pi": float(size_details.get("raspberry_pi", 0.0)),
            "jetson_nano": float(size_details.get("jetson_nano", 0.0)),
            "desktop_pc": float(size_details.get("desktop_pc", 0.0)),
            "aws_server": float(size_details.get("aws_server", 0.0)),
        },
        "size_score_latency": int(size_latency),

        "dataset_and_code_score": avail_score,
        "dataset_and_code_score_latency": avail_lat,

        "dataset_quality": dsq_score,
        "dataset_quality_latency": dsq_lat,

        "code_quality": score_of("code_quality"),
        "code_quality_latency": latency_of("code_quality"),
    }

def main() -> int:
    if len(sys.argv) != 2:
        print("usage: python -m ece461.emits_ndjson <URL_FILE>", file=sys.stderr)
        return 2
    for u in read_urls(sys.argv[1]):
        if is_hf_model(u):
            emit(evaluate_model(u))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())