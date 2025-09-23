import os, sys, json, time
from typing import Any, Dict
from ece461.metricCalcs.metrics import Metrics

#Read url file (commas/newlines), return non-empty URLs
def read_urls(path: str) -> list[str]:
    text = open(path, "r", encoding="utf-8").read()
    return [u.strip() for u in text.replace("\n", ",").split(",") if u.strip()]

#Only keep hf model urls (skip datasets and others)
def is_hf_model(u: str) -> bool:
    return "huggingface.co/" in u and "/datasets/" not in u

#Converts url to model id "owner/name"
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

#Converts url to just model name
def model_name(u: str) -> str:
    mid = model_id(u)
    return mid.split("/", 1)[1] if "/" in mid else mid

#Round all floats to 2 decimals in dicts
def two_decimals(x: Any) -> Any:
    if isinstance(x, float): return float(f"{x:.2f}")
    if isinstance(x, dict):  return {k: two_decimals(v) for k, v in x.items()}
    if isinstance(x, list):  return [two_decimals(v) for v in x]
    return x

#Emit one ndjson line to stdout and also to $LOG_FILE (if set)
def emit(obj: Dict[str, Any]) -> None:
    line = json.dumps(two_decimals(obj), separators=(",", ":"), ensure_ascii=False)
    sys.stdout.write(line + "\n")
    sys.stdout.flush()
    log = os.getenv("LOG_FILE")
    if log:
        with open(log, "a", encoding="utf-8") as f:
            f.write(line + "\n")

#Calls the implemented metrics and left others as 0
def evaluate_model(url: str) -> Dict[str, Any]:
    mid = model_id(url)
    name = model_name(url)

    #Implemented metrics in metrics.py
    ramp_score, ramp_lat = Metrics.calculate_ramp_up_metric(mid)
    lic_score,  lic_lat  = Metrics.calculate_license_metric(mid)
    perf_score, perf_lat = Metrics.calculate_performance_metric(mid)

    t0 = time.perf_counter()
    size_map = Metrics.calculate_size_metric(mid)
    size_lat = int((time.perf_counter() - t0) * 1000)

    #Analyzes current folder or overrides with CODE_PATH
    code_path = os.getenv("CODE_PATH") or os.getcwd()
    codeq_score, codeq_lat = Metrics.score_code_quality(code_path)

    #Everything not implemented yet stays 0
    net_score, net_lat = 0.0, 0
    bus_score, bus_lat = 0.0, 0
    avail_score, avail_lat = 0.0, 0
    dsq_score, dsq_lat = 0.0, 0

    return {
        "name": name,
        "category": "MODEL",

        "net_score": net_score,
        "net_score_latency": net_lat,

        "ramp_up_time": ramp_score,
        "ramp_up_time_latency": int(ramp_lat),

        "bus_factor": bus_score,
        "bus_factor_latency": bus_lat,

        "performance_claims": perf_score,
        "performance_claims_latency": int(perf_lat),

        "license": lic_score,
        "license_latency": int(lic_lat),

        "size_score": {
            "raspberry_pi": float(size_map.get("raspberry_pi", 0.0)),
            "jetson_nano": float(size_map.get("jetson_nano", 0.0)),
            "desktop_pc": float(size_map.get("desktop_pc", 0.0)),
            "aws_server": float(size_map.get("aws_server", 0.0)),
        },
        "size_score_latency": size_lat,

        "dataset_and_code_score": avail_score,
        "dataset_and_code_score_latency": avail_lat,

        "dataset_quality": dsq_score,
        "dataset_quality_latency": dsq_lat,

        "code_quality": codeq_score,
        "code_quality_latency": int(codeq_lat),
    }

#Read urls, keep hf models, evaluate, emit ndjson
def main() -> int:
    if len(sys.argv) != 2:
        print("usage: python -m ece461.emit_ndjson <URL_FILE>", file=sys.stderr)
        return 2
    for u in read_urls(sys.argv[1]):
        if is_hf_model(u):
            emit(evaluate_model(u))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())