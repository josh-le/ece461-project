import os, sys, json
from typing import Any, Dict
from ece461.metricCalcs.metrics import run_metrics
from ece461.url_file_parser import ModelLinks

#Reads non-empty lines from file
def read_urls(path: str) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip()]

#Checks if URL is a HuggingFace model URL
def is_hf_model(u: str) -> bool:
    return "huggingface.co/" in u and "/datasets/" not in u

#Extracts model name from model ID
def model_name(mid: str) -> str:
    return mid.split("/", 1)[-1] if "/" in mid else mid

#Rounds to 2 decimals
def two_decimals(x: Any) -> Any:
    if isinstance(x, float): return float(f"{x:.2f}")
    if isinstance(x, dict):  return {k: two_decimals(v) for k, v in x.items()}
    if isinstance(x, list):  return [two_decimals(v) for v in x]
    return x

#Emits object as NDJSON line to stdout and optionally to log file
def emit(obj: Dict[str, Any]) -> None:
    line = json.dumps(two_decimals(obj), separators=(",", ":"), ensure_ascii=False)
    sys.stdout.write(line + "\n")
    sys.stdout.flush()
    log = os.getenv("LOG_FILE")
    if log:
        with open(log, "a", encoding="utf-8") as f:
            f.write(line + "\n")

#Parses a line into ModelLinks object
def parse_line_to_links(line: str) -> ModelLinks:
    parts = [p.strip() for p in line.split(",") if p.strip()]
    model_url = next((p for p in parts if is_hf_model(p)), None)
    dataset_url = next((p for p in parts if "huggingface.co/datasets/" in p), None)
    code_url = next((p for p in parts if "github.com" in p or "gitlab.com" in p or "huggingface.co/spaces/" in p), None)
    #Use the HuggingFace model ID (owner/name) for ModelLinks.model
    mid = ""
    if model_url:
        if "://" in model_url:
            model_url = model_url.split("://", 1)[1]
        parts_url = model_url.split("/")
        try:
            i = parts_url.index("huggingface.co")
            owner = parts_url[i + 1] if i + 1 < len(parts_url) else ""
            name = parts_url[i + 2] if i + 2 < len(parts_url) else ""
            mid = f"{owner}/{name}".strip("/")
        except ValueError:
            mid = model_url
    return ModelLinks(model=mid, dataset=dataset_url, code=code_url)

#Runs all metrics and returns a dictionary of results
def evaluate_model(line: str) -> Dict[str, Any]:
    links = parse_line_to_links(line)
    name = model_name(links.model)

    results = run_metrics(links)

    size_details = {}
    size_latency = 0
    size_entry = results.get("size", {})
    size_latency = int(size_entry.get("latency_ms", 0))
    if isinstance(size_entry.get("score"), dict):
        size_details = size_entry.get("score")
    else:
        size_details = {}

    return {
        "name": name,
        "category": "MODEL",

        "net_score": float(results.get("net_score", {}).get("score", 0.0)),
        "net_score_latency": int(results.get("net_score", {}).get("latency_ms", 0)),

        "ramp_up_time": float(results.get("ramp_up", {}).get("score", 0.0)),
        "ramp_up_time_latency": int(results.get("ramp_up", {}).get("latency_ms", 0)),

        "bus_factor": float(results.get("bus-factor", {}).get("score", 0.0)),
        "bus_factor_latency": int(results.get("bus-factor", {}).get("latency_ms", 0)),

        "performance_claims": float(results.get("performance", {}).get("score", 0.0)),
        "performance_claims_latency": int(results.get("performance", {}).get("latency_ms", 0)),

        "license": float(results.get("license", {}).get("score", 0.0)),
        "license_latency": int(results.get("license", {}).get("latency_ms", 0)),

        "size_score": {
            "raspberry_pi": float(size_details.get("raspberry_pi", 0.0)),
            "jetson_nano": float(size_details.get("jetson_nano", 0.0)),
            "desktop_pc": float(size_details.get("desktop_pc", 0.0)),
            "aws_server": float(size_details.get("aws_server", 0.0)),
        },
        "size_score_latency": size_latency,

        "dataset_and_code_score": float(results.get("dataset_and_code_quality", {}).get("score", 0.0)),
        "dataset_and_code_score_latency": int(results.get("dataset_and_code_quality", {}).get("latency_ms", 0)),

        "dataset_quality": float(results.get("dataset_quality", {}).get("score", 0.0)),
        "dataset_quality_latency": int(results.get("dataset_quality", {}).get("latency_ms", 0)),

        "code_quality": float(results.get("code_quality", {}).get("score", 0.0)),
        "code_quality_latency": int(results.get("code_quality", {}).get("latency_ms", 0)),
    }

def main() -> int:
    if len(sys.argv) != 2:
        print("usage: python -m ece461.emits_ndjson <URL_FILE>", file=sys.stderr)
        return 2
    for line in read_urls(sys.argv[1]):
        if is_hf_model(line):
            emit(evaluate_model(line))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())