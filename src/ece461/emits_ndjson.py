import os, sys, json
from typing import Any, Dict
from ece461.metricCalcs.metrics import run_metrics
from ece461.url_file_parser import ModelLinks
from ece461.metricCalcs.net_score import calculate_net_score

# Reads non-empty lines from file
def read_urls(path: str) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip()]

# Checks if URL is a HuggingFace model URL
def is_hf_model(u: str) -> bool:
    return "huggingface.co/" in u and "/datasets/" not in u

# Extracts model name from model ID
def model_name(mid: str) -> str:
    return mid.split("/", 1)[-1] if "/" in mid else mid

# Rounds floats to 2 decimals recursively
def two_decimals(x: Any) -> Any:
    if isinstance(x, float): return float(f"{x:.2f}")
    if isinstance(x, dict):  return {k: two_decimals(v) for k, v in x.items()}
    if isinstance(x, list):  return [two_decimals(v) for v in x]
    return x

# Emits object as NDJSON line to stdout and optionally to log file
def emit(obj: Dict[str, Any]) -> None:
    line = json.dumps(two_decimals(obj), separators=(",", ":"), ensure_ascii=False)
    sys.stdout.write(line + "\n")
    sys.stdout.flush()
    log = os.getenv("LOG_FILE")
    if log:
        with open(log, "a", encoding="utf-8") as f:
            f.write(line + "\n")

# Parses a line into ModelLinks object
def parse_line_to_links(line: str) -> ModelLinks:
    parts = [p.strip() for p in line.split(",") if p.strip()]
    model_url = next((p for p in parts if is_hf_model(p)), None)
    dataset_url = next((p for p in parts if "huggingface.co/datasets/" in p), None)
    code_url = next((p for p in parts if "github.com" in p or "gitlab.com" in p or "huggingface.co/spaces/" in p), None)
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

# Runs all metrics, builds metrics_dict, calculates net score, and emits NDJSON
def evaluate_model(line: str) -> Dict[str, Any]:
    links = parse_line_to_links(line)
    name = model_name(links.model)

    results = run_metrics(links)
    metrics_dict = {
        "name": name,
        "category": "MODEL"
    }

    for metric_name, metric_result in results.items():
        if metric_name == 'license':
            metrics_dict['license'] = metric_result.get('score') or 0.0
            metrics_dict['license_latency'] = metric_result.get('latency_ms') or 0.0
        elif metric_name == 'ramp_up':
            metrics_dict['ramp_up_time'] = metric_result.get('score') or 0.0
            metrics_dict['ramp_up_time_latency'] = metric_result.get('latency_ms') or 0.0
        elif metric_name == 'dataset_and_code_quality':
            metrics_dict['dataset_and_code_score'] = metric_result.get('score') or 0.0
            metrics_dict['dataset_and_code_score_latency'] = metric_result.get('latency_ms') or 0.0
        elif metric_name == 'bus-factor':
            metrics_dict['bus_factor'] = metric_result.get('score') or 0.0
            metrics_dict['bus_factor_latency'] = metric_result.get('latency_ms') or 0.0
        elif metric_name == 'performance':
            metrics_dict['performance_claims'] = metric_result.get('score') or 0.0
            metrics_dict['performance_claims_latency'] = metric_result.get('latency_ms') or 0.0
        elif metric_name == 'code_quality':
            metrics_dict['code_quality'] = metric_result.get('score') or 0.0
            metrics_dict['code_quality_latency'] = metric_result.get('latency_ms') or 0.0
        elif metric_name == 'dataset_quality':
            metrics_dict['dataset_quality'] = metric_result.get('score') or 0.0
            metrics_dict['dataset_quality_latency'] = metric_result.get('latency_ms') or 0.0
        elif metric_name == 'size':
            metrics_dict['size_score'] = metric_result.get('score') or {}
            metrics_dict['size_score_latency'] = metric_result.get('latency_ms') or 0.0

    net_score, net_score_latency = calculate_net_score(metrics_dict)
    metrics_dict['net_score'] = net_score
    metrics_dict['net_score_latency'] = net_score_latency

    return metrics_dict

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