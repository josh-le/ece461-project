import sys, json
from typing import Any, List
from ece461.url_file_parser import parse_url_file
from ece461.metricCalcs import metrics as met

def emit(obj):
    line = json.dumps(obj, separators=(",", ":"), ensure_ascii=False)
    sys.stdout.write(line + "\n")
    sys.stdout.flush()

def get_model_name(model_obj):
    model_id = getattr(model_obj, "model_id", None) or getattr(model_obj, "model", "")
    if "/" in model_id:
        return model_id.split("/")[-1]
    return model_id

def two_decimals(x: Any) -> Any:
    if isinstance(x, float): return float(f"{x:.2f}")
    if isinstance(x, dict):  return {k: two_decimals(v) for k, v in x.items()}
    if isinstance(x, list):  return [two_decimals(v) for v in x]
    return x

def main():
    if len(sys.argv) != 2:
        print("usage: python -m ece461.emits_ndjson <URL_FILE>", file=sys.stderr)
        return 2

    models: List = parse_url_file(sys.argv[1])
    for m in models:
        results = met.run_metrics(m)
        output = {
            "name": get_model_name(m),
            "category": "MODEL",
            "net_score": results.get("net_score", 0.0),
            "net_score_latency": results.get("net_score_latency", 0),
            "ramp_up_time": results.get("ramp_up", {}).get("score", 0.0),
            "ramp_up_time_latency": int(results.get("ramp_up", {}).get("latency_ms", 0) or 0),
            "bus_factor": results.get("bus-factor", {}).get("score", 0.0),
            "bus_factor_latency": int(results.get("bus-factor", {}).get("latency_ms", 0) or 0),
            "performance_claims": results.get("performance", {}).get("score", 0.0),
            "performance_claims_latency": int(results.get("performance", {}).get("latency_ms", 0) or 0),
            "license": results.get("license", {}).get("score", 0.0),
            "license_latency": int(results.get("license", {}).get("latency_ms", 0) or 0),
            "size_score": results.get("size", {}).get("score") or {
                "raspberry_pi": 0.0,
                "jetson_nano": 0.0,
                "desktop_pc": 0.0,
                "aws_server": 0.0,
            },
            "size_score_latency": int(results.get("size", {}).get("latency_ms", 0) or 0),
            "dataset_and_code_score": results.get("dataset_and_code_quality", {}).get("score", 0.0),
            "dataset_and_code_score_latency": int(results.get("dataset_and_code_quality", {}).get("latency_ms", 0) or 0),
            "dataset_quality": results.get("dataset_quality", {}).get("score", 0.0),
            "dataset_quality_latency": int(results.get("dataset_quality", {}).get("latency_ms", 0) or 0),
            "code_quality": results.get("code_quality", {}).get("score", 0.0),
            "code_quality_latency": int(results.get("code_quality", {}).get("latency_ms", 0) or 0),
        }
        emit(two_decimals(output))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())