import sys, json
from ece461.url_file_parser import parse_url_file
from ece461.metricCalcs import metrics as met
from ece461.metricCalcs.net_score import calculate_net_score

def emit(obj):
    line = json.dumps(obj, separators=(",", ":"), ensure_ascii=False)
    sys.stdout.write(line + "\n")
    sys.stdout.flush()

def get_model_name(model_obj):
    # Use model_id if available, else model
    model_id = getattr(model_obj, "model_id", None) or getattr(model_obj, "model", "")
    # If model_id is "org/model", take only the model part
    if "/" in model_id:
        return model_id.split("/")[-1]
    return model_id

def main():
    if len(sys.argv) != 2:
        print("usage: python -m ece461.emits_ndjson <URL_FILE>", file=sys.stderr)
        return 2

    models = parse_url_file(sys.argv[1])
    for m in models:
        results = met.run_metrics(m)
        metrics_dict = {}

        # Build metrics_dict exactly as in main.py
        metrics_dict['license'] = results.get('license', {}).get('score') or 0.0
        metrics_dict['license_latency'] = int(results.get('license', {}).get('latency_ms') or 0)
        metrics_dict['ramp_up_time'] = results.get('ramp_up', {}).get('score') or 0.0
        metrics_dict['ramp_up_time_latency'] = int(results.get('ramp_up', {}).get('latency_ms') or 0)
        metrics_dict['dataset_and_code_score'] = results.get('dataset_and_code_quality', {}).get('score') or 0.0
        metrics_dict['dataset_and_code_score_latency'] = int(results.get('dataset_and_code_quality', {}).get('latency_ms') or 0)
        metrics_dict['bus_factor'] = results.get('bus-factor', {}).get('score') or 0.0
        metrics_dict['bus_factor_latency'] = int(results.get('bus-factor', {}).get('latency_ms') or 0)
        metrics_dict['performance_claims'] = results.get('performance', {}).get('score') or 0.0
        metrics_dict['performance_claims_latency'] = int(results.get('performance', {}).get('latency_ms') or 0)
        metrics_dict['code_quality'] = results.get('code_quality', {}).get('score') or 0.0
        metrics_dict['code_quality_latency'] = int(results.get('code_quality', {}).get('latency_ms') or 0)
        metrics_dict['dataset_quality'] = results.get('dataset_quality', {}).get('score') or 0.0
        metrics_dict['dataset_quality_latency'] = int(results.get('dataset_quality', {}).get('latency_ms') or 0)
        size_scores = results.get('size', {}).get('score') or {}
        metrics_dict['size_score'] = {
            "raspberry_pi": float(size_scores.get('raspberry_pi', 0.0)),
            "jetson_nano": float(size_scores.get('jetson_nano', 0.0)),
            "desktop_pc": float(size_scores.get('desktop_pc', 0.0)),
            "aws_server": float(size_scores.get('aws_server', 0.0)),
        }
        metrics_dict['size_score_latency'] = int(results.get('size', {}).get('latency_ms') or 0)

        # Calculate net score
        net_score, net_score_latency = calculate_net_score({
            "license": metrics_dict["license"],
            "ramp_up_time": metrics_dict["ramp_up_time"],
            "dataset_and_code_score": metrics_dict["dataset_and_code_score"],
            "bus_factor": metrics_dict["bus_factor"],
            "performance_claims": metrics_dict["performance_claims"],
            "size_scores": metrics_dict["size_score"],
            "code_quality": metrics_dict["code_quality"],
            "dataset_quality": metrics_dict["dataset_quality"]
        })

        output = {
            "name": get_model_name(m),
            "category": "MODEL",
            "net_score": float(net_score),
            "net_score_latency": int(net_score_latency),
            "ramp_up_time": metrics_dict["ramp_up_time"],
            "ramp_up_time_latency": metrics_dict["ramp_up_time_latency"],
            "bus_factor": metrics_dict["bus_factor"],
            "bus_factor_latency": metrics_dict["bus_factor_latency"],
            "performance_claims": metrics_dict["performance_claims"],
            "performance_claims_latency": metrics_dict["performance_claims_latency"],
            "license": metrics_dict["license"],
            "license_latency": metrics_dict["license_latency"],
            "size_score": metrics_dict["size_score"],
            "size_score_latency": metrics_dict["size_score_latency"],
            "dataset_and_code_score": metrics_dict["dataset_and_code_score"],
            "dataset_and_code_score_latency": metrics_dict["dataset_and_code_score_latency"],
            "dataset_quality": metrics_dict["dataset_quality"],
            "dataset_quality_latency": metrics_dict["dataset_quality_latency"],
            "code_quality": metrics_dict["code_quality"],
            "code_quality_latency": metrics_dict["code_quality_latency"],
        }
        emit(output)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())