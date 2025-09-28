import sys
import logging
import json
from ece461.logging_setup import setup as setup_logging
from ece461.url_file_parser import parse_url_file, ModelLinks
from ece461.metricCalcs import metrics as met
from ece461.metricCalcs.net_score import calculate_net_score
from typing import List

def main() -> int:
    # 1) Turn logging on/off based on env vars (LOG_LEVEL / LOG_FILE)
    setup_logging()
    log = logging.getLogger("ece461.main")
    log.info("Logging is enabled")
    log.debug("Debug logging is enabled")

    models: List[ModelLinks] = parse_url_file(sys.argv[1])
      
    for m in models:
        results = met.run_metrics(m)
        # Create dict of metric results to pass to net score
        metrics_dict = {"name": m.model_id, "category": "MODEL", 
        "net_score": 0.0,
        "net_score_latency": 0.0,
        "ramp_up_time": 0.0,
        "ramp_up_time_latency": 0.0,
        "bus_factor": 0.0,
        "bus_factor_latency": 0.0,
        "performance_claims": 0.0,
        "performance_claims_latency": 0.0,
        "license": 0.0,
        "license_latency": 0.0,
        "size_scores": {},
        "size_score_latency": 0.0,
        "dataset_and_code_score": 0.0,
        "dataset_and_code_score_latency": 0.0,
        "dataset_quality": 0.0,
        "dataset_quality_latency": 0.0,
        "code_quality": 0.0,
        "code_quality_latency": 0.0,
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
                # Size returns a dict of hardware compatibility scores
                metrics_dict['size_scores'] = metric_result.get('score') or {}
                metrics_dict['size_score_latency'] = metric_result.get('latency_ms') or 0.0
        net_score, latency = calculate_net_score(metrics_dict)
        metrics_dict['net_score'] = round(net_score, 2)
        metrics_dict['net_score_latency'] = latency
        metrics_dict['net_score_latency'] = int(metrics_dict['net_score_latency'])

        # Round latency values
        latency_keys = [k for k in metrics_dict.keys() if 'latency' in k]
        for key in latency_keys:
            if metrics_dict[key] < 0.01:
                metrics_dict[key] = 0
            else:
                metrics_dict[key] = int(metrics_dict[key])

        print(json.dumps(metrics_dict, separators=(',', ':')))

    return 0

if __name__ == "__main__":
    sys.exit(main())