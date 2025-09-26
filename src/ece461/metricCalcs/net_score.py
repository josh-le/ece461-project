import time
import logging
from typing import Dict, Any

def calculate_net_score(metrics_results: Dict[str, Dict[str, Any]]) -> tuple[float, float]:
    """
        Calculate the overall net score for a model based on metrics results.
    """
    # Start latency calculation
    start_time = time.perf_counter()
    
    # Handle size scores
    size_details = metrics_results.get('size', {}).get('details', {})
    if size_details and isinstance(size_details, dict):
        size_scores = [v for v in size_details.values() if isinstance(v, (int, float))]
        avg_size_score = sum(size_scores) / len(size_scores) if size_scores else 0.0
    else:
        avg_size_score = 0.0
    
    net_score = (
        0.25 * (metrics_results.get('license', {}).get('score', 0.0) or 0.0) +                  
        0.20 * (metrics_results.get('ramp_up', {}).get('score', 0.0) or 0.0) +                  
        0.15 * (metrics_results.get('dataset_code', {}).get('score', 0.0) or 0.0) +              
        0.15 * (metrics_results.get('bus_factor', {}).get('score', 0.0) or 0.0) +                
        0.10 * (metrics_results.get('performance', {}).get('score', 0.0) or 0.0) +               
        0.08 * avg_size_score +                                                                   
        0.04 * (metrics_results.get('code_quality', {}).get('score', 0.0) or 0.0) +            
        0.03 * (metrics_results.get('dataset_quality', {}).get('score', 0.0) or 0.0)       
    )
    
    net_score = max(0.0, min(1.0, net_score))
    
    # End latency calculation
    end_time = time.perf_counter()
    latency = round((end_time - start_time) * 1000)  # Convert to milliseconds
    logging.info(f"Net score: {net_score:.3f}, latency: {latency} ms")

    return net_score, latency
