import time
import logging

def calculate_net_score(metrics: dict) -> tuple[float, float]:
    """
        Calculate the overall net score for a model based on metrics.
    """

    # Start latency calculation
    start_time = time.perf_counter()
    
    # Take average of size scores
    size_scores = metrics.get('size_score', {})
    if size_scores and isinstance(size_scores, dict):
        avg_size_score = sum(size_scores.values()) / len(size_scores)
    else:
        avg_size_score = 0.0
    
    # Weights for each metric
    weights = {
        'license': 0.25,      
        'ramp_up_time': 0.20,        
        'dataset_and_code_score': 0.15, 
        'bus_factor': 0.15,       
        'performance_claims': 0.10,   
        'size_compatibility': 0.05,  
        'code_quality': 0.05,       
        'dataset_quality': 0.05       
    }
    
    # Calculate weighted sum
    net_score = (
        weights['license'] * metrics.get('license', 0.0) +
        weights['ramp_up_time'] * metrics.get('ramp_up_time', 0.0) +
        weights['dataset_and_code_score'] * metrics.get('dataset_and_code_score', 0.0) +
        weights['bus_factor'] * metrics.get('bus_factor', 0.0) +
        weights['performance_claims'] * metrics.get('performance_claims', 0.0) +
        weights['size_compatibility'] * avg_size_score +
        weights['code_quality'] * metrics.get('code_quality', 0.0) +
        weights['dataset_quality'] * metrics.get('dataset_quality', 0.0)
    )
    
    # End latency calculation
    end_time = time.perf_counter()
    latency = round((end_time - start_time) * 1000)  # Convert to milliseconds
    logging.info("Net score latency: %d ms", latency)

    return max(0.0, min(1.0, net_score)), latency
