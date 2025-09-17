from typing import Dict, Any

def calculate_net_score(metrics: Dict[str, Any]) -> float:
    """
    Calculate the overall NetScore as a weighted sum of individual metrics.
    
    Args:
        metrics: Dictionary containing all calculated metric scores
        
    Returns:
        float: NetScore between 0.0 and 1.0
    """
    
    # Define weights based on Sarah's priorities from the project spec
    # These should sum to 1.0 for a proper weighted average
    weights = {
        'ramp_up_time': 0.15,           # Ease of use is important for ACME teams
        'bus_factor': 0.10,             # Maintainer responsiveness concern
        'performance_claims': 0.15,     # Evidence-based decision making
        'license': 0.20,                # Critical for LGPLv2.1 compatibility
        'size_score': 0.15,             # Hardware deployment considerations
        'dataset_and_code_score': 0.10, # Documentation quality
        'dataset_quality': 0.10,        # Training data transparency
        'code_quality': 0.05            # Secondary concern
    }
    
    # Verify weights sum to 1.0
    assert abs(sum(weights.values()) - 1.0) < 0.001, f"Weights sum to {sum(weights.values())}, not 1.0"
    
    net_score = 0.0
    
    # Add weighted contribution from each metric
    for metric_name, weight in weights.items():
        if metric_name == 'size_score':
            # Size score is a dict - use average across hardware types
            size_scores = metrics.get('size_score', {})
            if size_scores:
                avg_size_score = sum(size_scores.values()) / len(size_scores)
                net_score += weight * avg_size_score
        else:
            # Regular 0-1 metric
            metric_value = metrics.get(metric_name, 0.0)
            net_score += weight * metric_value
    
    # Ensure result is in [0,1] range
    return max(0.0, min(1.0, net_score))


def validate_metrics(metrics: Dict[str, Any]) -> bool:
    """
    Validate that all required metrics are present and in correct format.
    
    Args:
        metrics: Dictionary of calculated metrics
        
    Returns:
        bool: True if all metrics are valid
    """
    required_metrics = [
        'ramp_up_time', 'bus_factor', 'performance_claims', 
        'license', 'size_score', 'dataset_and_code_score',
        'dataset_quality', 'code_quality'
    ]
    
    for metric in required_metrics:
        if metric not in metrics:
            print(f"Missing required metric: {metric}")
            return False
            
        if metric == 'size_score':
            # Should be a dict with hardware types
            if not isinstance(metrics[metric], dict):
                print(f"size_score should be dict, got {type(metrics[metric])}")
                return False
        else:
            # Should be a float in [0,1] range
            value = metrics[metric]
            if not isinstance(value, (int, float)) or not (0.0 <= value <= 1.0):
                print(f"Metric {metric} should be float in [0,1], got {value}")
                return False
    
    return True


# Example usage
def test_net_score():
    """Test NetScore calculation with sample metrics."""
    
    sample_metrics = {
        'ramp_up_time': 0.8,
        'bus_factor': 0.6,
        'performance_claims': 0.9,
        'license': 1.0,  # Perfect LGPLv2.1 compatibility
        'size_score': {
            'raspberry_pi': 0.2,
            'jetson_nano': 0.7,
            'desktop_pc': 0.9,
            'aws_server': 1.0
        },
        'dataset_and_code_score': 0.7,
        'dataset_quality': 0.8,
        'code_quality': 0.6
    }
    
    if validate_metrics(sample_metrics):
        net_score = calculate_net_score(sample_metrics)
        print(f"Sample NetScore: {net_score:.3f}")
        
        # Show breakdown
        print("\nScore breakdown:")
        for metric, value in sample_metrics.items():
            if metric == 'size_score':
                avg_size = sum(value.values()) / len(value.values())
                print(f"  {metric}: {avg_size:.3f} (avg across hardware)")
            else:
                print(f"  {metric}: {value:.3f}")
    else:
        print("Invalid metrics provided")


if __name__ == "__main__":
    test_net_score()
