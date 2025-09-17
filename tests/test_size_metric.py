# Test size metric calculation
from src.ece461.metricCalcs.size_metric import calculate_size_metric

def test_size_metric_small_model():
    # Test with a small model
    scores = calculate_size_metric("distilbert-base-uncased")
    assert all(0.0 <= score <= 1.0 for score in scores.values())
    expected_hardware = ['raspberry_pi', 'jetson_nano', 'desktop_pc', 'aws_server']
    assert all(hw in scores for hw in expected_hardware)
    assert len(scores) == 4

def test_size_metric_large_model():
    # Test with a large model
    scores = calculate_size_metric("bert-base-uncased")
    assert all(0.0 <= score <= 1.0 for score in scores.values())
    expected_hardware = ['raspberry_pi', 'jetson_nano', 'desktop_pc', 'aws_server']
    assert all(hw in scores for hw in expected_hardware)
    assert len(scores) == 4

