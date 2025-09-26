from src.ece461.metricCalcs.metrics import (
    calculate_ramp_up_metric,
    calculate_license_metric,
    calculate_performance_metric,
    calculate_dataset_code_metric,
    calculate_size_metric
)

####################### Test performance metric calculation ########################
def test_performance():
    score1, latency_ms1 = Metrics.calculate_ramp_up_metric("openai/whisper-tiny")
    assert 0.0 <= score1 <= 1
    assert latency_ms1 >= 0.0

####################### Test license metric calculation ########################
def test_license():
    score1, latency_ms1 = Metrics.calculate_ramp_up_metric("openai/whisper-tiny")
    assert 0.0 <= score1 <= 1
    assert latency_ms1 >= 0.0

####################### Test ramp-up metric calculation ########################
def test_ramp_up():
    score1, latency_ms1 = Metrics.calculate_ramp_up_metric("openai/whisper-tiny")
    assert 0.0 <= score1 <= 1
    assert latency_ms1 >= 0.0

####################### Test size metric calculation ########################
def test_size_metric_small_model():
    # Test with a small model
    scores = Metrics.calculate_size_metric("distilbert-base-uncased")
    assert all(0.0 <= score <= 1.0 for score in scores.values())
    expected_hardware = ['raspberry_pi', 'jetson_nano', 'desktop_pc', 'aws_server']
    assert all(hw in scores for hw in expected_hardware)
    assert len(scores) == 4

def test_size_metric_large_model():
    # Test with a large model
    scores = Metrics.calculate_size_metric("bert-base-uncased")
    assert all(0.0 <= score <= 1.0 for score in scores.values())
    expected_hardware = ['raspberry_pi', 'jetson_nano', 'desktop_pc', 'aws_server']
    assert all(hw in scores for hw in expected_hardware)
    assert len(scores) == 4

####################### Test dataset_code metric calculation ########################
def test_dataset_code_metric():
    """Test the dataset_code metric with a model that has both code and dataset links"""
    from src.ece461.metricCalcs.metrics import calculate_dataset_code_metric
    
    score, latency_ms = calculate_dataset_code_metric("microsoft/DialoGPT-medium")
    
    # Basic validation
    assert 0.0 <= score <= 1.0, f"Score {score} should be between 0 and 1"
    assert latency_ms >= 0.0, f"Latency {latency_ms} should be non-negative"
    assert isinstance(score, float), "Score should be a float"
    assert isinstance(latency_ms, (int, float)), "Latency should be numeric"

def test_dataset_code_metric_openai():
    """Test with OpenAI whisper model which should have high-quality code"""
    from src.ece461.metricCalcs.metrics import calculate_dataset_code_metric
    
    score, latency_ms = calculate_dataset_code_metric("openai/whisper-tiny")
    
    assert 0.0 <= score <= 1.0
    assert latency_ms >= 0.0