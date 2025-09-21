from src.ece461.metricCalcs.metrics import Metrics

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