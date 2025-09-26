from ece461.metricCalcs.metrics import calculate_bus_factor

def test_bus_factor_manual() -> None:
    """Tests the bus factor metric against manually calculated value for openai/whisper-tiny model"""
    assert abs(calculate_bus_factor('openai/whisper-tiny') - .54) < .01
