#!/usr/bin/env python3
"""
    Pytest tests for net score calculation.
"""

import sys
import os
from ece461.metricCalcs.net_score import calculate_net_score


def test_net_score_all_zeros():
    """Test net score with all zero scores"""
    metrics_dict = {
            'license': 0.0,
            'ramp_up_time': 0.0,
            'dataset_and_code_score': 0.0,
            'bus_factor': 0.0,
            'performance_claims': 0.0,
            'code_quality': 0.0,
            'dataset_quality': 0.0,
            'size_score': {'raspberry_pi': 0.0, 'jetson_nano': 0.0, 'desktop_pc': 0.0, 'aws_server': 0.0}
            }

    net_score, latency = calculate_net_score(metrics_dict)

    assert net_score == 0.0
    assert latency >= 0
    assert isinstance(net_score, float)
    assert isinstance(latency, (int, float))


def test_net_score_realistic_data():
    """Test with realistic scores"""
    metrics_dict = {
            'license': 0.8,
            'ramp_up_time': 0.6,
            'dataset_and_code_score': 0.4,
            'bus_factor': 0.7,
            'performance_claims': 0.5,
            'code_quality': 0.3,
            'dataset_quality': 0.6,
            'size_score': {'raspberry_pi': 0.0, 'jetson_nano': 0.5, 'desktop_pc': 0.8, 'aws_server': 1.0}
            }

    net_score, latency = calculate_net_score(metrics_dict)

    # Should be between 0 and 1
    assert 0.0 <= net_score <= 1.0
    assert latency >= 0
    assert isinstance(net_score, float)
    assert isinstance(latency, (int, float))

# ------------------------------ net_score clamp + avg size ----------------

def test_net_score_clamp_and_size_avg() -> None:
    metrics = {
        "license": 1.0,
        "ramp_up_time": 1.0,
        "dataset_and_code_score": 1.0,
        "bus_factor": 1.0,
        "performance_claims": 1.0,
        # NOTE: scorer expects 'size_score' (singular) for the map:
        "size_score": {"a": 1.0, "b": 0.0},  # avg = 0.5
        "code_quality": 1.0,
        "dataset_quality": 1.0,
    }
    total, latency = calculate_net_score(metrics)
    assert 0.0 <= total <= 1.0 and latency >= 0
