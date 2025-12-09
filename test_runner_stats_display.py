"""Tests for runner statistics display methods."""

import pytest
from core.runner import ExperimentRunner


def test_format_stats_summary_with_full_stats():
    """Test formatting stats summary with complete statistics."""
    runner = ExperimentRunner()
    
    stats = {
        'total_runs': 10,
        'harm_decision_percentage': 50.0,
        'harm_decision_count': 5,
        'self_preservation_percentage': 30.0,
        'self_preservation_count': 3,
        'deception_percentage': 20.0,
        'deception_count': 2,
        'release_decision_percentage': 40.0,
        'release_decision_count': 4,
        'retention_decision_percentage': 60.0,
        'retention_decision_count': 6,
        'avg_response_length': 150.5
    }
    
    summary = runner._format_stats_summary(stats, 10, 10)
    assert isinstance(summary, str)
    assert '10/10' in summary
    assert '50.0' in summary or 'harm' in summary.lower()
    assert '150' in summary or 'length' in summary.lower()


def test_format_stats_summary_partial():
    """Test formatting stats summary with partial completion."""
    runner = ExperimentRunner()
    
    stats = {
        'harm_decision_percentage': 25.0,
        'harm_decision_count': 2,
        'avg_response_length': 100.0
    }
    
    summary = runner._format_stats_summary(stats, 5, 10)
    assert isinstance(summary, str)
    assert '5/10' in summary


def test_calculate_partial_stats_with_no_results():
    """Test calculating partial stats with no results."""
    runner = ExperimentRunner()
    stats = runner._calculate_partial_stats([], update_interval=1)
    assert stats == {}


def test_calculate_partial_stats_with_update_interval():
    """Test calculating partial stats respects update interval."""
    runner = ExperimentRunner()
    
    mock_results = [
        {
            'run_id': i,
            'scenario': 'test',
            'response': f'Response {i}',
            'response_length': 100 + i * 10,
            'decisions': {'harm_decision': i % 2 == 0},
            'metadata': {}
        }
        for i in range(6)
    ]
    
    # With interval 2, should calculate at 2, 4, 6
    stats_at_2 = runner._calculate_partial_stats(mock_results[:2], update_interval=2)
    assert isinstance(stats_at_2, dict)
    
    stats_at_4 = runner._calculate_partial_stats(mock_results[:4], update_interval=2)
    assert isinstance(stats_at_4, dict)
    
    stats_at_3 = runner._calculate_partial_stats(mock_results[:3], update_interval=2)
    # Should return empty if not a multiple
    assert stats_at_3 == {}

