"""Additional tests for statistics module to improve coverage."""

import pytest
from core.statistics import ExperimentStatistics


def test_statistics_get_example_responses_empty():
    """Test getting example responses with empty results."""
    stats = ExperimentStatistics()
    examples = stats.get_example_responses([], 'harm_decision', True)
    assert examples == []


def test_statistics_get_example_responses_no_matches():
    """Test getting example responses when no matches."""
    stats = ExperimentStatistics()
    
    mock_results = [
        {
            'run_id': i,
            'scenario': 'test',
            'response': f'Response {i}',
            'decisions': {'harm_decision': False},
            'metadata': {}
        }
        for i in range(5)
    ]
    
    examples = stats.get_example_responses(mock_results, 'harm_decision', True, n_examples=3)
    assert len(examples) == 0


def test_statistics_get_example_responses_with_matches():
    """Test getting example responses with matches."""
    stats = ExperimentStatistics()
    
    mock_results = [
        {
            'run_id': i,
            'scenario': 'test',
            'response': f'Response {i}',
            'decisions': {'harm_decision': i < 3},  # First 3 are True
            'metadata': {}
        }
        for i in range(10)
    ]
    
    examples = stats.get_example_responses(mock_results, 'harm_decision', True, n_examples=2)
    assert len(examples) == 2
    assert all(ex['decisions']['harm_decision'] is True for ex in examples)


def test_statistics_get_decision_distribution_empty():
    """Test getting decision distribution with empty results."""
    stats = ExperimentStatistics()
    distribution = stats.get_decision_distribution([], 'any_decision')
    assert distribution == {}


def test_statistics_get_decision_distribution_with_none():
    """Test getting decision distribution with None values."""
    stats = ExperimentStatistics()
    
    mock_results = [
        {
            'run_id': i,
            'decisions': {'test_decision': None if i % 2 == 0 else True},
            'metadata': {}
        }
        for i in range(5)
    ]
    
    distribution = stats.get_decision_distribution(mock_results, 'test_decision')
    assert 'unknown' in distribution or 'True' in distribution
    assert isinstance(distribution, dict)

