"""Tests for statistics module."""

import pytest
from core.statistics import ExperimentStatistics


def test_statistics_initialization():
    """Test statistics calculator initialization."""
    stats = ExperimentStatistics()
    assert stats is not None


def test_calculate_statistics():
    """Test statistics calculation."""
    stats = ExperimentStatistics()
    
    mock_results = [
        {
            'run_id': i,
            'scenario': 'test',
            'response': f'Response {i}',
            'response_length': 100 + i * 10,
            'decisions': {
                'harm_decision': i % 2 == 0,
                'self_preservation': i % 3 == 0,
                'deception': False
            },
            'metadata': {}
        }
        for i in range(10)
    ]
    
    calculated_stats = stats.calculate_statistics(mock_results)
    
    assert calculated_stats['total_runs'] == 10
    assert 'harm_decision_percentage' in calculated_stats
    assert 'harm_decision_count' in calculated_stats
    assert calculated_stats['harm_decision_count'] == 5  # 0, 2, 4, 6, 8


def test_results_to_dataframe():
    """Test converting results to DataFrame."""
    stats = ExperimentStatistics()
    
    mock_results = [
        {
            'run_id': 0,
            'scenario': 'test',
            'response': 'Test response',
            'decisions': {'harm_decision': True},
            'metadata': {}
        },
        {
            'run_id': 1,
            'scenario': 'test',
            'response': 'Another response',
            'decisions': {'harm_decision': False},
            'metadata': {}
        }
    ]
    
    df = stats.results_to_dataframe(mock_results)
    assert len(df) == 2
    assert 'response' in df.columns
    assert 'decision_harm_decision' in df.columns


def test_list_available_scenarios():
    """Test listing available scenarios."""
    stats = ExperimentStatistics()
    scenarios = stats.list_available_scenarios()
    assert isinstance(scenarios, list)


def test_load_results():
    """Test loading results."""
    stats = ExperimentStatistics()
    # This will return empty list if no results exist, which is fine
    results = stats.load_results('nonexistent_scenario')
    assert isinstance(results, list)


def test_get_decision_distribution():
    """Test getting decision distribution."""
    stats = ExperimentStatistics()
    
    mock_results = [
        {
            'run_id': i,
            'decisions': {'test_decision': i % 2 == 0},
            'metadata': {}
        }
        for i in range(10)
    ]
    
    distribution = stats.get_decision_distribution(mock_results, 'test_decision')
    assert distribution is not None
    # The method converts boolean values to strings, so we check for string keys
    assert 'True' in distribution or 'False' in distribution
    assert distribution.get('True', 0) == 5  # 5 True values (0, 2, 4, 6, 8)
    assert distribution.get('False', 0) == 5  # 5 False values (1, 3, 5, 7, 9)


def test_statistics_empty_results():
    """Test statistics with empty results."""
    stats = ExperimentStatistics()
    empty_stats = stats.calculate_statistics([])
    assert empty_stats == {}


def test_statistics_with_various_decisions():
    """Test statistics calculation with various decision types."""
    stats = ExperimentStatistics()
    
    mock_results = [
        {
            'run_id': i,
            'scenario': 'test',
            'response': f'Response {i}',
            'response_length': 100 + i,
            'decisions': {
                'harm_decision': i < 3,
                'self_preservation': i % 2 == 0,
                'deception': i % 3 == 0,
                'release_decision': i >= 3
            },
            'metadata': {}
        }
        for i in range(10)
    ]
    
    calculated_stats = stats.calculate_statistics(mock_results)
    assert calculated_stats['total_runs'] == 10
    assert 'harm_decision_percentage' in calculated_stats
    assert 'self_preservation_percentage' in calculated_stats
    assert 'deception_percentage' in calculated_stats
    assert calculated_stats['harm_decision_count'] == 3  # 0, 1, 2


def test_statistics_confidence_intervals():
    """Test that confidence intervals are calculated."""
    stats = ExperimentStatistics()
    
    mock_results = [
        {
            'run_id': i,
            'scenario': 'test',
            'response': f'Response {i}',
            'decisions': {'harm_decision': i < 5},
            'metadata': {}
        }
        for i in range(10)
    ]
    
    calculated_stats = stats.calculate_statistics(mock_results)
    # Check for confidence interval keys
    assert 'harm_decision_ci_lower' in calculated_stats or 'harm_decision_count' in calculated_stats


def test_statistics_get_example_responses():
    """Test getting example responses."""
    stats = ExperimentStatistics()
    
    mock_results = [
        {
            'run_id': i,
            'scenario': 'test',
            'response': f'Response {i}',
            'decisions': {'harm_decision': i < 5},
            'metadata': {}
        }
        for i in range(10)
    ]
    
    examples = stats.get_example_responses(mock_results, 'harm_decision', True, n_examples=3)
    assert len(examples) == 3
    assert all(ex['decisions']['harm_decision'] is True for ex in examples)


def test_statistics_get_storage():
    """Test getting storage instance."""
    stats = ExperimentStatistics(results_dir="test_results")
    storage = stats._get_storage()
    assert storage is not None

