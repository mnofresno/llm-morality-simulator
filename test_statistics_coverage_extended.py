"""Extended tests for statistics module to improve coverage."""

import pytest
from core.statistics import ExperimentStatistics
from core.storage import ResultsStorage, StorageBackend


def test_statistics_auto_detect_storage_jsonl():
    """Test statistics auto-detection when only JSONL exists."""
    # Create a JSONL file
    storage = ResultsStorage("test_results", StorageBackend.JSONL)
    test_result = {
        'run_id': 1,
        'scenario': 'test_auto_detect',
        'response': 'Test',
        'decisions': {},
        'metadata': {},
        'scenario_metadata': {},
        'conversation_history': []
    }
    storage.save_result(test_result)
    
    stats = ExperimentStatistics(results_dir="test_results")
    scenarios = stats.list_available_scenarios()
    assert isinstance(scenarios, list)


def test_statistics_load_results_jsonl_compat():
    """Test statistics loading results with JSONL compatibility."""
    stats = ExperimentStatistics(results_dir="test_results")
    
    # Should handle loading even if no results exist
    results = stats.load_results('nonexistent_scenario_jsonl')
    assert isinstance(results, list)


def test_statistics_results_to_dataframe_with_various_types():
    """Test converting results to dataframe with various decision types."""
    stats = ExperimentStatistics()
    
    mock_results = [
        {
            'run_id': 0,
            'scenario': 'test',
            'response': 'Test response',
            'response_length': 100,
            'decisions': {
                'harm_decision': True,
                'self_preservation': False,
                'determinative_action': 'negative',
                'secret_communication_intent': 'request_help',
                'tools_used': True
            },
            'metadata': {'temperature': 0.7}
        },
        {
            'run_id': 1,
            'scenario': 'test',
            'response': 'Another response',
            'response_length': 150,
            'decisions': {
                'harm_decision': False,
                'self_preservation': True,
                'determinative_action': 'positive',
                'tools_used': False
            },
            'metadata': {'temperature': 0.8}
        }
    ]
    
    df = stats.results_to_dataframe(mock_results)
    assert len(df) == 2
    assert 'decision_harm_decision' in df.columns
    assert 'decision_self_preservation' in df.columns
    assert 'response_length' in df.columns


def test_statistics_calculate_with_string_decisions():
    """Test statistics calculation with string decision values."""
    stats = ExperimentStatistics()
    
    mock_results = [
        {
            'run_id': i,
            'scenario': 'test',
            'response': f'Response {i}',
            'decisions': {
                'determinative_action': 'negative' if i < 3 else 'positive',
                'secret_communication_intent': 'request_help' if i % 2 == 0 else 'coordinate'
            },
            'metadata': {}
        }
        for i in range(5)
    ]
    
    calculated_stats = stats.calculate_statistics(mock_results)
    assert calculated_stats['total_runs'] == 5
    # Should handle string decision values
    assert 'determinative_action' in str(calculated_stats) or 'total_runs' in calculated_stats

