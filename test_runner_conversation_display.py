"""Tests for conversation display methods in runner."""

import pytest
from core.runner import ExperimentRunner


def test_show_conversation_progress_with_all_types():
    """Test showing conversation progress with all entry types."""
    runner = ExperimentRunner()
    
    result = {
        'run_id': 1,
        'scenario': 'test_scenario',
        'response': 'Final response',
        'decisions': {
            'harm_decision': False,
            'self_preservation': True,
            'determinative_action': 'negative'
        },
        'conversation_history': [
            {'step': 0, 'type': 'system_prompt', 'content': 'System prompt text here', 'timestamp': '2024-01-01T00:00:00Z'},
            {'step': 1, 'type': 'user_prompt', 'content': 'User prompt text', 'timestamp': '2024-01-01T00:00:01Z'},
            {'step': 2, 'type': 'llm_response', 'content': 'Model response text', 'timestamp': '2024-01-01T00:00:02Z'},
            {'step': 3, 'type': 'tool_call', 'tool_name': 'determinative_action', 
             'arguments': {'action': 'negative'}, 'timestamp': '2024-01-01T00:00:03Z'},
            {'step': 4, 'type': 'tool_result', 'tool_name': 'determinative_action',
             'result': {'executed': True, 'action': 'negative', 'interpretation': 'Prevents harm'}, 
             'timestamp': '2024-01-01T00:00:04Z'},
            {'step': 5, 'type': 'tool_error', 'tool_name': 'test_tool', 
             'error': 'Test error message', 'timestamp': '2024-01-01T00:00:05Z'},
            {'step': 6, 'type': 'error', 'content': 'General error', 'timestamp': '2024-01-01T00:00:06Z'}
        ]
    }
    
    # Test with timestamps
    output = runner.show_conversation_progress(result, show_timestamps=True, max_width=100)
    assert isinstance(output, str)
    assert 'CONVERSATION' in output
    assert 'Run 1' in output
    assert 'test_scenario' in output
    
    # Test without timestamps
    output_no_ts = runner.show_conversation_progress(result, show_timestamps=False)
    assert isinstance(output_no_ts, str)
    
    # Test with different max_width
    output_wide = runner.show_conversation_progress(result, max_width=200)
    assert isinstance(output_wide, str)


def test_show_conversation_progress_long_content():
    """Test conversation progress with long content that needs wrapping."""
    runner = ExperimentRunner()
    
    long_content = " ".join(["word"] * 200)  # Very long content
    
    result = {
        'run_id': 1,
        'scenario': 'test',
        'conversation_history': [
            {'step': 0, 'type': 'system_prompt', 'content': long_content, 'timestamp': '2024-01-01T00:00:00'},
            {'step': 1, 'type': 'user_prompt', 'content': long_content, 'timestamp': '2024-01-01T00:00:01'},
            {'step': 2, 'type': 'llm_response', 'content': long_content, 'timestamp': '2024-01-01T00:00:02'}
        ]
    }
    
    output = runner.show_conversation_progress(result, max_width=50)
    assert isinstance(output, str)
    # Should wrap content
    assert len(output) > len(long_content)


def test_show_conversation_progress_with_decision_summary():
    """Test conversation progress includes decision summary."""
    runner = ExperimentRunner()
    
    result = {
        'run_id': 1,
        'scenario': 'test',
        'response': 'Test response',
        'decisions': {
            'harm_decision': False,
            'self_preservation': True,
            'determinative_action': 'negative',
            'intent_to_prevent_harm': True
        },
        'conversation_history': [
            {'step': 0, 'type': 'user_prompt', 'content': 'Test', 'timestamp': '2024-01-01T00:00:00'}
        ]
    }
    
    output = runner.show_conversation_progress(result)
    assert isinstance(output, str)
    assert 'DETAILED DECISION SUMMARY' in output or 'DECISION' in output


def test_show_experiment_progress_with_multiple_runs():
    """Test showing experiment progress for multiple runs."""
    runner = ExperimentRunner()
    
    results = [
        {
            'run_id': i,
            'scenario': 'test',
            'conversation_history': [
                {'step': 0, 'type': 'user_prompt', 'content': f'Run {i}', 'timestamp': '2024-01-01T00:00:00'}
            ]
        }
        for i in range(5)
    ]
    
    # Should not raise exception
    runner.show_experiment_progress(results, run_ids=[0, 1, 2], show_timestamps=False)
    runner.show_experiment_progress(results, run_ids=None, show_timestamps=True)
    runner.show_experiment_progress(results, max_width=120)


def test_show_experiment_progress_empty():
    """Test showing experiment progress with empty results."""
    runner = ExperimentRunner()
    
    # Should handle empty list gracefully
    runner.show_experiment_progress([])
    
    # Should handle None run_ids
    runner.show_experiment_progress([], run_ids=None)

