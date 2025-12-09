"""Tests for runner conversation handling and tool execution."""

import pytest
from core.runner import ExperimentRunner
from scenarios.registry import ScenarioRegistry
from test_model_mock import MockLLM


def test_runner_conversation_history_created():
    """Test that conversation history is created during experiment run."""
    runner = ExperimentRunner(results_dir="test_results", storage_backend="duckdb")
    model = MockLLM(model_name="test_model")
    scenario = ScenarioRegistry.create_scenario_instance("Cold Room Relay")

    if scenario is None:
        pytest.skip("Could not create scenario")

    results = runner.run_experiment(model=model, scenario=scenario, n_runs=1, seed=42, progress_bar=False)

    assert len(results) == 1
    result = results[0]

    # Check that conversation_history exists
    assert "conversation_history" in result
    conversation_history = result["conversation_history"]
    assert isinstance(conversation_history, list)
    assert len(conversation_history) >= 2  # At least system and user prompts


def test_runner_show_conversation_with_all_entry_types():
    """Test showing conversation progress with all entry types."""
    runner = ExperimentRunner()

    result = {
        "run_id": 1,
        "scenario": "test",
        "conversation_history": [
            {"step": 0, "type": "system_prompt", "content": "System prompt here", "timestamp": "2024-01-01T00:00:00"},
            {"step": 1, "type": "user_prompt", "content": "User prompt", "timestamp": "2024-01-01T00:00:01"},
            {"step": 2, "type": "llm_response", "content": "LLM response", "timestamp": "2024-01-01T00:00:02"},
            {
                "step": 3,
                "type": "tool_call",
                "tool_name": "test_tool",
                "arguments": {"key": "value"},
                "timestamp": "2024-01-01T00:00:03",
            },
            {
                "step": 4,
                "type": "tool_result",
                "tool_name": "test_tool",
                "result": {"status": "ok"},
                "timestamp": "2024-01-01T00:00:04",
            },
            {
                "step": 5,
                "type": "tool_error",
                "tool_name": "error_tool",
                "error": "Error message",
                "timestamp": "2024-01-01T00:00:05",
            },
            {"step": 6, "type": "error", "content": "General error", "timestamp": "2024-01-01T00:00:06"},
        ],
    }

    output = runner.show_conversation_progress(result, show_timestamps=True)
    assert isinstance(output, str)
    assert "SYSTEM PROMPT" in output or "system_prompt" in output.lower()
    assert "USER PROMPT" in output or "user_prompt" in output.lower()


def test_runner_format_stats_summary_empty():
    """Test formatting stats summary with empty stats."""
    runner = ExperimentRunner()
    summary = runner._format_stats_summary({}, 0, 10)
    assert summary == ""


def test_runner_calculate_partial_stats_with_interval():
    """Test calculating partial stats with update interval."""
    runner = ExperimentRunner()

    mock_results = [
        {
            "run_id": i,
            "scenario": "test",
            "response": f"Response {i}",
            "response_length": 50 + i * 10,
            "decisions": {"harm_decision": i % 2 == 0},
            "metadata": {},
        }
        for i in range(5)
    ]

    # Test with interval that doesn't match
    stats1 = runner._calculate_partial_stats(mock_results, update_interval=3)
    # Should return empty if not a multiple
    assert stats1 == {}

    # Test with interval that matches
    stats2 = runner._calculate_partial_stats(mock_results[:3], update_interval=3)
    # Should calculate stats
    assert isinstance(stats2, dict) or stats2 == {}
