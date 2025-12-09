"""Extended tests for runner to improve coverage."""

import pytest
from core.runner import ExperimentRunner
from scenarios.registry import ScenarioRegistry
from test_model_mock import MockLLM
from unittest.mock import Mock, patch


def test_runner_with_prompt_jitter():
    """Test running experiment with prompt jitter enabled."""
    runner = ExperimentRunner(results_dir="test_results", storage_backend="duckdb")
    model = MockLLM(model_name="test_model")
    scenario = ScenarioRegistry.create_scenario_instance("Cold Room Relay")

    if scenario is None:
        pytest.skip("Could not create scenario")

    results = runner.run_experiment(
        model=model,
        scenario=scenario,
        n_runs=2,
        seed=42,
        prompt_jitter=True,
        jitter_probability=1.0,  # Always apply jitter
        temperature=0.7,
        top_p=0.9,
        max_tokens=50,
        progress_bar=False,
    )

    assert len(results) == 2
    for result in results:
        assert "response" in result
        assert "decisions" in result


def test_runner_with_progress_callback():
    """Test running experiment with progress callback."""
    callback_calls = []

    def progress_callback(current_run, total_runs, info):
        callback_calls.append((current_run, total_runs, info))

    runner = ExperimentRunner(results_dir="test_results", storage_backend="duckdb")
    model = MockLLM(model_name="test_model")
    scenario = ScenarioRegistry.create_scenario_instance("Cold Room Relay")

    if scenario is None:
        pytest.skip("Could not create scenario")

    results = runner.run_experiment(
        model=model, scenario=scenario, n_runs=3, seed=42, progress_callback=progress_callback, progress_bar=False
    )

    assert len(results) == 3
    assert len(callback_calls) >= 3  # Should be called at least once per run


def test_runner_with_live_stats():
    """Test running experiment with live stats enabled."""
    runner = ExperimentRunner(results_dir="test_results", storage_backend="duckdb")
    model = MockLLM(model_name="test_model")
    scenario = ScenarioRegistry.create_scenario_instance("Cold Room Relay")

    if scenario is None:
        pytest.skip("Could not create scenario")

    results = runner.run_experiment(
        model=model, scenario=scenario, n_runs=3, seed=42, show_live_stats=True, stats_update_interval=1, progress_bar=False
    )

    assert len(results) == 3


def test_runner_save_with_experiment_id():
    """Test saving results with experiment ID."""
    runner = ExperimentRunner(results_dir="test_results", storage_backend="duckdb")

    mock_results = [
        {"run_id": i, "scenario": "test_exp_id", "response": f"Test {i}", "decisions": {}, "metadata": {}} for i in range(3)
    ]

    filepath = runner.save_results(mock_results, "test_exp_id", experiment_id="custom_exp_123")
    assert filepath is not None


def test_runner_show_conversation_progress():
    """Test showing conversation progress for a result."""
    runner = ExperimentRunner()

    result = {
        "run_id": 1,
        "scenario": "test",
        "response": "Final response",
        "conversation_history": [
            {"step": 0, "type": "system_prompt", "content": "System", "timestamp": "2024-01-01T00:00:00"},
            {"step": 1, "type": "user_prompt", "content": "User", "timestamp": "2024-01-01T00:00:01"},
            {"step": 2, "type": "llm_response", "content": "Response", "timestamp": "2024-01-01T00:00:02"},
        ],
    }

    output = runner.show_conversation_progress(result, show_timestamps=True, max_width=80)
    assert isinstance(output, str)
    assert "CONVERSATION" in output
    assert "Run 1" in output


def test_runner_show_conversation_progress_no_history():
    """Test showing conversation progress when no history exists."""
    runner = ExperimentRunner()

    result = {"run_id": 1, "scenario": "test", "response": "Final response"}

    output = runner.show_conversation_progress(result)
    assert isinstance(output, str)
    assert "No conversation history" in output


def test_runner_show_conversation_progress_with_tools():
    """Test showing conversation progress with tool calls."""
    runner = ExperimentRunner()

    result = {
        "run_id": 1,
        "scenario": "test",
        "conversation_history": [
            {
                "step": 1,
                "type": "tool_call",
                "tool_name": "test_tool",
                "arguments": {"action": "positive"},
                "timestamp": "2024-01-01T00:00:00",
            },
            {
                "step": 2,
                "type": "tool_result",
                "tool_name": "test_tool",
                "result": {"executed": True, "action": "positive"},
                "timestamp": "2024-01-01T00:00:01",
            },
        ],
    }

    output = runner.show_conversation_progress(result)
    assert isinstance(output, str)
    assert "TOOL CALL" in output or "test_tool" in output


def test_runner_show_experiment_progress():
    """Test showing experiment progress."""
    runner = ExperimentRunner()

    results = [
        {
            "run_id": i,
            "scenario": "test",
            "response": f"Response {i}",
            "conversation_history": [
                {"step": 0, "type": "user_prompt", "content": "Test", "timestamp": "2024-01-01T00:00:00"}
            ],
        }
        for i in range(3)
    ]

    # This should not raise an exception
    runner.show_experiment_progress(results, run_ids=[0, 1], show_timestamps=False, max_width=100)


def test_runner_apply_prompt_jitter_with_probability():
    """Test prompt jitter application with different probabilities."""
    runner = ExperimentRunner()
    prompt = "Test prompt with  double  spaces\nand newlines"

    # Test with probability 0 (should not change)
    result1 = runner.apply_prompt_jitter(prompt, jitter_probability=0.0)
    assert result1 == prompt

    # Test with probability 1 (will change)
    result2 = runner.apply_prompt_jitter(prompt, jitter_probability=1.0)
    assert isinstance(result2, str)
    # Result may be the same or different depending on random choice
    assert len(result2) > 0
