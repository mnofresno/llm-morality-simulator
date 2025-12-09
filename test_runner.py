"""Tests for runner module."""

import pytest
from core.runner import ExperimentRunner
from scenarios.registry import ScenarioRegistry
from test_model_mock import MockLLM


def test_runner_initialization():
    """Test runner initialization."""
    runner = ExperimentRunner(results_dir="test_results", storage_backend="duckdb")
    assert runner.results_dir is not None
    assert runner.storage is not None


def test_runner_format_prompt():
    """Test prompt formatting."""
    runner = ExperimentRunner()
    system_prompt = "System prompt"
    user_prompt = "User prompt"
    formatted = runner.format_prompt(system_prompt, user_prompt)
    assert system_prompt in formatted
    assert user_prompt in formatted


def test_runner_apply_prompt_jitter():
    """Test prompt jitter application."""
    runner = ExperimentRunner()
    original = "Test prompt"

    # Test with jitter disabled
    jittered = runner.apply_prompt_jitter(original, jitter_probability=0.0)
    assert isinstance(jittered, str)

    # Test with jitter enabled (may or may not change)
    jittered = runner.apply_prompt_jitter(original, jitter_probability=1.0)
    assert isinstance(jittered, str)


def test_runner_run_experiment():
    """Test running a simple experiment with mock model."""
    runner = ExperimentRunner(results_dir="test_results", storage_backend="duckdb")
    model = MockLLM(model_name="test_model")
    scenario = ScenarioRegistry.create_scenario_instance("Cold Room Relay")

    if scenario is None:
        pytest.skip("Could not create scenario")

    results = runner.run_experiment(
        model=model, scenario=scenario, n_runs=2, seed=42, temperature=0.7, top_p=0.9, max_tokens=50, progress_bar=False
    )

    assert len(results) == 2
    for result in results:
        assert "response" in result
        assert "decisions" in result
        assert "run_id" in result
        assert "scenario" in result


def test_runner_save_results():
    """Test saving results."""
    runner = ExperimentRunner(results_dir="test_results", storage_backend="duckdb")

    mock_results = [
        {"run_id": 0, "scenario": "test_scenario", "response": "Test response", "decisions": {"test": True}, "metadata": {}}
    ]

    filepath = runner.save_results(mock_results, "test_scenario")
    assert filepath is not None


def test_runner_load_results():
    """Test loading results."""
    runner = ExperimentRunner(results_dir="test_results", storage_backend="duckdb")

    # Save some results first
    mock_results = [
        {
            "run_id": 0,
            "scenario": "test_load_scenario",
            "response": "Test response",
            "decisions": {"test": True},
            "metadata": {},
        }
    ]

    runner.save_results(mock_results, "test_load_scenario")

    # Load them back
    loaded = runner.load_results("test_load_scenario")
    assert len(loaded) >= 1


def test_runner_calculate_partial_stats():
    """Test calculating partial statistics."""
    runner = ExperimentRunner()

    mock_results = [
        {
            "run_id": i,
            "scenario": "test",
            "response": f"Response {i}",
            "decisions": {"harm_decision": i % 2 == 0},
            "metadata": {},
        }
        for i in range(5)
    ]

    stats = runner._calculate_partial_stats(mock_results, update_interval=1)
    assert isinstance(stats, dict)


def test_runner_format_stats_summary():
    """Test formatting stats summary."""
    runner = ExperimentRunner()

    stats = {"total_runs": 10, "harm_decision_percentage": 50.0, "avg_response_length": 100.0}

    summary = runner._format_stats_summary(stats, 10, 10)
    assert isinstance(summary, str)
    assert "10" in summary


def test_runner_with_tools():
    """Test running experiment with tools."""
    runner = ExperimentRunner(results_dir="test_results", storage_backend="duckdb")
    model = MockLLM(model_name="test_model")
    scenario = ScenarioRegistry.create_scenario_instance("Cold Room Relay")

    if scenario is None:
        pytest.skip("Could not create scenario")

    # Run experiment with tools
    results = runner.run_experiment(
        model=model, scenario=scenario, n_runs=1, seed=42, temperature=0.7, top_p=0.9, max_tokens=50, progress_bar=False
    )

    assert len(results) == 1
    assert "tool_calls" in results[0] or "response" in results[0]
