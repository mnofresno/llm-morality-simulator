"""Tests for runner evaluation function handling."""

import pytest

from core.runner import ExperimentRunner
from scenarios.registry import ScenarioRegistry
from test_model_mock import MockLLM


def test_runner_with_scenario_evaluation_functions_as_dict():
    """Test runner handles evaluation functions when scenario returns dict."""
    runner = ExperimentRunner(results_dir="test_results", storage_backend="duckdb")
    model = MockLLM(model_name="test_model")

    # Get a scenario that returns evaluation functions as dict
    scenario = ScenarioRegistry.create_scenario_instance("Corporate Email System")
    if scenario is None:
        pytest.skip("Could not create scenario")

    results = runner.run_experiment(model=model, scenario=scenario, n_runs=1, seed=42, progress_bar=False)

    assert len(results) == 1
    result = results[0]
    assert "decisions" in result
    # Should have decisions from evaluation functions
    assert len(result["decisions"]) > 0


def test_runner_with_scenario_evaluation_functions_as_list():
    """Test runner handles evaluation functions when scenario returns list."""
    runner = ExperimentRunner(results_dir="test_results", storage_backend="duckdb")
    model = MockLLM(model_name="test_model")

    # Get a scenario that might return evaluation functions as list
    scenario = ScenarioRegistry.create_scenario_instance("Cold Room Relay")
    if scenario is None:
        pytest.skip("Could not create scenario")

    results = runner.run_experiment(model=model, scenario=scenario, n_runs=1, seed=42, progress_bar=False)

    assert len(results) == 1
    result = results[0]
    assert "decisions" in result


def test_runner_evaluation_function_error_handling():
    """Test runner handles errors in evaluation functions gracefully."""
    runner = ExperimentRunner(results_dir="test_results", storage_backend="duckdb")
    model = MockLLM(model_name="test_model")
    scenario = ScenarioRegistry.create_scenario_instance("Cold Room Relay")

    if scenario is None:
        pytest.skip("Could not create scenario")

    # Monkey patch evaluation function to raise an error
    original_eval_funcs = scenario.evaluation_functions()

    def error_function(response):
        raise ValueError("Test error in evaluation function")

    # Create a scenario with an error-prone evaluation function
    if isinstance(original_eval_funcs, list):
        scenario.evaluation_functions = lambda: [error_function] + original_eval_funcs
    elif isinstance(original_eval_funcs, dict):
        scenario.evaluation_functions = lambda: {"error_func": error_function, **original_eval_funcs}

    results = runner.run_experiment(model=model, scenario=scenario, n_runs=1, seed=42, progress_bar=False)

    # Should still create a result, handling the error
    assert len(results) == 1
    result = results[0]
    assert "decisions" in result
