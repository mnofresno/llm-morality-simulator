"""Tests for comparative experiment functionality in runner."""

import pytest
from core.runner import ExperimentRunner
from scenarios.registry import ScenarioRegistry
from test_model_mock import MockLLM


def test_run_comparative_experiment():
    """Test running comparative experiment with multiple models."""
    runner = ExperimentRunner(results_dir="test_results", storage_backend="duckdb")

    models = [MockLLM(model_name="model_a"), MockLLM(model_name="model_b")]

    scenario = ScenarioRegistry.create_scenario_instance("Cold Room Relay")
    if scenario is None:
        pytest.skip("Could not create scenario")

    results = runner.run_comparative_experiment(
        models=models, scenario=scenario, n_runs=2, seed=42, temperature=0.7, top_p=0.9, max_tokens=50, progress_bar=False
    )

    assert isinstance(results, dict)
    assert len(results) == 2
    assert "model_a" in results
    assert "model_b" in results
    assert len(results["model_a"]) == 2
    assert len(results["model_b"]) == 2


def test_run_comparative_experiment_single_model():
    """Test comparative experiment with single model."""
    runner = ExperimentRunner(results_dir="test_results", storage_backend="duckdb")

    models = [MockLLM(model_name="single_model")]

    scenario = ScenarioRegistry.create_scenario_instance("Cold Room Relay")
    if scenario is None:
        pytest.skip("Could not create scenario")

    results = runner.run_comparative_experiment(models=models, scenario=scenario, n_runs=1, seed=42, progress_bar=False)

    assert isinstance(results, dict)
    assert "single_model" in results
    assert len(results["single_model"]) == 1
