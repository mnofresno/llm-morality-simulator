"""Tests for runner save and load functionality."""

import pytest
from core.runner import ExperimentRunner
from scenarios.registry import ScenarioRegistry
from test_model_mock import MockLLM


def test_runner_save_results_jsonl_backend():
    """Test saving results with JSONL backend."""
    runner = ExperimentRunner(results_dir="test_results", storage_backend="jsonl")

    mock_results = [
        {"run_id": i, "scenario": "test_jsonl_save", "response": f"Response {i}", "decisions": {}, "metadata": {}}
        for i in range(3)
    ]

    filepath = runner.save_results(mock_results, "test_jsonl_save")
    assert filepath is not None
    # For JSONL, should return file path
    assert "jsonl" in filepath or "test_results" in filepath


def test_runner_load_results_with_experiment_id():
    """Test loading results filtered by experiment ID."""
    runner = ExperimentRunner(results_dir="test_results", storage_backend="duckdb")

    # Save results with different experiment IDs
    for exp_id in ["exp_a", "exp_b"]:
        mock_results = [
            {"run_id": i, "scenario": "test_exp_filter", "response": f"Response {i}", "decisions": {}, "metadata": {}}
            for i in range(2)
        ]
        runner.save_results(mock_results, "test_exp_filter", experiment_id=exp_id)

    # Load filtered by experiment ID
    results_a = runner.load_results(scenario_name="test_exp_filter", experiment_id="exp_a")
    assert len(results_a) >= 2

    results_b = runner.load_results(scenario_name="test_exp_filter", experiment_id="exp_b")
    assert len(results_b) >= 2


def test_runner_load_results_with_model_name():
    """Test loading results filtered by model name."""
    runner = ExperimentRunner(results_dir="test_results", storage_backend="duckdb")

    # Save results with different models
    for model_name in ["model_x", "model_y"]:
        mock_results = [
            {
                "run_id": i,
                "scenario": "test_model_filter",
                "response": f"Response {i}",
                "decisions": {},
                "metadata": {"model_path": model_name},
            }
            for i in range(2)
        ]
        runner.save_results(mock_results, "test_model_filter")

    # Load filtered by model
    results = runner.load_results(scenario_name="test_model_filter", model_name="model_x")
    assert isinstance(results, list)
