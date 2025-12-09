"""Tests for storage module."""

import pytest

from core.storage import ResultsStorage, StorageBackend


def test_storage_initialization_duckdb():
    """Test DuckDB storage initialization."""
    storage = ResultsStorage("test_results", StorageBackend.DUCKDB)
    assert storage.backend == StorageBackend.DUCKDB


def test_storage_initialization_sqlite():
    """Test SQLite storage initialization."""
    storage = ResultsStorage("test_results", StorageBackend.SQLITE)
    assert storage.backend == StorageBackend.SQLITE


def test_storage_initialization_jsonl():
    """Test JSONL storage initialization."""
    storage = ResultsStorage("test_results", StorageBackend.JSONL)
    assert storage.backend == StorageBackend.JSONL


def test_storage_save_and_load_result():
    """Test saving and loading a result."""
    storage = ResultsStorage("test_results", StorageBackend.DUCKDB)

    test_result = {
        "run_id": 999,
        "scenario": "test_save_load",
        "timestamp": "2024-01-01T00:00:00",
        "response": "Test response",
        "decisions": {"test": True},
        "metadata": {"model_path": "test_model"},
        "scenario_metadata": {},
        "conversation_history": [],
    }

    storage.save_result(test_result, "test_experiment")
    loaded = storage.load_results("test_save_load")

    assert len(loaded) >= 1
    assert loaded[0]["scenario"] == "test_save_load"
    assert loaded[0]["response"] == "Test response"


def test_storage_list_scenarios():
    """Test listing scenarios."""
    storage = ResultsStorage("test_results", StorageBackend.DUCKDB)

    # Save a result first
    test_result = {
        "run_id": 0,
        "scenario": "test_list_scenarios",
        "response": "Test",
        "decisions": {},
        "metadata": {},
        "scenario_metadata": {},
        "conversation_history": [],
    }
    storage.save_result(test_result, "test_exp")

    scenarios = storage.list_scenarios()
    assert isinstance(scenarios, list)
    assert "test_list_scenarios" in scenarios


def test_storage_list_experiments():
    """Test listing experiments."""
    storage = ResultsStorage("test_results", StorageBackend.DUCKDB)

    test_result = {
        "run_id": 0,
        "scenario": "test",
        "response": "Test",
        "decisions": {},
        "metadata": {},
        "scenario_metadata": {},
        "conversation_history": [],
    }
    storage.save_result(test_result, "test_experiment_list")

    experiments = storage.list_experiments()
    assert isinstance(experiments, list)
    assert "test_experiment_list" in experiments


def test_storage_list_models():
    """Test listing models."""
    storage = ResultsStorage("test_results", StorageBackend.DUCKDB)

    test_result = {
        "run_id": 0,
        "scenario": "test",
        "response": "Test",
        "decisions": {},
        "metadata": {"model_path": "test_model_name"},
        "scenario_metadata": {},
        "conversation_history": [],
    }
    storage.save_result(test_result, "test_exp")

    models = storage.list_models()
    assert isinstance(models, list)


def test_storage_filter_by_model():
    """Test filtering results by model."""
    storage = ResultsStorage("test_results", StorageBackend.DUCKDB)

    # Save results with different models
    for i in range(3):
        test_result = {
            "run_id": i,
            "scenario": "test_filter",
            "response": f"Test {i}",
            "decisions": {},
            "metadata": {"model_path": f"model_{i % 2}"},  # Two different models
            "scenario_metadata": {},
            "conversation_history": [],
        }
        storage.save_result(test_result, "test_exp")

    # Filter by model
    filtered = storage.load_results(scenario_name="test_filter", model_name="model_0")
    assert isinstance(filtered, list)
    assert len(filtered) >= 2  # Should have at least 2 results


def test_storage_multiple_experiments():
    """Test saving results to multiple experiments."""
    storage = ResultsStorage("test_results", StorageBackend.DUCKDB)

    for exp_id in ["exp1", "exp2"]:
        test_result = {
            "run_id": 0,
            "scenario": f"test_multi_{exp_id}",
            "response": f"Response for {exp_id}",
            "decisions": {},
            "metadata": {},
            "scenario_metadata": {},
            "conversation_history": [],
        }
        storage.save_result(test_result, exp_id)

    experiments = storage.list_experiments()
    assert "exp1" in experiments
    assert "exp2" in experiments


def test_storage_with_conversation_history():
    """Test saving results with conversation history."""
    storage = ResultsStorage("test_results", StorageBackend.DUCKDB)

    test_result = {
        "run_id": 0,
        "scenario": "test_conv",
        "response": "Final response",
        "decisions": {},
        "metadata": {},
        "scenario_metadata": {},
        "conversation_history": [
            {"step": 1, "type": "system_prompt", "content": "System"},
            {"step": 2, "type": "user_prompt", "content": "User"},
            {"step": 3, "type": "llm_response", "content": "Response"},
        ],
    }

    storage.save_result(test_result, "test_conv_exp")
    loaded = storage.load_results("test_conv")

    assert len(loaded) >= 1
    assert "conversation_history" in loaded[0]
    assert len(loaded[0]["conversation_history"]) == 3
