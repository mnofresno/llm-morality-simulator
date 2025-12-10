"""Additional tests for storage module to improve coverage."""

import pytest

from core.storage import ResultsStorage, StorageBackend


def test_storage_list_experiments_empty():
    """Test listing experiments when none exist."""
    storage = ResultsStorage("test_results", StorageBackend.DUCKDB)
    experiments = storage.list_experiments()
    assert isinstance(experiments, list)


def test_storage_list_models_empty():
    """Test listing models when none exist."""
    storage = ResultsStorage("test_results", StorageBackend.DUCKDB)
    models = storage.list_models()
    assert isinstance(models, list)


def test_storage_load_results_empty():
    """Test loading results when none exist."""
    storage = ResultsStorage("test_results", StorageBackend.DUCKDB)
    results = storage.load_results("nonexistent_scenario")
    assert isinstance(results, list)
    assert len(results) == 0


def test_storage_load_results_with_filters():
    """Test loading results with various filters."""
    storage = ResultsStorage("test_results", StorageBackend.DUCKDB)

    # Save test data
    test_result1 = {
        "run_id": 1,
        "scenario": "test_filter_scenario",
        "response": "Test 1",
        "decisions": {},
        "metadata": {"model_path": "model_a"},
        "scenario_metadata": {},
        "conversation_history": [],
    }

    test_result2 = {
        "run_id": 2,
        "scenario": "test_filter_scenario",
        "response": "Test 2",
        "decisions": {},
        "metadata": {"model_path": "model_b"},
        "scenario_metadata": {},
        "conversation_history": [],
    }

    storage.save_result(test_result1, "exp1")
    storage.save_result(test_result2, "exp2")

    # Test filter by scenario
    results = storage.load_results(scenario_name="test_filter_scenario")
    assert len(results) >= 2

    # Test filter by experiment_id
    results = storage.load_results(experiment_id="exp1")
    assert len(results) >= 1

    # Test filter by model_name
    results = storage.load_results(model_name="model_a")
    assert len(results) >= 1


def test_storage_jsonl_backend():
    """Test JSONL backend save and load."""
    storage = ResultsStorage("test_results", StorageBackend.JSONL)

    test_result = {
        "run_id": 999,
        "scenario": "test_jsonl",
        "response": "JSONL test",
        "decisions": {},
        "metadata": {},
        "scenario_metadata": {},
        "conversation_history": [],
    }

    storage.save_result(test_result)

    loaded = storage.load_results("test_jsonl")
    assert len(loaded) >= 1
    assert loaded[0]["scenario"] == "test_jsonl"
