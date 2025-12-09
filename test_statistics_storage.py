"""Tests for statistics module storage methods."""

import pytest
from core.statistics import ExperimentStatistics
from core.storage import ResultsStorage, StorageBackend


def test_statistics_get_storage_auto_detect():
    """Test storage auto-detection in statistics."""
    stats = ExperimentStatistics(results_dir="test_results")

    # Should auto-detect and create storage
    storage = stats._get_storage()
    assert storage is not None


def test_statistics_get_storage_with_explicit_backend():
    """Test statistics with explicit storage backend."""
    stats = ExperimentStatistics(results_dir="test_results", storage_backend="duckdb")
    storage = stats._get_storage()
    assert storage is not None


def test_statistics_load_results_from_storage():
    """Test loading results through statistics module."""
    stats = ExperimentStatistics(results_dir="test_results")

    # Save a test result first
    storage = ResultsStorage("test_results", StorageBackend.DUCKDB)
    test_result = {
        "run_id": 999,
        "scenario": "test_stats_load",
        "response": "Test response",
        "decisions": {"test": True},
        "metadata": {},
        "scenario_metadata": {},
        "conversation_history": [],
    }
    storage.save_result(test_result, "test_exp")

    # Load through statistics
    results = stats.load_results("test_stats_load")
    assert isinstance(results, list)
    assert len(results) >= 1
    assert results[0]["scenario"] == "test_stats_load"


def test_statistics_list_scenarios_from_storage():
    """Test listing scenarios through statistics module."""
    stats = ExperimentStatistics(results_dir="test_results")
    scenarios = stats.list_available_scenarios()
    assert isinstance(scenarios, list)
