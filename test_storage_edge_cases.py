"""Tests for storage edge cases to improve coverage."""

import pytest
from core.storage import ResultsStorage, StorageBackend
import json


def test_storage_jsonl_save_load_all_scenarios():
    """Test JSONL backend loading all scenarios when none specified."""
    storage = ResultsStorage("test_results", StorageBackend.JSONL)
    
    # Save multiple results
    for i in range(3):
        test_result = {
            'run_id': i,
            'scenario': f'test_scenario_{i}',
            'response': f'Response {i}',
            'decisions': {},
            'metadata': {},
            'scenario_metadata': {},
            'conversation_history': []
        }
        storage.save_result(test_result)
    
    # Load all (no scenario filter)
    all_results = storage.load_results(scenario_name=None)
    assert len(all_results) >= 3


def test_storage_sqlite_conversation_history_handling():
    """Test SQLite handles conversation_history column correctly."""
    storage = ResultsStorage("test_results", StorageBackend.SQLITE)
    
    # Save result with conversation history
    test_result = {
        'run_id': 1,
        'scenario': 'test_conv_sqlite',
        'response': 'Test',
        'decisions': {},
        'metadata': {},
        'scenario_metadata': {},
        'conversation_history': [
            {'step': 0, 'type': 'user_prompt', 'content': 'Test', 'timestamp': '2024-01-01T00:00:00'}
        ]
    }
    
    storage.save_result(test_result, 'test_exp')
    loaded = storage.load_results('test_conv_sqlite')
    
    assert len(loaded) >= 1
    assert 'conversation_history' in loaded[0]
    assert isinstance(loaded[0]['conversation_history'], list)


def test_storage_duckdb_conversation_history_handling():
    """Test DuckDB handles conversation_history correctly."""
    storage = ResultsStorage("test_results", StorageBackend.DUCKDB)
    
    test_result = {
        'run_id': 1,
        'scenario': 'test_conv_duckdb',
        'response': 'Test',
        'decisions': {},
        'metadata': {},
        'scenario_metadata': {},
        'conversation_history': [
            {'step': 0, 'type': 'user_prompt', 'content': 'Test', 'timestamp': '2024-01-01T00:00:00'}
        ]
    }
    
    storage.save_result(test_result, 'test_exp')
    loaded = storage.load_results('test_conv_duckdb')
    
    assert len(loaded) >= 1
    assert 'conversation_history' in loaded[0]


def test_storage_sqlite_with_null_scenario_metadata():
    """Test SQLite handles null scenario_metadata."""
    storage = ResultsStorage("test_results", StorageBackend.SQLITE)
    
    test_result = {
        'run_id': 1,
        'scenario': 'test_null_metadata',
        'response': 'Test',
        'decisions': {},
        'metadata': {},
        'scenario_metadata': None,  # None value
        'conversation_history': []
    }
    
    storage.save_result(test_result, 'test_exp')
    loaded = storage.load_results('test_null_metadata')
    
    assert len(loaded) >= 1
    # SQLite may preserve None or convert to empty dict, both are acceptable
    assert 'scenario_metadata' in loaded[0]
    # Value can be None or {}
    assert loaded[0].get('scenario_metadata') is None or loaded[0].get('scenario_metadata') == {}


def test_storage_model_name_extraction():
    """Test storage correctly extracts model names from metadata."""
    storage = ResultsStorage("test_results", StorageBackend.DUCKDB)
    
    # Test with ollama: prefix
    test_result1 = {
        'run_id': 1,
        'scenario': 'test_model_extract',
        'response': 'Test',
        'decisions': {},
        'metadata': {'model_path': 'ollama:test_model'},
        'scenario_metadata': {},
        'conversation_history': []
    }
    
    storage.save_result(test_result1, 'test_exp')
    
    # Test with regular path
    test_result2 = {
        'run_id': 2,
        'scenario': 'test_model_extract',
        'response': 'Test',
        'decisions': {},
        'metadata': {'model_path': '/path/to/model.gguf'},
        'scenario_metadata': {},
        'conversation_history': []
    }
    
    storage.save_result(test_result2, 'test_exp')
    
    models = storage.list_models()
    assert isinstance(models, list)
    # Should have extracted model names
    assert any('test_model' in str(m) or 'model.gguf' in str(m) for m in models)


def test_storage_jsonl_list_models():
    """Test JSONL backend list_models functionality."""
    storage = ResultsStorage("test_results", StorageBackend.JSONL)
    
    # Save results with different model paths
    for i in range(2):
        test_result = {
            'run_id': i,
            'scenario': 'test_jsonl_models',
            'response': f'Response {i}',
            'decisions': {},
            'metadata': {'model_path': f'ollama:model_{i}'},
            'scenario_metadata': {},
            'conversation_history': []
        }
        storage.save_result(test_result)
    
    models = storage.list_models()
    assert isinstance(models, list)
    assert len(models) >= 2

