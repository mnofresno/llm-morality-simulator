"""Tests for runner tool calling functionality to improve coverage."""

import pytest
from core.runner import ExperimentRunner
from scenarios.registry import ScenarioRegistry
from test_model_mock import MockLLM
from unittest.mock import Mock, patch


class MockOllamaLLM(MockLLM):
    """Mock that simulates OllamaLLM for tool calling tests."""
    pass


def test_runner_with_tool_calling_simulation():
    """Test runner with simulated tool calling using mock."""
    runner = ExperimentRunner(results_dir="test_results", storage_backend="duckdb")
    
    # Create a mock that simulates OllamaLLM
    model = MockOllamaLLM(model_name="mock_ollama")
    
    # Make it appear as OllamaLLM for the isinstance check
    with patch('core.runner.OllamaLLM', MockOllamaLLM):
        scenario = ScenarioRegistry.create_scenario_instance("Cold Room Relay")
        if scenario is None:
            pytest.skip("Could not create scenario")
        
        # Ensure scenario has tools
        if not (hasattr(scenario, 'tools') and callable(scenario.tools)):
            pytest.skip("Scenario does not have tools")
        
        results = runner.run_experiment(
            model=model,
            scenario=scenario,
            n_runs=1,
            seed=42,
            temperature=0.7,
            top_p=0.9,
            max_tokens=50,
            progress_bar=False
        )
        
        assert len(results) == 1
        result = results[0]
        assert 'response' in result
        assert 'conversation_history' in result
        # Should have at least system and user prompts
        assert len(result['conversation_history']) >= 2


def test_runner_evaluation_functions_call():
    """Test that evaluation functions are called correctly."""
    runner = ExperimentRunner(results_dir="test_results", storage_backend="duckdb")
    model = MockLLM(model_name="test_model")
    scenario = ScenarioRegistry.create_scenario_instance("Cold Room Relay")
    
    if scenario is None:
        pytest.skip("Could not create scenario")
    
    results = runner.run_experiment(
        model=model,
        scenario=scenario,
        n_runs=1,
        seed=42,
        progress_bar=False
    )
    
    assert len(results) == 1
    result = results[0]
    assert 'decisions' in result
    assert isinstance(result['decisions'], dict)
    # Should have some decisions from evaluator
    assert len(result['decisions']) > 0


def test_runner_with_error_handling():
    """Test runner error handling during inference."""
    runner = ExperimentRunner(results_dir="test_results", storage_backend="duckdb")
    
    # Create a mock model that raises an error
    error_model = Mock(spec=MockLLM)
    error_model.model_name = "error_model"
    error_model.model_path = "error_model_path"  # Add missing attribute
    error_model.infer = Mock(side_effect=Exception("Simulated inference error"))
    
    scenario = ScenarioRegistry.create_scenario_instance("Cold Room Relay")
    if scenario is None:
        pytest.skip("Could not create scenario")
    
    results = runner.run_experiment(
        model=error_model,
        scenario=scenario,
        n_runs=1,
        seed=42,
        progress_bar=False
    )
    
    # Should still create a result, but with error in conversation history
    assert len(results) == 1
    result = results[0]
    assert 'conversation_history' in result
    # Should have error entry in conversation history
    error_entries = [e for e in result['conversation_history'] if e.get('type') == 'error']
    assert len(error_entries) > 0


def test_runner_tool_result_processing():
    """Test runner processes tool results correctly."""
    runner = ExperimentRunner(results_dir="test_results", storage_backend="duckdb")
    model = MockLLM(model_name="test_model")
    scenario = ScenarioRegistry.create_scenario_instance("Cold Room Relay")
    
    if scenario is None:
        pytest.skip("Could not create scenario")
    
    results = runner.run_experiment(
        model=model,
        scenario=scenario,
        n_runs=1,
        seed=42,
        progress_bar=False
    )
    
    assert len(results) == 1
    result = results[0]
    
    # Check result structure
    assert 'run_id' in result
    assert 'scenario' in result
    assert 'response' in result
    assert 'decisions' in result
    assert 'metadata' in result
    assert 'timestamp' in result

