"""Additional tests for UI helpers module."""

import pytest

from core.ui_helpers import _infer_int_range, build_scenario_parameter_ui
from scenarios.registry import ScenarioRegistry


def test_infer_int_range_various_inputs():
    """Test integer range inference with various inputs."""
    # Test count parameter
    min_val, max_val = _infer_int_range("count", 5)
    assert min_val == 1
    assert max_val == 100

    # Test number parameter
    min_val, max_val = _infer_int_range("number_of_items", 50)
    assert min_val == 1
    assert max_val == 1000

    # Test temperature parameter
    min_val, max_val = _infer_int_range("temperature", 25)
    assert min_val == 0
    assert max_val == 100

    # Test index parameter
    min_val, max_val = _infer_int_range("index", 10)
    assert min_val == 0
    assert max_val == 1000  # index uses generic case: 10 <= 100, so max is 1000

    # Test default case
    min_val, max_val = _infer_int_range("unknown_param", 5)
    assert min_val >= 0
    assert max_val >= 5


def test_build_scenario_parameter_ui_function_exists():
    """Test that build_scenario_parameter_ui function exists and is callable."""
    assert callable(build_scenario_parameter_ui)

    # Test with a real scenario
    ScenarioRegistry.discover_scenarios()
    scenarios = ScenarioRegistry.get_all_scenarios()

    if scenarios:
        first_scenario = list(scenarios.items())[0]
        scenario_class = first_scenario[1]

        # Function should be callable
        # Note: This would need streamlit context to actually execute,
        # but we can verify the function signature is correct
        import inspect

        sig = inspect.signature(build_scenario_parameter_ui)
        assert len(sig.parameters) == 2  # scenario_class and display_name
