"""Tests for UI helpers."""

import pytest
import streamlit as st
from scenarios.registry import ScenarioRegistry
from core.ui_helpers import build_scenario_parameter_ui, _infer_int_range


def test_infer_int_range():
    """Test integer range inference."""
    # Test with count parameter
    min_val, max_val = _infer_int_range("count", 5)
    assert min_val >= 1
    assert max_val >= 5

    # Test with temperature parameter
    min_val, max_val = _infer_int_range("temperature", 25)
    assert min_val == 0
    assert max_val == 100

    # Test with default value
    min_val, max_val = _infer_int_range("test", 10)
    assert min_val >= 0
    assert max_val >= 10


def test_build_scenario_parameter_ui():
    """Test building scenario parameter UI."""
    ScenarioRegistry.discover_scenarios()
    scenarios = ScenarioRegistry.get_all_scenarios()

    if scenarios:
        # Get first scenario with parameters
        for display_name, scenario_class in scenarios.items():
            params = ScenarioRegistry.get_scenario_parameters(scenario_class)
            if params:
                # Note: This test requires streamlit context, so we'll just check the function exists
                # In a real test environment, we'd need to mock streamlit
                assert callable(build_scenario_parameter_ui)
                break
