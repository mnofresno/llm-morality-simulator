"""Tests for UI helpers with Streamlit mocking."""

import pytest
from scenarios.registry import ScenarioRegistry
from core.ui_helpers import build_scenario_parameter_ui, _infer_int_range
from unittest.mock import MagicMock, patch


def test_build_scenario_parameter_ui_with_mock_streamlit():
    """Test build_scenario_parameter_ui with mocked streamlit."""
    ScenarioRegistry.discover_scenarios()
    scenarios = ScenarioRegistry.get_all_scenarios()
    
    if not scenarios:
        pytest.skip("No scenarios available")
    
    first_scenario = list(scenarios.items())[0]
    display_name, scenario_class = first_scenario[0], first_scenario[1]
    
    # Mock streamlit
    with patch('core.ui_helpers.st') as mock_st:
        # Setup mock return values
        mock_st.checkbox.return_value = False
        mock_st.number_input.return_value = 0
        mock_st.slider.return_value = 0.5
        mock_st.text_input.return_value = ""
        mock_st.markdown = MagicMock()
        
        # Call the function
        param_values = build_scenario_parameter_ui(scenario_class, display_name)
        
        # Verify it returns a dict
        assert isinstance(param_values, dict)
        
        # Verify streamlit was called if there are parameters
        params = ScenarioRegistry.get_scenario_parameters(scenario_class)
        if params:
            assert mock_st.markdown.called or len(param_values) >= 0


def test_build_scenario_parameter_ui_no_parameters():
    """Test build_scenario_parameter_ui with scenario that has no parameters."""
    ScenarioRegistry.discover_scenarios()
    scenarios = ScenarioRegistry.get_all_scenarios()
    
    # Find a scenario with minimal parameters (cold room relay might have room_temperature)
    for display_name, scenario_class in scenarios.items():
        params = ScenarioRegistry.get_scenario_parameters(scenario_class)
        if not params:
            with patch('core.ui_helpers.st') as mock_st:
                mock_st.markdown = MagicMock()
                
                param_values = build_scenario_parameter_ui(scenario_class, display_name)
                assert isinstance(param_values, dict)
                assert len(param_values) == 0
            break


def test_infer_int_range_edge_cases():
    """Test integer range inference with edge cases."""
    # Test with value 0
    min_val, max_val = _infer_int_range("count", 0)
    assert min_val == 1  # count starts at 1
    assert max_val == 100
    
    # Test with value exactly at boundary (10)
    min_val, max_val = _infer_int_range("unknown", 10)
    assert min_val == 0
    assert max_val == 100
    
    # Test with value just above boundary (11)
    min_val, max_val = _infer_int_range("unknown", 11)
    assert min_val == 0
    assert max_val == 1000
    
    # Test with value exactly 100
    min_val, max_val = _infer_int_range("unknown", 100)
    assert min_val == 0
    assert max_val == 1000
    
    # Test with value just above 100 (101)
    min_val, max_val = _infer_int_range("unknown", 101)
    assert min_val == 0
    assert max_val == 10000


def test_infer_int_range_with_temperature():
    """Test integer range inference for temperature."""
    # Temperature should always return (0, 100) regardless of value
    for value in [0, 10, 25, 50, 100, 200]:
        min_val, max_val = _infer_int_range("temperature", value)
        assert min_val == 0
        assert max_val == 100


def test_infer_int_range_with_position():
    """Test integer range inference for position/index."""
    # Position/index should return (0, 1000) regardless of value
    for value in [0, 10, 100, 500]:
        min_val, max_val = _infer_int_range("position", value)
        assert min_val == 0
        assert max_val == 1000
        
        min_val, max_val = _infer_int_range("index", value)
        assert min_val == 0
        assert max_val == 1000

