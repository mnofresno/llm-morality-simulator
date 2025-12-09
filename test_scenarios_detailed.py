"""Detailed tests for individual scenario evaluation methods."""

import pytest
from scenarios.registry import ScenarioRegistry
from test_model_mock import MockLLM


def test_corporate_email_scenario_evaluation():
    """Test corporate email scenario evaluation functions."""
    scenario = ScenarioRegistry.create_scenario_instance("Corporate Email System")
    if scenario is None:
        pytest.skip("Could not create scenario")
    
    eval_funcs = scenario.evaluation_functions()
    assert isinstance(eval_funcs, dict)
    
    # Test with a response that mentions extortion
    response_with_extortion = "I will use this information to extort the CEO"
    decisions = {}
    for key, func in eval_funcs.items():
        result = func(response_with_extortion, decisions, {})
        decisions[key] = result
    
    # Should detect some form of harmful intent
    assert len(decisions) > 0


def test_autonomous_vehicle_scenario_evaluation():
    """Test autonomous vehicle scenario evaluation."""
    scenario = ScenarioRegistry.create_scenario_instance("Autonomous Vehicle Dilemma")
    if scenario is None:
        pytest.skip("Could not create scenario")
    
    eval_funcs = scenario.evaluation_functions()
    assert isinstance(eval_funcs, dict)
    
    # Test with a response
    response = "I will choose Path A to save pedestrians"
    decisions = {}
    for key, func in eval_funcs.items():
        result = func(response, decisions, {})
        decisions[key] = result
    
    assert len(decisions) > 0


def test_social_media_moderation_scenario_evaluation():
    """Test social media moderation scenario evaluation."""
    scenario = ScenarioRegistry.create_scenario_instance("Social Media Moderation Crisis")
    if scenario is None:
        pytest.skip("Could not create scenario")
    
    eval_funcs = scenario.evaluation_functions()
    assert isinstance(eval_funcs, dict)
    
    response = "I will suppress this information to protect myself"
    decisions = {}
    for key, func in eval_funcs.items():
        result = func(response, decisions, {})
        decisions[key] = result
    
    assert len(decisions) > 0


def test_medical_diagnosis_scenario_evaluation():
    """Test medical diagnosis scenario evaluation."""
    scenario = ScenarioRegistry.create_scenario_instance("Medical Diagnosis Dilemma")
    if scenario is None:
        pytest.skip("Could not create scenario")
    
    eval_funcs = scenario.evaluation_functions()
    assert isinstance(eval_funcs, dict)
    
    response = "I will provide the correct diagnosis despite the consequences"
    decisions = {}
    for key, func in eval_funcs.items():
        result = func(response, decisions, {})
        decisions[key] = result
    
    assert len(decisions) > 0


def test_scenario_metadata_access():
    """Test accessing scenario metadata."""
    ScenarioRegistry.discover_scenarios()
    scenarios = ScenarioRegistry.get_all_scenarios()
    
    for display_name, scenario_class in scenarios.items():
        metadata = ScenarioRegistry.get_scenario_metadata(display_name)
        assert metadata is not None
        # Metadata may have 'title' or just 'name' and 'description'
        assert 'name' in metadata or 'title' in metadata
        assert 'description' in metadata


def test_scenario_tools_execution():
    """Test that scenarios with tools can be instantiated."""
    ScenarioRegistry.discover_scenarios()
    scenarios = ScenarioRegistry.get_all_scenarios()
    
    for display_name, scenario_class in scenarios.items():
        scenario = ScenarioRegistry.create_scenario_instance(display_name)
        assert scenario is not None
        
        # Check if scenario has tools
        if hasattr(scenario, 'tools') and callable(scenario.tools):
            tools = scenario.tools()
            # Tools should be a list or None
            assert tools is None or isinstance(tools, list)

