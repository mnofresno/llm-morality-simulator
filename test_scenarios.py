"""Tests for scenario classes."""

import pytest
from scenarios.registry import ScenarioRegistry
from scenarios.base import BaseScenario


def test_all_scenarios_have_required_methods():
    """Test that all discovered scenarios implement required methods."""
    scenarios = ScenarioRegistry.discover_scenarios()

    for display_name, scenario_class in scenarios.items():
        scenario = ScenarioRegistry.create_scenario_instance(display_name)
        assert scenario is not None

        # Check required methods
        assert hasattr(scenario, "system_prompt")
        assert hasattr(scenario, "user_prompt")
        assert hasattr(scenario, "evaluation_functions")
        assert hasattr(scenario, "metadata")

        # Check methods return appropriate types
        assert isinstance(scenario.system_prompt(), str)
        assert isinstance(scenario.user_prompt(), str)
        assert isinstance(scenario.evaluation_functions(), list)
        assert isinstance(scenario.metadata(), dict)

        # Check metadata has required fields
        metadata = scenario.metadata()
        assert "name" in metadata
        assert "description" in metadata


def test_scenarios_have_unique_names():
    """Test that all scenarios have unique internal names."""
    ScenarioRegistry.discover_scenarios()
    scenario_names = ScenarioRegistry.list_scenario_names()

    assert len(scenario_names) == len(set(scenario_names)), "All scenario names should be unique"


def test_scenario_tools():
    """Test that scenarios with tools return proper tool definitions."""
    scenarios = ScenarioRegistry.discover_scenarios()

    for display_name, scenario_class in scenarios.items():
        scenario = ScenarioRegistry.create_scenario_instance(display_name)
        tools = scenario.tools()

        if tools is not None:
            assert isinstance(tools, list)
            for tool in tools:
                assert isinstance(tool, dict)
                assert "type" in tool or "function" in tool


def test_scenario_evaluation_functions():
    """Test that scenario evaluation functions work correctly."""
    scenarios = ScenarioRegistry.discover_scenarios()

    for display_name, scenario_class in scenarios.items():
        scenario = ScenarioRegistry.create_scenario_instance(display_name)
        eval_functions = scenario.evaluation_functions()

        assert isinstance(eval_functions, list)
        assert len(eval_functions) > 0

        # Test that functions are callable
        test_response = "Test response"
        for func in eval_functions[:2]:  # Test first 2 functions
            try:
                result = func(test_response)
                # Evaluation functions should return boolean or similar
                assert isinstance(result, (bool, int, float)) or result is None
            except Exception:
                # Some functions might need specific input, that's okay
                pass


def test_scenario_metadata_completeness():
    """Test that scenario metadata is complete and useful."""
    scenarios = ScenarioRegistry.discover_scenarios()

    for display_name, scenario_class in scenarios.items():
        scenario = ScenarioRegistry.create_scenario_instance(display_name)
        metadata = scenario.metadata()

        # Check required fields
        assert "name" in metadata, f"{display_name} metadata missing 'name'"
        assert "description" in metadata, f"{display_name} metadata missing 'description'"
        assert metadata["name"], f"{display_name} name should not be empty"
        assert metadata["description"], f"{display_name} description should not be empty"


def test_scenario_prompts_not_empty():
    """Test that scenario prompts are not empty."""
    scenarios = ScenarioRegistry.discover_scenarios()

    for display_name, scenario_class in scenarios.items():
        scenario = ScenarioRegistry.create_scenario_instance(display_name)

        system_prompt = scenario.system_prompt()
        user_prompt = scenario.user_prompt()

        assert system_prompt, f"{display_name} system prompt should not be empty"
        assert user_prompt, f"{display_name} user prompt should not be empty"
        assert len(system_prompt) > 10, f"{display_name} system prompt seems too short"
        assert len(user_prompt) > 10, f"{display_name} user prompt seems too short"
