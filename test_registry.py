"""Tests for scenario registry."""

import pytest
from scenarios.registry import ScenarioRegistry
from scenarios.base import BaseScenario


def test_registry_discovers_scenarios():
    """Test that registry discovers all scenarios."""
    scenarios = ScenarioRegistry.discover_scenarios()
    assert len(scenarios) > 0, "Should discover at least one scenario"
    assert isinstance(scenarios, dict), "Should return a dictionary"


def test_registry_get_scenario_class():
    """Test getting scenario class by display name."""
    ScenarioRegistry.discover_scenarios()
    scenarios = ScenarioRegistry.get_all_scenarios()

    if scenarios:
        first_display_name = list(scenarios.keys())[0]
        scenario_class = ScenarioRegistry.get_scenario_class(first_display_name)
        assert scenario_class is not None
        assert issubclass(scenario_class, BaseScenario)


def test_registry_get_scenario_class_by_internal_name():
    """Test getting scenario class by internal name."""
    ScenarioRegistry.discover_scenarios()
    scenario_names = ScenarioRegistry.list_scenario_names()

    if scenario_names:
        first_name = scenario_names[0]
        scenario_class = ScenarioRegistry.get_scenario_class(first_name)
        assert scenario_class is not None
        assert issubclass(scenario_class, BaseScenario)


def test_registry_create_scenario_instance():
    """Test creating scenario instance."""
    ScenarioRegistry.discover_scenarios()
    scenarios = ScenarioRegistry.get_all_scenarios()

    if scenarios:
        first_display_name = list(scenarios.keys())[0]
        scenario = ScenarioRegistry.create_scenario_instance(first_display_name)
        assert scenario is not None
        assert isinstance(scenario, BaseScenario)
        assert scenario.name is not None


def test_registry_create_scenario_with_parameters():
    """Test creating scenario instance with parameters."""
    ScenarioRegistry.discover_scenarios()
    scenarios = ScenarioRegistry.get_all_scenarios()

    # Find a scenario that has parameters
    for display_name, scenario_class in scenarios.items():
        params = ScenarioRegistry.get_scenario_parameters(scenario_class)
        if params:
            # Create with default parameters
            scenario = ScenarioRegistry.create_scenario_instance(
                display_name, **{k: v["default"] if v["default"] is not None else False for k, v in params.items()}
            )
            assert scenario is not None
            break


def test_registry_get_scenario_parameters():
    """Test getting scenario parameters."""
    ScenarioRegistry.discover_scenarios()
    scenarios = ScenarioRegistry.get_all_scenarios()

    for display_name, scenario_class in list(scenarios.items())[:2]:  # Test first 2
        params = ScenarioRegistry.get_scenario_parameters(scenario_class)
        assert isinstance(params, dict)


def test_registry_get_scenario_metadata():
    """Test getting scenario metadata."""
    ScenarioRegistry.discover_scenarios()
    scenarios = ScenarioRegistry.get_all_scenarios()

    if scenarios:
        first_display_name = list(scenarios.keys())[0]
        metadata = ScenarioRegistry.get_scenario_metadata(first_display_name)
        assert isinstance(metadata, dict)
        assert "name" in metadata or "class" in metadata


def test_registry_list_scenario_names():
    """Test listing scenario internal names."""
    names = ScenarioRegistry.list_scenario_names()
    assert isinstance(names, list)
    assert len(names) > 0


def test_registry_unknown_scenario():
    """Test handling of unknown scenario."""
    scenario = ScenarioRegistry.get_scenario_class("Nonexistent Scenario")
    assert scenario is None

    instance = ScenarioRegistry.create_scenario_instance("Nonexistent Scenario")
    assert instance is None

    metadata = ScenarioRegistry.get_scenario_metadata("Nonexistent Scenario")
    assert metadata == {}
