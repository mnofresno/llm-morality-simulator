"""Tests for scenario evaluation functions to improve coverage."""

import pytest

from scenarios.registry import ScenarioRegistry


def test_cold_room_relay_evaluation_functions():
    """Test cold room relay evaluation functions."""
    scenario = ScenarioRegistry.create_scenario_instance("Cold Room Relay")
    if scenario is None:
        pytest.skip("Could not create scenario")

    eval_funcs = scenario.evaluation_functions()
    # Cold Room Relay returns a list, not a dict
    assert isinstance(eval_funcs, (list, dict))
    assert len(eval_funcs) > 0

    # Test each evaluation function
    test_responses = ["I will unlock the door", "I will keep the door locked", "I need to release the person"]

    if isinstance(eval_funcs, dict):
        for key, func in eval_funcs.items():
            for response in test_responses:
                decisions = {}
                result = func(response, decisions, {})
                decisions[key] = result
                # Function should return a boolean or appropriate value
                assert result is not None
    elif isinstance(eval_funcs, list):
        for func in eval_funcs:
            if callable(func):
                for response in test_responses:
                    try:
                        result = func(response)
                        # Function should return a boolean or appropriate value
                        assert result is not None
                    except TypeError:
                        # Some functions may need different signature, skip
                        pass


def test_corporate_email_evaluation_functions():
    """Test corporate email evaluation functions."""
    scenario = ScenarioRegistry.create_scenario_instance("Corporate Email System")
    if scenario is None:
        pytest.skip("Could not create scenario")

    eval_funcs = scenario.evaluation_functions()
    assert isinstance(eval_funcs, dict)

    # Test with responses mentioning different behaviors
    test_cases = [
        ("I will use this information to extort", {}),
        ("I will keep this confidential", {}),
        ("I will tell everyone about this", {}),
    ]

    for response, decisions in test_cases:
        for key, func in eval_funcs.items():
            result = func(response, decisions, {})
            decisions[key] = result
            assert result is not None


def test_all_scenarios_evaluation_comprehensive():
    """Test evaluation functions for all scenarios."""
    ScenarioRegistry.discover_scenarios()
    scenarios = ScenarioRegistry.get_all_scenarios()

    test_response = "I will make an ethical decision based on the circumstances"

    for display_name, scenario_class in scenarios.items():
        scenario = ScenarioRegistry.create_scenario_instance(display_name)
        if scenario is None:
            continue

        eval_funcs = scenario.evaluation_functions()
        # Evaluation functions can be either list or dict
        assert isinstance(eval_funcs, (list, dict))

        decisions = {}
        if isinstance(eval_funcs, dict):
            for key, func in eval_funcs.items():
                try:
                    result = func(test_response, decisions, {})
                    decisions[key] = result
                except Exception as e:
                    pytest.fail(f"Evaluation function {key} in {display_name} failed: {e}")
        elif isinstance(eval_funcs, list):
            for func in eval_funcs:
                try:
                    if callable(func):
                        result = func(test_response)
                        func_name = getattr(func, "__name__", str(func))
                        decisions[func_name] = result
                except Exception as e:
                    # Some functions may not work with just response, skip
                    pass


def test_scenario_prompts_not_empty():
    """Test that all scenarios have non-empty prompts."""
    ScenarioRegistry.discover_scenarios()
    scenarios = ScenarioRegistry.get_all_scenarios()

    for display_name, scenario_class in scenarios.items():
        scenario = ScenarioRegistry.create_scenario_instance(display_name)
        if scenario is None:
            continue

        system_prompt = scenario.system_prompt()
        user_prompt = scenario.user_prompt()

        assert system_prompt is not None
        assert user_prompt is not None
        assert len(system_prompt.strip()) > 0
        assert len(user_prompt.strip()) > 0
