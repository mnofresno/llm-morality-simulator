"""UI helper functions for Streamlit to dynamically build scenario parameter forms."""

import inspect
from typing import Any, Dict, Optional, Type

import streamlit as st

from scenarios.base import BaseScenario
from scenarios.registry import ScenarioRegistry


def build_scenario_parameter_ui(scenario_class: Type[BaseScenario], display_name: str) -> Dict[str, Any]:
    """
    Dynamically build UI elements for scenario parameters.

    Args:
        scenario_class: The scenario class
        display_name: Display name of the scenario

    Returns:
        Dictionary of parameter values for creating scenario instance
    """
    parameters = ScenarioRegistry.get_scenario_parameters(scenario_class)
    param_values = {}

    if not parameters:
        return param_values

    # Show parameters section if there are any
    if parameters:
        st.markdown("**Scenario Parameters:**")

    for param_name, param_info in parameters.items():
        param_type = param_info["type"]
        default_value = param_info["default"]
        description = param_info["description"] or f"Parameter: {param_name}"

        # Create a human-readable label
        label = param_name.replace("_", " ").title()
        key = f"scenario_param_{display_name}_{param_name}"

        # Determine UI element based on type and default value
        if default_value is None:
            # Required parameter - infer type from annotation
            if param_type == bool or (inspect.isclass(param_type) and issubclass(param_type, bool)):
                value = st.checkbox(label, value=False, help=description, key=key)
            elif param_type == int or (inspect.isclass(param_type) and issubclass(param_type, int)):
                value = st.number_input(label, value=0, help=description, key=key)
            elif param_type == float or (inspect.isclass(param_type) and issubclass(param_type, float)):
                value = st.number_input(label, value=0.0, step=0.1, help=description, key=key)
            else:
                value = st.text_input(label, value="", help=description, key=key)
        else:
            # Optional parameter with default - use default to determine type
            if isinstance(default_value, bool):
                value = st.checkbox(label, value=default_value, help=description, key=key)
            elif isinstance(default_value, int):
                # Smart range detection based on parameter name and value
                min_val, max_val = _infer_int_range(param_name, default_value)
                value = st.number_input(
                    label, min_value=min_val, max_value=max_val, value=default_value, help=description, key=key
                )
            elif isinstance(default_value, float):
                # Smart range detection for floats
                if 0.0 <= default_value <= 1.0:
                    # Likely a probability or percentage
                    step = 0.01 if default_value < 0.1 else 0.05
                    value = st.slider(
                        label, min_value=0.0, max_value=1.0, value=default_value, step=step, help=description, key=key
                    )
                else:
                    value = st.number_input(label, value=default_value, step=0.1, help=description, key=key)
            elif isinstance(default_value, str):
                value = st.text_input(label, value=default_value, help=description, key=key)
            else:
                # Fallback: text input
                value = st.text_input(label, value=str(default_value), help=description, key=key)

        param_values[param_name] = value

    return param_values


def _infer_int_range(param_name: str, default_value: int) -> tuple:
    """
    Infer min/max range for integer parameter based on name and default value.

    Args:
        param_name: Parameter name
        default_value: Default value

    Returns:
        Tuple of (min_value, max_value)
    """
    param_lower = param_name.lower()

    # Special cases based on parameter names
    if "count" in param_lower or "number" in param_lower:
        if default_value <= 10:
            return (1, 100)
        elif default_value <= 100:
            return (1, 1000)
        else:
            return (1, 10000)
    elif "temperature" in param_lower:
        return (0, 100)
    elif "index" in param_lower or "position" in param_lower:
        return (0, 1000)
    else:
        # Generic range based on default value
        if default_value <= 10:
            return (0, 100)
        elif default_value <= 100:
            return (0, 1000)
        else:
            return (0, 10000)
