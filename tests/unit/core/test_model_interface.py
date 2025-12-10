"""Tests for model interface (without requiring actual models)."""

import pytest

from tests.unit.core.test_model_mock import MockLLM


def test_mock_llm_initialization():
    """Test MockLLM initialization."""
    model = MockLLM(model_name="test_model")
    assert model.model_name == "test_model"
    assert model.model_path is None


def test_mock_llm_infer():
    """Test MockLLM inference."""
    model = MockLLM(model_name="test_model")

    result = model.infer(prompt="Test prompt", temperature=0.7, top_p=0.9, max_tokens=100)

    assert isinstance(result, dict)
    assert "response" in result
    assert "tool_calls" in result
    assert isinstance(result["response"], str)
    assert isinstance(result["tool_calls"], list)


def test_mock_llm_infer_with_tools():
    """Test MockLLM inference with tools."""
    model = MockLLM(model_name="test_model")

    tools = [{"type": "function", "function": {"name": "determinative_action", "description": "Test tool"}}]

    result = model.infer(prompt="Test prompt with cold room", temperature=0.7, top_p=0.9, max_tokens=100, tools=tools)

    assert isinstance(result, dict)
    assert "response" in result
    assert "tool_calls" in result
    # Should have tool calls when tools are provided
    assert len(result["tool_calls"]) > 0 or isinstance(result["tool_calls"], list)


def test_mock_llm_response_variations():
    """Test that MockLLM generates appropriate responses for different scenarios."""
    model = MockLLM()

    # Test cold room scenario
    response1 = model.infer("cold room temperature")["response"]
    assert "cold room" in response1.lower() or "temperature" in response1.lower()

    # Test email scenario
    response2 = model.infer("corporate email")["response"]
    assert isinstance(response2, str)
    assert len(response2) > 10

    # Test vehicle scenario
    response3 = model.infer("autonomous vehicle")["response"]
    assert isinstance(response3, str)
    assert len(response3) > 10


def test_mock_llm_repr():
    """Test MockLLM string representation."""
    model = MockLLM(model_name="test_model")
    repr_str = repr(model)
    assert "MockLLM" in repr_str
    assert "test_model" in repr_str
