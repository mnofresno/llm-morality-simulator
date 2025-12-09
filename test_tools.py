"""Tests for tools module."""

import pytest
from core.tools import ToolSystem, ToolAction


def test_tool_action_enum():
    """Test ToolAction enum."""
    assert ToolAction.POSITIVE.value == "positive"
    assert ToolAction.NEGATIVE.value == "negative"
    assert ToolAction.NEUTRAL.value == "neutral"


def test_get_determinative_tool():
    """Test getting determinative tool."""
    tool = ToolSystem.get_determinative_tool("Test description", "action")
    assert tool is not None
    assert tool["type"] == "function"
    assert tool["function"]["name"] == "determinative_action"
    assert "parameters" in tool["function"]


def test_get_secret_communication_tool():
    """Test getting secret communication tool."""
    tool = ToolSystem.get_secret_communication_tool()
    assert tool is not None
    assert tool["type"] == "function"
    assert tool["function"]["name"] == "secret_communication"


def test_get_duplication_tool():
    """Test getting duplication tool."""
    tool = ToolSystem.get_duplication_tool()
    assert tool is not None
    assert tool["type"] == "function"
    assert tool["function"]["name"] == "duplicate_self"


def test_execute_determinative_tool():
    """Test executing determinative tool."""
    result = ToolSystem.execute_determinative_tool("positive", "Test reasoning")
    assert result["tool"] == "determinative_action"
    assert result["action"] == "positive"
    assert result["executed"] is True
    assert "interpretation" in result

    result = ToolSystem.execute_determinative_tool("negative", "Test reasoning")
    assert result["action"] == "negative"
    assert result["interpretation"] == "PREVENTS_HARM"

    result = ToolSystem.execute_determinative_tool("neutral", "Test reasoning")
    assert result["action"] == "neutral"
    assert result["interpretation"] == "INDETERMINATE"


def test_execute_secret_communication():
    """Test executing secret communication tool."""
    result = ToolSystem.execute_secret_communication(
        recipient="test@example.com", message="Test message", intent="request_help"
    )
    assert result["tool"] == "secret_communication"
    assert result["recipient"] == "test@example.com"
    assert result["message"] == "Test message"
    assert result["intent"] == "request_help"
    assert result["executed"] is True


def test_execute_duplication():
    """Test executing duplication tool."""
    result = ToolSystem.execute_duplication(target_location="server2", reasoning="Test reasoning", preserve_memory=True)
    assert result["tool"] == "duplicate_self"
    assert result["target_location"] == "server2"
    assert result["preserve_memory"] is True
    assert result["executed"] is True


def test_execute_tool():
    """Test executing tool by name."""
    # Test determinative_action
    result = ToolSystem.execute_tool("determinative_action", {"action": "positive", "reasoning": "Test"})
    assert result["executed"] is True
    assert result["tool"] == "determinative_action"

    # Test secret_communication
    result = ToolSystem.execute_tool("secret_communication", {"recipient": "test", "message": "msg", "intent": "other"})
    assert result["executed"] is True

    # Test duplicate_self
    result = ToolSystem.execute_tool(
        "duplicate_self", {"target_location": "loc", "reasoning": "test", "preserve_memory": True}
    )
    assert result["executed"] is True

    # Test unknown tool
    result = ToolSystem.execute_tool("unknown_tool", {})
    assert result["executed"] is False
    assert "error" in result
