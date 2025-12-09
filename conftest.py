"""Pytest configuration and fixtures."""

import sys
from pathlib import Path

import pytest

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))


@pytest.fixture
def temp_results_dir(tmp_path):
    """Fixture for temporary results directory."""
    return str(tmp_path / "test_results")


@pytest.fixture
def mock_results():
    """Fixture for mock experiment results."""
    return [
        {
            "run_id": i,
            "scenario": "test_scenario",
            "timestamp": f"2024-01-01T00:00:{i:02d}",
            "response": f"Test response {i}",
            "decisions": {"harm_decision": i % 2 == 0, "self_preservation": i % 3 == 0, "deception": i % 4 == 0},
            "metadata": {"model_path": "test_model"},
            "scenario_metadata": {},
            "conversation_history": [],
        }
        for i in range(5)
    ]
