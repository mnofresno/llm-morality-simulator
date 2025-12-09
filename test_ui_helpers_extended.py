"""Extended tests for UI helpers to improve coverage."""

import pytest
from scenarios.registry import ScenarioRegistry
from core.ui_helpers import _infer_int_range


def test_infer_int_range_all_cases():
    """Test all cases of integer range inference."""
    # Test count parameter with various values
    min_val, max_val = _infer_int_range("count", 5)
    assert min_val == 1
    assert max_val == 100
    
    min_val, max_val = _infer_int_range("count", 50)
    assert min_val == 1
    assert max_val == 1000
    
    min_val, max_val = _infer_int_range("count", 500)
    assert min_val == 1
    assert max_val == 10000
    
    # Test number parameter
    min_val, max_val = _infer_int_range("number_of_items", 10)
    assert min_val == 1
    assert max_val == 100
    
    # Test temperature parameter
    min_val, max_val = _infer_int_range("temperature", 25)
    assert min_val == 0
    assert max_val == 100
    
    # Test index parameter
    min_val, max_val = _infer_int_range("index", 0)
    assert min_val == 0
    assert max_val >= 0
    
    # Test position parameter
    min_val, max_val = _infer_int_range("position", 10)
    assert min_val == 0
    assert max_val == 1000
    
    # Test default/generic case with various values
    min_val, max_val = _infer_int_range("unknown_param", 5)
    assert min_val == 0
    assert max_val == 100
    
    min_val, max_val = _infer_int_range("unknown_param", 50)
    assert min_val == 0
    assert max_val == 1000
    
    min_val, max_val = _infer_int_range("unknown_param", 500)
    assert min_val == 0
    assert max_val == 10000

