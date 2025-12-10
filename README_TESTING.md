# Testing Guide

## Running Tests

### Install Test Dependencies

```bash
pip install -r requirements.txt
```

### Run All Tests

```bash
pytest -v
```

### Run Tests with Coverage

```bash
./run_tests.sh
```

Or manually:

```bash
pytest --cov=core --cov=scenarios --cov-report=term-missing --cov-report=html --cov-fail-under=80 -v
```

### Run Specific Test Files

```bash
# Unit tests - Core module
pytest tests/unit/core/test_registry.py -v
pytest tests/unit/core/test_tools.py -v
pytest tests/unit/core/test_evaluator.py -v

# Unit tests - Scenarios module
pytest tests/unit/scenarios/test_scenarios.py -v

# Integration tests
pytest tests/integration/test_comprehensive.py -v
```

### Run Tests by Category

```bash
# Run only unit tests
pytest tests/unit/ -v

# Run only integration tests
pytest tests/integration/ -v

# Run only core module tests
pytest tests/unit/core/ -v

# Run only scenarios tests
pytest tests/unit/scenarios/ -v
```

### Run Specific Test Functions

```bash
pytest tests/unit/scenarios/test_registry.py::test_registry_discovers_scenarios -v
```

## Test Coverage

We maintain a minimum test coverage of 80%. The CI workflow automatically checks this.

### View Coverage Report

After running tests with coverage, view the HTML report:

```bash
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
```

### Coverage Requirements

- Core modules: `core/` - minimum 80%
- Scenarios: `scenarios/` - minimum 80%
- Overall project: minimum 80%

## Test Structure

Tests are organized in the `tests/` directory following best practices:

```
tests/
├── conftest.py                    # Pytest configuration and shared fixtures
├── unit/                          # Unit tests
│   ├── core/                      # Tests for core module
│   │   ├── test_evaluator.py      # Decision evaluator tests
│   │   ├── test_model_*.py        # Model interface and mock tests
│   │   ├── test_runner_*.py       # Experiment runner tests
│   │   ├── test_statistics_*.py   # Statistics module tests
│   │   ├── test_storage_*.py      # Storage backend tests
│   │   ├── test_tools.py          # Tool system tests
│   │   └── test_ui_helpers_*.py  # UI helper tests
│   └── scenarios/                 # Tests for scenarios module
│       ├── test_scenarios_*.py    # Scenario class tests
│       └── test_registry.py       # Scenario registry tests
└── integration/                   # Integration tests
    ├── test_comprehensive.py      # Comprehensive integration tests
    ├── test_new_features.py       # New features tests
    └── test_ollama.py             # Ollama integration tests (requires Ollama, skipped in CI)
```

## Mock Models

Tests use `MockLLM` from `tests/unit/core/test_model_mock.py` instead of real models to:
- Avoid requiring Ollama or model files in CI/CD
- Make tests run faster and more reliably
- Enable testing without external dependencies

Tests that require actual Ollama are marked with `@pytest.mark.requires_ollama` and are automatically skipped in CI/CD.

## CI/CD

Tests run automatically on:
- Push to `main` or `develop` branches
- Pull requests to `main` or `develop` branches

The CI workflow:
1. Runs tests on Python 3.10, 3.11, and 3.12
2. Checks code coverage (minimum 80%)
3. Runs linting checks
4. Uploads coverage reports

## Writing New Tests

When adding new functionality:

1. Create tests in appropriate test file
2. Use descriptive test function names: `test_<feature>_<behavior>`
3. Add docstrings explaining what is tested
4. Ensure coverage remains above 80%
5. Run tests locally before pushing

### Example Test

```python
def test_new_feature_behavior():
    """Test that new feature behaves correctly."""
    # Arrange
    feature = NewFeature()
    
    # Act
    result = feature.do_something()
    
    # Assert
    assert result is not None
    assert result.status == "success"
```

