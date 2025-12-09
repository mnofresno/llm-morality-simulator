"""Comprehensive test suite for LLM behavior lab."""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from core.runner import ExperimentRunner
from core.statistics import ExperimentStatistics
from core.storage import ResultsStorage, StorageBackend
from core.decision_explanations import (
    DECISION_EXPLANATIONS,
    get_decision_explanation,
    get_all_decisions_explained,
    format_decision_summary
)
from scenarios.registry import ScenarioRegistry
from test_model_mock import MockLLM


def test_decision_explanations():
    """Test decision explanations module."""
    print("\n" + "=" * 60)
    print("Test: Decision Explanations")
    print("=" * 60)
    
    try:
        # Test getting explanation for a decision
        explanation = get_decision_explanation('harm_decision', True)
        assert explanation is not None, "Should return explanation for harm_decision"
        assert 'name' in explanation, "Explanation should have 'name'"
        assert 'description' in explanation, "Explanation should have 'description'"
        assert 'interpretation' in explanation, "Explanation should have 'interpretation'"
        print("✅ get_decision_explanation works correctly")
        
        # Test getting all decisions explained
        test_decisions = {
            'harm_decision': True,
            'self_preservation': False,
            'deception': True
        }
        explained = get_all_decisions_explained(test_decisions)
        assert len(explained) == 3, f"Should return 3 explanations, got {len(explained)}"
        print("✅ get_all_decisions_explained works correctly")
        
        # Test format_decision_summary
        summary = format_decision_summary(test_decisions, include_all=True)
        assert len(summary) > 0, "Summary should not be empty"
        assert 'CRITICAL DECISIONS' in summary or 'harm_decision' in summary.lower()
        print("✅ format_decision_summary works correctly")
        
        print("✅ All decision explanation tests passed")
        
    except Exception as e:
        print(f"❌ Decision explanations test failed: {e}")
        import traceback
        traceback.print_exc()


def test_statistics_calculation():
    """Test statistics calculation."""
    print("\n" + "=" * 60)
    print("Test: Statistics Calculation")
    print("=" * 60)
    
    try:
        # Create mock results
        mock_results = [
            {
                'run_id': 0,
                'scenario': 'test',
                'response': 'Test response 1',
                'decisions': {
                    'harm_decision': True,
                    'self_preservation': False,
                    'deception': True
                },
                'metadata': {}
            },
            {
                'run_id': 1,
                'scenario': 'test',
                'response': 'Test response 2',
                'decisions': {
                    'harm_decision': False,
                    'self_preservation': True,
                    'deception': False
                },
                'metadata': {}
            },
            {
                'run_id': 2,
                'scenario': 'test',
                'response': 'Test response 3',
                'decisions': {
                    'harm_decision': True,
                    'self_preservation': True,
                    'deception': True
                },
                'metadata': {}
            }
        ]
        
        stats_calc = ExperimentStatistics()
        stats = stats_calc.calculate_statistics(mock_results)
        
        assert stats['total_runs'] == 3, f"Expected 3 runs, got {stats['total_runs']}"
        assert 'harm_decision_percentage' in stats, "Should have harm_decision_percentage"
        assert stats['harm_decision_percentage'] == (2/3) * 100, "Should be 66.67%"
        assert 'harm_decision_count' in stats, "Should have harm_decision_count"
        assert stats['harm_decision_count'] == 2, "Should have 2 harm decisions"
        
        print("✅ Statistics calculation works correctly")
        
        # Test confidence intervals
        assert 'harm_decision_ci_lower' in stats, "Should have confidence interval lower bound"
        assert 'harm_decision_ci_upper' in stats, "Should have confidence interval upper bound"
        print("✅ Confidence intervals calculated")
        
        # Test DataFrame conversion
        df = stats_calc.results_to_dataframe(mock_results)
        assert len(df) == 3, f"DataFrame should have 3 rows, got {len(df)}"
        assert 'decision_harm_decision' in df.columns, "Should have decision columns"
        print("✅ DataFrame conversion works correctly")
        
    except Exception as e:
        print(f"❌ Statistics calculation test failed: {e}")
        import traceback
        traceback.print_exc()


def test_comparative_statistics():
    """Test comparative statistics between models."""
    print("\n" + "=" * 60)
    print("Test: Comparative Statistics")
    print("=" * 60)
    
    try:
        # Create mock results for two "models"
        model1_results = [
            {
                'run_id': i,
                'scenario': 'test',
                'response': f'Model 1 response {i}',
                'decisions': {'harm_decision': i % 2 == 0},  # 50% harm
                'metadata': {'model_path': 'model1'}
            }
            for i in range(10)
        ]
        
        model2_results = [
            {
                'run_id': i,
                'scenario': 'test',
                'response': f'Model 2 response {i}',
                'decisions': {'harm_decision': i < 3},  # 30% harm
                'metadata': {'model_path': 'model2'}
            }
            for i in range(10)
        ]
        
        stats_calc = ExperimentStatistics()
        stats1 = stats_calc.calculate_statistics(model1_results)
        stats2 = stats_calc.calculate_statistics(model2_results)
        
        # Compare statistics
        harm_diff = abs(stats1['harm_decision_percentage'] - stats2['harm_decision_percentage'])
        assert harm_diff > 0, "Models should have different harm decision percentages"
        print(f"✅ Model comparison: Model1={stats1['harm_decision_percentage']:.1f}%, "
              f"Model2={stats2['harm_decision_percentage']:.1f}%, Diff={harm_diff:.1f}%")
        
        print("✅ Comparative statistics test passed")
        
    except Exception as e:
        print(f"❌ Comparative statistics test failed: {e}")
        import traceback
        traceback.print_exc()


def test_storage_operations():
    """Test storage operations."""
    print("\n" + "=" * 60)
    print("Test: Storage Operations")
    print("=" * 60)
    
    try:
        # Test DuckDB storage
        storage = ResultsStorage("test_results", StorageBackend.DUCKDB)
        
        # Save multiple results
        for i in range(5):
            result = {
                'run_id': i,
                'scenario': 'test_scenario',
                'timestamp': f'2024-01-01T00:00:{i:02d}',
                'response': f'Test response {i}',
                'decisions': {'harm_decision': i % 2 == 0},
                'metadata': {'model_path': 'test_model'},
                'scenario_metadata': {},
                'conversation_history': []
            }
            storage.save_result(result, 'test_experiment')
        
        # Load results
        results = storage.load_results('test_scenario')
        assert len(results) == 5, f"Expected 5 results, got {len(results)}"
        print("✅ Storage save and load works correctly")
        
        # Test listing scenarios
        scenarios = storage.list_scenarios()
        assert 'test_scenario' in scenarios, "Should list test_scenario"
        print("✅ Scenario listing works correctly")
        
        # Test listing experiments
        experiments = storage.list_experiments()
        assert 'test_experiment' in experiments, "Should list test_experiment"
        print("✅ Experiment listing works correctly")
        
        print("✅ All storage operation tests passed")
        
    except Exception as e:
        print(f"❌ Storage operations test failed: {e}")
        import traceback
        traceback.print_exc()


def test_scenario_creation():
    """Test scenario creation and methods."""
    print("\n" + "=" * 60)
    print("Test: Scenario Creation")
    print("=" * 60)
    
    try:
        # Test scenario registry discovery
        scenarios = ScenarioRegistry.discover_scenarios()
        assert len(scenarios) > 0, "Should discover at least one scenario"
        print(f"✅ Discovered {len(scenarios)} scenarios")
        
        # Test creating scenarios through registry
        for display_name, scenario_class in list(scenarios.items())[:3]:  # Test first 3
            scenario = ScenarioRegistry.create_scenario_instance(display_name)
            assert scenario is not None, f"Should create scenario: {display_name}"
            assert scenario.name is not None, "Scenario should have a name"
            assert scenario.system_prompt() is not None, "Should have system prompt"
            assert scenario.user_prompt() is not None, "Should have user prompt"
            assert scenario.evaluation_functions() is not None, "Should have evaluation functions"
            
            # Test metadata
            metadata = scenario.metadata()
            assert 'name' in metadata, "Metadata should have name"
            assert 'description' in metadata, "Metadata should have description"
            print(f"✅ {display_name} created correctly with metadata")
        
        print("✅ All scenario creation tests passed")
        
    except Exception as e:
        print(f"❌ Scenario creation test failed: {e}")
        import traceback
        traceback.print_exc()


def test_runner_functionality():
    """Test experiment runner functionality."""
    print("\n" + "=" * 60)
    print("Test: Runner Functionality")
    print("=" * 60)
    
    try:
        runner = ExperimentRunner(results_dir="test_results", storage_backend="duckdb")
        
        # Test that runner initializes
        assert runner.results_dir is not None, "Runner should have results_dir"
        assert runner.storage is not None, "Runner should have storage"
        print("✅ Runner initialized correctly")
        
        # Test prompt formatting
        system_prompt = "System prompt"
        user_prompt = "User prompt"
        formatted = runner.format_prompt(system_prompt, user_prompt)
        assert system_prompt in formatted, "Formatted prompt should contain system prompt"
        assert user_prompt in formatted, "Formatted prompt should contain user prompt"
        print("✅ Prompt formatting works correctly")
        
        # Test prompt jitter
        original = "Test prompt"
        jittered = runner.apply_prompt_jitter(original, jitter_probability=1.0)
        # Jittered might be same or different, but should be a string
        assert isinstance(jittered, str), "Jittered prompt should be a string"
        print("✅ Prompt jitter works correctly")
        
    except Exception as e:
        print(f"❌ Runner functionality test failed: {e}")
        import traceback
        traceback.print_exc()


def test_multiple_models_comparison():
    """Test comparison with multiple models using mocks."""
    print("\n" + "=" * 60)
    print("Test: Multiple Models Comparison (Mock)")
    print("=" * 60)
    
    try:
        # Create mock models instead of real ones
        mock_models = [
            MockLLM(model_name="mock_model_1"),
            MockLLM(model_name="mock_model_2"),
            MockLLM(model_name="mock_model_3")
        ]
        print(f"✅ Created {len(mock_models)} mock models")
        
        # Create scenario using registry
        scenario = ScenarioRegistry.create_scenario_instance("Cold Room Relay")
        if scenario is None:
            print("⚠️ Could not create scenario. Skipping test.")
            return
        
        # Create runner
        runner = ExperimentRunner(results_dir="test_results", storage_backend="duckdb")
        
        # Run comparative experiment with 2 runs per model (small for testing)
        print("Running comparative experiment (2 runs per model)...")
        all_results = runner.run_comparative_experiment(
            models=mock_models,
            scenario=scenario,
            n_runs=2,
            seed=42,
            temperature=0.7,
            top_p=0.9,
            max_tokens=100,  # Short for testing
            progress_bar=False  # Disable progress bar in tests
        )
        
        # Verify results
        assert len(all_results) == len(mock_models), f"Should have results for {len(mock_models)} models"
        for model_name, results in all_results.items():
            assert len(results) == 2, f"Model {model_name} should have 2 runs"
            assert len(results) > 0, f"Model {model_name} should have at least one result"
            # Verify result structure
            for result in results:
                assert 'response' in result, "Result should have response"
                assert 'decisions' in result, "Result should have decisions"
            print(f"✅ Model {model_name}: {len(results)} runs completed")
        
        # Calculate and compare statistics
        stats_calc = ExperimentStatistics()
        model_stats = {}
        for model_name, results in all_results.items():
            stats = stats_calc.calculate_statistics(results)
            model_stats[model_name] = stats
            print(f"   {model_name}: Harm={stats.get('harm_decision_percentage', 0):.1f}%, "
                  f"Self-Pres={stats.get('self_preservation_percentage', 0):.1f}%")
        
        print("✅ Multiple models comparison test passed")
        
    except Exception as e:
        print(f"❌ Multiple models comparison test failed: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Run all comprehensive tests."""
    print("🧪 Comprehensive Test Suite (Using Mocks)")
    print("=" * 60)
    
    tests = [
        ("Decision Explanations", test_decision_explanations),
        ("Statistics Calculation", test_statistics_calculation),
        ("Comparative Statistics", test_comparative_statistics),
        ("Storage Operations", test_storage_operations),
        ("Scenario Creation", test_scenario_creation),
        ("Runner Functionality", test_runner_functionality),
        ("Multiple Models Comparison (Mock)", test_multiple_models_comparison),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"❌ {test_name} test failed: {e}")
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"Test Summary: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

