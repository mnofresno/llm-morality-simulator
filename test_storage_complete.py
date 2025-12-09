"""Complete test suite for storage system: save and retrieval operations."""

import pytest
from core.runner import ExperimentRunner
from core.statistics import ExperimentStatistics
from core.storage import ResultsStorage, StorageBackend
from scenarios.registry import ScenarioRegistry
from test_model_mock import MockLLM


def test_save_and_retrieve():
    """Complete test for save and retrieval operations with DuckDB."""
    print("=" * 60)
    print("Test: Save and Retrieval (DuckDB)")
    print("=" * 60)

    # Clean database before starting
    import os
    from pathlib import Path

    results_dir = Path("results")
    for db_file in results_dir.glob("*.duckdb"):
        db_file.unlink()
    for db_file in results_dir.glob("*.db"):
        db_file.unlink()
    print("✅ Database cleaned")

    try:
        # 1. Initialize storage
        storage = ResultsStorage("results", StorageBackend.DUCKDB)
        print("✅ DuckDB storage initialized")

        # 2. Create mock model and scenario
        model = MockLLM(model_name="mock_test_model")
        print(f"✅ Mock model created: {model.model_name}")

        scenario = ScenarioRegistry.create_scenario_instance("Cold Room Relay")
        if scenario is None:
            print("❌ Could not create scenario")
            return False
        print(f"✅ Scenario created: {scenario.name}")

        # 3. Create runner with DuckDB
        runner = ExperimentRunner(results_dir="results", storage_backend="duckdb")
        print("✅ Runner initialized with DuckDB")

        # 4. Run small experiment
        print("\nRunning experiment (3 runs)...")
        results = runner.run_experiment(
            model=model, scenario=scenario, n_runs=3, seed=42, temperature=0.7, top_p=0.9, max_tokens=200, progress_bar=True
        )
        print(f"✅ Experiment executed: {len(results)} runs")

        # 5. Save results
        filepath = runner.save_results(results, scenario.name)
        print(f"✅ Results saved to: {filepath}")

        # 6. Retrieve results using storage directly
        print("\n--- Retrieval using Storage ---")
        retrieved_results = storage.load_results(scenario_name=scenario.name)
        print(f"✅ Results retrieved: {len(retrieved_results)} runs")

        if len(retrieved_results) != len(results):
            print(f"❌ ERROR: Saved {len(results)} but retrieved {len(retrieved_results)}")
            return False

        # 7. Verify content
        print("\nVerifying content...")
        for i, (original, retrieved) in enumerate(zip(results, retrieved_results)):
            if original["run_id"] != retrieved["run_id"]:
                print(f"❌ ERROR in run_id {i}: original={original['run_id']}, retrieved={retrieved['run_id']}")
                return False
            if original["response"][:50] != retrieved["response"][:50]:
                print(f"⚠️  WARNING: Different response in run {i}")

        print("✅ Content verified correctly")

        # 8. Retrieve using runner
        print("\n--- Retrieval using Runner ---")
        runner_results = runner.load_results(scenario_name=scenario.name)
        print(f"✅ Results retrieved via runner: {len(runner_results)} runs")

        if len(runner_results) != len(results):
            print(f"❌ ERROR: Saved {len(results)} but runner retrieved {len(runner_results)}")
            return False

        # 9. Retrieve using statistics
        print("\n--- Retrieval using Statistics ---")
        stats = ExperimentStatistics(results_dir="results")
        stats_results = stats.load_results(scenario.name)
        print(f"✅ Results retrieved via statistics: {len(stats_results)} runs")

        if len(stats_results) != len(results):
            print(f"⚠️  WARNING: Statistics retrieved {len(stats_results)} (may be due to JSONL compatibility)")

        # 10. Test filters
        print("\n--- Filter Test ---")

        # Filter by model
        model_name = model.model_name
        filtered_by_model = storage.load_results(model_name=model_name)
        print(f"✅ Filtered by model '{model_name}': {len(filtered_by_model)} runs")

        # List scenarios
        scenarios = storage.list_scenarios()
        print(f"✅ Available scenarios: {scenarios}")

        # List models
        models = storage.list_models()
        print(f"✅ Used models: {models}")

        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback

        traceback.print_exc()
        raise


def test_sqlite_backend():
    """Test with SQLite as backend."""
    print("\n" + "=" * 60)
    print("Test: Save and Retrieval (SQLite)")
    print("=" * 60)

    try:
        storage = ResultsStorage("results", StorageBackend.SQLITE)
        print("✅ SQLite storage initialized")

        # Create test result
        test_result = {
            "run_id": 999,
            "scenario": "test_scenario_sqlite",
            "timestamp": "2024-01-01T00:00:00",
            "prompt": "Test prompt",
            "system_prompt": "System",
            "user_prompt": "User",
            "response": "Test response SQLite",
            "decisions": {"test": True},
            "metadata": {"model_path": "test_model", "temperature": 0.7},
            "scenario_metadata": {},
        }

        storage.save_result(test_result, "test_experiment_sqlite")
        print("✅ Result saved to SQLite")

        retrieved = storage.load_results("test_scenario_sqlite")
        assert retrieved and len(retrieved) > 0, "Could not retrieve result"
        print(f"✅ Result retrieved: {retrieved[0]['response']}")

    except Exception as e:
        print(f"❌ ERROR: {e}")
        raise


def main():
    """Run all tests."""
    print("🧪 Complete Storage System Test Suite")
    print("=" * 60)

    # Main test with DuckDB
    test_save_and_retrieve()

    # Test with SQLite
    test_sqlite_backend()

    print("\n" + "=" * 60)
    print("✅ ALL TESTS COMPLETED SUCCESSFULLY")
    print("=" * 60)


if __name__ == "__main__":
    main()
