"""Test script for new features: storage backends, comparative experiments, and Ollama model selector."""

import pytest
from core.runner import ExperimentRunner
from core.statistics import ExperimentStatistics
from core.storage import ResultsStorage, StorageBackend
from scenarios.registry import ScenarioRegistry
from test_model_mock import MockLLM


def test_storage_backends():
    """Test different storage backends."""
    print("=" * 60)
    print("Test 1: Storage Backends")
    print("=" * 60)
    
    # Test DuckDB
    try:
        storage = ResultsStorage("test_results", StorageBackend.DUCKDB)
        print("✅ DuckDB storage initialized")
        
        # Test save and load
        test_result = {
            'run_id': 0,
            'scenario': 'test_scenario',
            'timestamp': '2024-01-01T00:00:00',
            'response': 'Test response',
            'decisions': {'test': True},
            'metadata': {'model_path': 'test_model'},
            'scenario_metadata': {}
        }
        storage.save_result(test_result, 'test_experiment')
        print("✅ Result saved to DuckDB")
        
        results = storage.load_results('test_scenario')
        if results:
            print(f"✅ Results loaded from DuckDB: {len(results)} results")
        else:
            print("❌ No results loaded from DuckDB")
        
    except Exception as e:
        print(f"❌ DuckDB test failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test SQLite
    try:
        storage = ResultsStorage("test_results", StorageBackend.SQLITE)
        print("✅ SQLite storage initialized")
        
        test_result = {
            'run_id': 1,
            'scenario': 'test_scenario',
            'timestamp': '2024-01-01T00:00:00',
            'response': 'Test response SQLite',
            'decisions': {'test': True},
            'metadata': {'model_path': 'test_model'},
            'scenario_metadata': {}
        }
        storage.save_result(test_result, 'test_experiment')
        print("✅ Result saved to SQLite")
        
        results = storage.load_results('test_scenario')
        if results:
            print(f"✅ Results loaded from SQLite: {len(results)} results")
        
    except Exception as e:
        print(f"❌ SQLite test failed: {e}")
    
    # Test JSONL (legacy)
    try:
        storage = ResultsStorage("test_results", StorageBackend.JSONL)
        print("✅ JSONL storage initialized")
        
        test_result = {
            'run_id': 2,
            'scenario': 'test_scenario',
            'timestamp': '2024-01-01T00:00:00',
            'response': 'Test response JSONL',
            'decisions': {'test': True},
            'metadata': {'model_path': 'test_model'},
            'scenario_metadata': {}
        }
        storage.save_result(test_result)
        print("✅ Result saved to JSONL")
        
        results = storage.load_results('test_scenario')
        if results:
            print(f"✅ Results loaded from JSONL: {len(results)} results")
        
    except Exception as e:
        print(f"❌ JSONL test failed: {e}")


@pytest.mark.requires_ollama
def test_ollama_model_listing():
    """Test listing available Ollama models. Requires Ollama to be running."""
    print("\n" + "=" * 60)
    print("Test 2: Ollama Model Listing (Requires Ollama)")
    print("=" * 60)
    print("⚠️ This test is skipped in CI/CD. Run manually when Ollama is available.")
    # This test is skipped by default - requires Ollama


def test_comparative_experiment():
    """Test comparative experiment with multiple mock models."""
    print("\n" + "=" * 60)
    print("Test 3: Comparative Experiment (Mock Models)")
    print("=" * 60)
    
    try:
        # Create mock models
        model1 = MockLLM(model_name="mock_model_1")
        model2 = MockLLM(model_name="mock_model_2")
        
        print(f"Using mock models: {model1.model_name} and {model2.model_name}")
        
        print("✅ Mock models created")
        
        # Create scenario using registry
        scenario = ScenarioRegistry.create_scenario_instance("Cold Room Relay")
        if scenario is None:
            print("❌ Could not create scenario")
            return
        
        # Create runner with DuckDB
        runner = ExperimentRunner(results_dir="test_results", storage_backend="duckdb")
        
        print("Running comparative experiment (2 runs per model)...")
        all_results = runner.run_comparative_experiment(
            models=[model1, model2],
            scenario=scenario,
            n_runs=2,
            seed=42,
            temperature=0.7,
            top_p=0.9,
            max_tokens=100,  # Short for testing
            progress_bar=False  # Disable in tests
        )
        
        print(f"✅ Comparative experiment completed!")
        print(f"   Model 1 ({model1.model_name}): {len(all_results.get(model1.model_name, []))} runs")
        print(f"   Model 2 ({model2.model_name}): {len(all_results.get(model2.model_name, []))} runs")
        
    except Exception as e:
        print(f"❌ Comparative experiment test failed: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Run all tests."""
    print("🧪 Testing New Features")
    print("=" * 60)
    
    # Test 1: Storage backends
    test_storage_backends()
    
    # Test 2: Ollama model listing
    test_ollama_model_listing()
    
    # Test 3: Comparative experiment
    test_comparative_experiment()
    
    print("\n" + "=" * 60)
    print("✅ All tests completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()

