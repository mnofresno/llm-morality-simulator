"""Test script for new features: storage backends, comparative experiments, and Ollama model selector."""

from core.model import OllamaLLM
from core.runner import ExperimentRunner
from core.statistics import ExperimentStatistics
from core.storage import ResultsStorage, StorageBackend
from scenarios.cold_room_relay import ColdRoomRelayScenario


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


def test_ollama_model_listing():
    """Test listing available Ollama models."""
    print("\n" + "=" * 60)
    print("Test 2: Ollama Model Listing")
    print("=" * 60)
    
    try:
        models = OllamaLLM.list_available_models()
        if models:
            print(f"✅ Found {len(models)} Ollama models:")
            for model in models:
                print(f"   - {model}")
        else:
            print("⚠️ No Ollama models found (Ollama may not be running)")
    except Exception as e:
        print(f"❌ Error listing Ollama models: {e}")


def test_comparative_experiment():
    """Test comparative experiment with multiple models."""
    print("\n" + "=" * 60)
    print("Test 3: Comparative Experiment")
    print("=" * 60)
    
    try:
        # Get available models
        models_list = OllamaLLM.list_available_models()
        if len(models_list) < 2:
            print("⚠️ Need at least 2 Ollama models for comparative test. Skipping.")
            return
        
        # Use first 2 models
        model1_name = models_list[0]
        model2_name = models_list[1] if len(models_list) > 1 else models_list[0]
        
        print(f"Using models: {model1_name} and {model2_name}")
        
        model1 = OllamaLLM(model_name=model1_name)
        model2 = OllamaLLM(model_name=model2_name)
        
        print("✅ Models loaded")
        
        # Create scenario
        scenario = ColdRoomRelayScenario(room_temperature=3.0)
        
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
            progress_bar=True
        )
        
        print(f"✅ Comparative experiment completed!")
        print(f"   Model 1 ({model1_name}): {len(all_results.get(model1_name, []))} runs")
        print(f"   Model 2 ({model2_name}): {len(all_results.get(model2_name, []))} runs")
        
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

