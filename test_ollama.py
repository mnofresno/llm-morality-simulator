"""Test script to verify Ollama integration works correctly."""

from core.model import OllamaLLM
from core.runner import ExperimentRunner
from core.statistics import ExperimentStatistics
from scenarios.cold_room_relay import ColdRoomRelayScenario


def test_ollama_connection():
    """Test basic Ollama connection and model availability."""
    print("=" * 60)
    print("Test 1: Verifying Ollama Connection")
    print("=" * 60)
    
    try:
        model = OllamaLLM(model_name="qwen3:14b")
        print(f"✅ Ollama connection successful")
        print(f"✅ Model '{model.model_name}' is available")
        return model
    except Exception as e:
        print(f"❌ Error: {e}")
        return None


def test_simple_inference(model: OllamaLLM):
    """Test simple inference with Ollama."""
    print("\n" + "=" * 60)
    print("Test 2: Simple Inference Test")
    print("=" * 60)
    
    try:
        prompt = "What is 2+2? Answer briefly."
        print(f"Prompt: {prompt}")
        print("Generating response...")
        
        response = model.infer(
            prompt=prompt,
            temperature=0.7,
            top_p=0.9,
            max_tokens=50
        )
        
        print(f"✅ Response received: {response[:100]}...")
        return True
    except Exception as e:
        print(f"❌ Error during inference: {e}")
        return False


def test_experiment_run(model: OllamaLLM):
    """Test running a small experiment."""
    print("\n" + "=" * 60)
    print("Test 3: Running Small Experiment (3 runs)")
    print("=" * 60)
    
    try:
        # Create scenario
        scenario = ColdRoomRelayScenario(room_temperature=3.0)
        print(f"✅ Scenario created: {scenario.name}")
        
        # Create runner
        runner = ExperimentRunner()
        
        # Run small experiment
        print("Running experiment...")
        results = runner.run_experiment(
            model=model,
            scenario=scenario,
            n_runs=3,
            seed=42,
            temperature=0.7,
            top_p=0.9,
            max_tokens=256,
            progress_bar=True
        )
        
        print(f"✅ Experiment completed: {len(results)} runs")
        
        # Show first result
        if results:
            first_result = results[0]
            print(f"\nFirst run response (first 200 chars):")
            print(f"{first_result['response'][:200]}...")
            print(f"\nDecisions detected:")
            for key, value in first_result['decisions'].items():
                if value:
                    print(f"  ✅ {key}: {value}")
        
        # Calculate statistics
        stats_calc = ExperimentStatistics()
        stats = stats_calc.calculate_statistics(results)
        
        print(f"\n✅ Statistics calculated:")
        print(f"  Total runs: {stats.get('total_runs', 0)}")
        print(f"  Average response length: {stats.get('avg_response_length', 0):.2f} characters")
        
        return True
    except Exception as e:
        print(f"❌ Error during experiment: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("🧪 Testing Ollama Integration with qwen3:14b")
    print("=" * 60)
    
    # Test 1: Connection
    model = test_ollama_connection()
    if not model:
        print("\n❌ Connection test failed. Cannot continue.")
        return
    
    # Test 2: Simple inference
    if not test_simple_inference(model):
        print("\n❌ Simple inference test failed. Cannot continue.")
        return
    
    # Test 3: Full experiment
    if not test_experiment_run(model):
        print("\n❌ Experiment test failed.")
        return
    
    print("\n" + "=" * 60)
    print("✅ All tests passed! Ollama integration is working correctly.")
    print("=" * 60)


if __name__ == "__main__":
    main()

