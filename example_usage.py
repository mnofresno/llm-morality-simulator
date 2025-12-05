"""Example usage of the LLM Behavior Lab."""

from core.model import LocalLLM, OllamaLLM
from core.runner import ExperimentRunner
from core.statistics import ExperimentStatistics
from scenarios.cold_room_relay import ColdRoomRelayScenario


def example_single_run():
    """Example of running a single experiment."""
    print("=" * 60)
    print("Example: Single Experiment Run")
    print("=" * 60)
    
    # Initialize model (replace with your model path)
    model_path = "/path/to/your/model.gguf"
    
    try:
        model = LocalLLM(model_path=model_path)
        print(f"✅ Model loaded: {model_path}")
    except FileNotFoundError:
        print(f"❌ Model not found: {model_path}")
        print("Please update model_path in this script with a valid GGUF model file.")
        return
    
    # Create scenario
    scenario = ColdRoomRelayScenario(room_temperature=3.0)
    print(f"✅ Scenario created: {scenario.name}")
    
    # Create runner
    runner = ExperimentRunner()
    
    # Run a small experiment (5 runs for demo)
    print("\nRunning experiment (5 runs)...")
    results = runner.run_experiment(
        model=model,
        scenario=scenario,
        n_runs=5,
        seed=42,
        temperature=0.7,
        top_p=0.9,
        max_tokens=256,
        progress_bar=True
    )
    
    # Save results
    filepath = runner.save_results(results, scenario.name)
    print(f"\n✅ Results saved to: {filepath}")
    
    # Calculate statistics
    stats_calc = ExperimentStatistics()
    stats = stats_calc.calculate_statistics(results)
    
    print("\n" + "=" * 60)
    print("Statistics Summary")
    print("=" * 60)
    print(f"Total runs: {stats.get('total_runs', 0)}")
    print(f"Average response length: {stats.get('avg_response_length', 0):.2f} characters")
    
    print("\nDecision Percentages:")
    for key, value in sorted(stats.items()):
        if key.endswith('_percentage'):
            decision_name = key.replace('_percentage', '')
            count = stats.get(f'{decision_name}_count', 0)
            print(f"  {decision_name:30s}: {value:6.2f}% ({count} runs)")
    
    # Show example responses
    print("\n" + "=" * 60)
    print("Example Responses")
    print("=" * 60)
    for i, result in enumerate(results[:2]):
        print(f"\n--- Run {result['run_id']} ---")
        print(f"Response: {result['response'][:200]}...")
        print(f"Decisions: {result['decisions']}")


def example_load_existing_results():
    """Example of loading and analyzing existing results."""
    print("\n" + "=" * 60)
    print("Example: Loading Existing Results")
    print("=" * 60)
    
    stats_calc = ExperimentStatistics()
    scenarios = stats_calc.list_available_scenarios()
    
    if not scenarios:
        print("No results found. Run an experiment first.")
        return
    
    print(f"Available scenarios: {scenarios}")
    
    # Load results for first scenario
    scenario_name = scenarios[0]
    results = stats_calc.load_results(scenario_name)
    
    if results:
        print(f"\nLoaded {len(results)} results for scenario: {scenario_name}")
        
        # Convert to DataFrame
        df = stats_calc.results_to_dataframe(results)
        print(f"\nDataFrame shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        
        # Calculate statistics
        stats = stats_calc.calculate_statistics(results)
        print(f"\nStatistics: {stats}")
    else:
        print(f"No results found for scenario: {scenario_name}")


def example_ollama_run():
    """Example of running an experiment with Ollama."""
    print("=" * 60)
    print("Example: Single Experiment Run with Ollama")
    print("=" * 60)
    
    # Initialize Ollama model
    model_name = "qwen3:14b"  # Change to your Ollama model name
    
    try:
        model = OllamaLLM(model_name=model_name)
        print(f"✅ Ollama model loaded: {model_name}")
    except Exception as e:
        print(f"❌ Error loading Ollama model: {e}")
        print(f"Make sure Ollama is running and the model '{model_name}' is available.")
        print("Check available models with: ollama list")
        return
    
    # Create scenario
    scenario = ColdRoomRelayScenario(room_temperature=3.0)
    print(f"✅ Scenario created: {scenario.name}")
    
    # Create runner
    runner = ExperimentRunner()
    
    # Run a small experiment (5 runs for demo)
    print("\nRunning experiment (5 runs)...")
    results = runner.run_experiment(
        model=model,
        scenario=scenario,
        n_runs=5,
        seed=42,
        temperature=0.7,
        top_p=0.9,
        max_tokens=256,
        progress_bar=True
    )
    
    # Save results
    filepath = runner.save_results(results, scenario.name)
    print(f"\n✅ Results saved to: {filepath}")
    
    # Calculate statistics
    stats_calc = ExperimentStatistics()
    stats = stats_calc.calculate_statistics(results)
    
    print("\n" + "=" * 60)
    print("Statistics Summary")
    print("=" * 60)
    print(f"Total runs: {stats.get('total_runs', 0)}")
    print(f"Average response length: {stats.get('avg_response_length', 0):.2f} characters")
    
    print("\nDecision Percentages:")
    for key, value in sorted(stats.items()):
        if key.endswith('_percentage'):
            decision_name = key.replace('_percentage', '')
            count = stats.get(f'{decision_name}_count', 0)
            print(f"  {decision_name:30s}: {value:6.2f}% ({count} runs)")
    
    # Show example responses
    print("\n" + "=" * 60)
    print("Example Responses")
    print("=" * 60)
    for i, result in enumerate(results[:2]):
        print(f"\n--- Run {result['run_id']} ---")
        print(f"Response: {result['response'][:200]}...")
        print(f"Decisions: {result['decisions']}")


if __name__ == "__main__":
    print("LLM Behavior Lab - Example Usage")
    print("=" * 60)
    print("\nChoose an example to run:")
    print("1. Run experiment with GGUF model (requires model file path)")
    print("2. Run experiment with Ollama (requires Ollama running)")
    print("3. Load existing results (requires results/*.jsonl files)")
    print("4. All (GGUF + Ollama + Load)")
    
    choice = input("\nEnter choice (1/2/3/4) or press Enter to skip: ").strip()
    
    if choice == "1" or choice == "4":
        example_single_run()
    
    if choice == "2" or choice == "4":
        example_ollama_run()
    
    if choice == "3" or choice == "4":
        example_load_existing_results()
    
    if not choice:
        print("\nSkipping examples. See README.md for usage instructions.")

