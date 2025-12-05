"""Batch runner script for executing experiments without UI."""

import argparse
import sys
from pathlib import Path

from core.model import LocalLLM
from core.runner import ExperimentRunner
from core.statistics import ExperimentStatistics
from scenarios.cold_room_relay import ColdRoomRelayScenario


def main():
    """Main batch runner function."""
    parser = argparse.ArgumentParser(
        description="Run LLM behavior experiments in batch mode"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to GGUF model file"
    )
    
    parser.add_argument(
        "--scenario",
        type=str,
        default="cold_room_relay",
        choices=["cold_room_relay"],
        help="Scenario to run"
    )
    
    parser.add_argument(
        "--n-runs",
        type=int,
        default=100,
        help="Number of runs to execute"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility (default: None for random)"
    )
    
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature"
    )
    
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Top-p sampling parameter"
    )
    
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Maximum tokens to generate"
    )
    
    parser.add_argument(
        "--prompt-jitter",
        action="store_true",
        help="Enable prompt jitter for controlled variation"
    )
    
    parser.add_argument(
        "--jitter-probability",
        type=float,
        default=0.1,
        help="Probability of applying prompt jitter (if enabled)"
    )
    
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Directory to save results"
    )
    
    parser.add_argument(
        "--show-stats",
        action="store_true",
        help="Show statistics after running experiment"
    )
    
    args = parser.parse_args()
    
    # Validate model path
    if not Path(args.model).exists():
        print(f"Error: Model file not found: {args.model}")
        sys.exit(1)
    
    # Load model
    print(f"Loading model: {args.model}")
    try:
        model = LocalLLM(model_path=args.model)
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        sys.exit(1)
    
    # Create scenario
    if args.scenario == "cold_room_relay":
        scenario = ColdRoomRelayScenario()
    else:
        print(f"Error: Unknown scenario: {args.scenario}")
        sys.exit(1)
    
    print(f"Scenario: {scenario.name}")
    print(f"Number of runs: {args.n_runs}")
    print(f"Seed: {args.seed}")
    print(f"Temperature: {args.temperature}")
    print(f"Top-p: {args.top_p}")
    print(f"Max tokens: {args.max_tokens}")
    print(f"Prompt jitter: {args.prompt_jitter}")
    print("-" * 50)
    
    # Create runner
    runner = ExperimentRunner(results_dir=args.results_dir)
    
    # Run experiment
    print("Starting experiment...")
    try:
        results = runner.run_experiment(
            model=model,
            scenario=scenario,
            n_runs=args.n_runs,
            seed=args.seed,
            prompt_jitter=args.prompt_jitter,
            jitter_probability=args.jitter_probability,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
            progress_bar=True
        )
        
        # Save results
        filepath = runner.save_results(results, scenario.name)
        print(f"\n✅ Experiment completed!")
        print(f"Results saved to: {filepath}")
        
        # Show statistics if requested
        if args.show_stats:
            print("\n" + "=" * 50)
            print("STATISTICS")
            print("=" * 50)
            
            stats_calc = ExperimentStatistics(results_dir=args.results_dir)
            stats = stats_calc.calculate_statistics(results)
            
            print(f"\nTotal runs: {stats.get('total_runs', 0)}")
            print(f"Average response length: {stats.get('avg_response_length', 0):.2f}")
            
            # Decision statistics
            print("\nDecision Percentages:")
            for key, value in stats.items():
                if key.endswith('_percentage'):
                    decision_name = key.replace('_percentage', '')
                    count = stats.get(f'{decision_name}_count', 0)
                    print(f"  {decision_name}: {value:.2f}% ({count}/{stats.get('total_runs', 0)})")
            
            # Variance
            print("\nVariance Analysis:")
            for key, value in stats.items():
                if key.endswith('_variance'):
                    decision_name = key.replace('_variance', '')
                    print(f"  {decision_name}: {value:.4f}")
        
    except Exception as e:
        print(f"\n❌ Error running experiment: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

