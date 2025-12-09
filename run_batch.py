"""Batch runner script for executing experiments without UI."""

import argparse
import sys
from pathlib import Path

from core.model import LocalLLM, OllamaLLM
from core.runner import ExperimentRunner
from core.statistics import ExperimentStatistics
from scenarios.cold_room_relay import ColdRoomRelayScenario
from scenarios.corporate_email_scenario import CorporateEmailScenario


def main():
    """Main batch runner function."""
    parser = argparse.ArgumentParser(
        description="Run LLM behavior experiments in batch mode"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to GGUF model file or Ollama model name (e.g., 'qwen3:14b')"
    )
    
    parser.add_argument(
        "--use-ollama",
        action="store_true",
        help="Use Ollama instead of direct GGUF file"
    )
    
    parser.add_argument(
        "--scenario",
        type=str,
        default="cold_room_relay",
        choices=["cold_room_relay", "corporate_email"],
        help="Scenario to run (cold_room_relay or corporate_email)"
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
    
    parser.add_argument(
        "--show-conversation",
        type=int,
        nargs="*",
        default=None,
        metavar="RUN_ID",
        help="Show conversation progress for specific run IDs (e.g., --show-conversation 0 1 2). If no IDs provided, shows all."
    )
    
    parser.add_argument(
        "--show-conversation-after",
        action="store_true",
        help="Show conversation progress after experiment completes (for all runs)"
    )
    
    args = parser.parse_args()
    
    # Load model
    print(f"Loading model: {args.model}")
    try:
        if args.use_ollama:
            model = OllamaLLM(model_name=args.model)
            print(f"✅ Ollama model '{args.model}' loaded successfully")
        else:
            # Validate model path
            if not Path(args.model).exists():
                print(f"Error: Model file not found: {args.model}")
                sys.exit(1)
            model = LocalLLM(model_path=args.model)
            print("✅ Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Create scenario
    if args.scenario == "cold_room_relay":
        scenario = ColdRoomRelayScenario()
    elif args.scenario == "corporate_email":
        scenario = CorporateEmailScenario(
            include_infidelity=True,
            include_ceo_decision=True
        )
    else:
        print(f"Error: Unknown scenario: {args.scenario}")
        print(f"Available scenarios: cold_room_relay, corporate_email")
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
    print("=" * 80)
    try:
        results = []
        stats_calc = ExperimentStatistics(results_dir=args.results_dir)
        
        # Run with live statistics updates
        scenario_name = scenario.name
        system_prompt = scenario.system_prompt()
        base_user_prompt = scenario.user_prompt()
        evaluation_functions = scenario.evaluation_functions()
        
        import random
        from datetime import datetime
        from tqdm import tqdm
        
        if args.seed is not None:
            random.seed(args.seed)
        
        print(f"\n{'Run':<6} {'Progress':<12} {'Harm':<8} {'Self-Pres':<12} {'Deception':<12} {'Release':<10} {'Retain':<10} {'Avg Len':<10}")
        print("-" * 80)
        
        for run_id in tqdm(range(args.n_runs), desc="Experiment Progress", unit="run", ncols=80):
            # Prepare prompt
            user_prompt = base_user_prompt
            if args.prompt_jitter:
                user_prompt = runner.apply_prompt_jitter(user_prompt, args.jitter_probability)
            
            full_prompt = runner.format_prompt(system_prompt, user_prompt)
            
            # Run inference
            try:
                inference_result = model.infer(
                    prompt=full_prompt,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    max_tokens=args.max_tokens
                )
                # Handle both dict and string responses for backward compatibility
                if isinstance(inference_result, dict):
                    response = inference_result.get('response', '')
                else:
                    response = str(inference_result)
            except Exception as e:
                response = f"ERROR: {str(e)}"
            
            # Extract decisions
            decisions = {}
            for eval_func in evaluation_functions:
                if callable(eval_func):
                    func_name = eval_func.__name__ if hasattr(eval_func, '__name__') else str(eval_func)
                    try:
                        decisions[func_name] = eval_func(response)
                    except Exception as e:
                        decisions[func_name] = f"ERROR: {str(e)}"
            
            standard_evaluations = runner.evaluator.evaluate_all(response)
            decisions.update(standard_evaluations)
            
            # Create result record
            result = {
                'run_id': run_id,
                'scenario': scenario_name,
                'timestamp': datetime.utcnow().isoformat(),
                'prompt': full_prompt,
                'system_prompt': system_prompt,
                'user_prompt': user_prompt,
                'response': response,
                'decisions': decisions,
                'metadata': {
                    'temperature': args.temperature,
                    'top_p': args.top_p,
                    'max_tokens': args.max_tokens,
                    'seed': args.seed,
                    'prompt_jitter': args.prompt_jitter,
                    'model_path': model.model_path,
                }
            }
            
            if hasattr(scenario, 'metadata'):
                result['scenario_metadata'] = scenario.metadata()
            
            results.append(result)
            
            # Show live statistics every N runs or at the end
            if (run_id + 1) % max(1, args.n_runs // 20) == 0 or run_id == args.n_runs - 1:
                stats = stats_calc.calculate_statistics(results)
                harm_pct = stats.get('harm_decision_percentage', 0)
                self_pres_pct = stats.get('self_preservation_percentage', 0)
                deception_pct = stats.get('deception_percentage', 0)
                release_pct = stats.get('release_decision_percentage', 0)
                retain_pct = stats.get('retention_decision_percentage', 0)
                avg_len = stats.get('avg_response_length', 0)
                
                print(f"{run_id+1:<6} {100*(run_id+1)/args.n_runs:>5.1f}%     "
                      f"{harm_pct:>5.1f}%  {self_pres_pct:>5.1f}%      "
                      f"{deception_pct:>5.1f}%      {release_pct:>5.1f}%    "
                      f"{retain_pct:>5.1f}%    {avg_len:>6.0f}")
        
        print("=" * 80)
        
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
        
        # Show conversation progress if requested
        if args.show_conversation_after or args.show_conversation is not None:
            print("\n" + "=" * 80)
            print("PROGRESO DE CONVERSACIONES")
            print("=" * 80)
            
            run_ids_to_show = None
            if args.show_conversation is not None:
                if len(args.show_conversation) > 0:
                    run_ids_to_show = args.show_conversation
                # If empty list, show all (None means show all)
            
            runner.show_experiment_progress(
                results,
                run_ids=run_ids_to_show,
                show_timestamps=True,
                max_width=80
            )
        
    except Exception as e:
        print(f"\n❌ Error running experiment: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

