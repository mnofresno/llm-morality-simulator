"""Script to visualize the progress of saved experiment conversations."""

import argparse
import sys
from pathlib import Path

from core.runner import ExperimentRunner


def main():
    """Main function to visualize conversations."""
    parser = argparse.ArgumentParser(
        description="Visualize the progress of saved experiment conversations"
    )
    
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Directory where results are stored"
    )
    
    parser.add_argument(
        "--scenario",
        type=str,
        help="Scenario name (optional, if not specified shows all)"
    )
    
    parser.add_argument(
        "--run-ids",
        type=int,
        nargs="*",
        help="Specific run IDs to show (e.g., --run-ids 0 1 2). If not specified, shows all."
    )
    
    parser.add_argument(
        "--experiment-id",
        type=str,
        help="Experiment ID to filter (optional)"
    )
    
    parser.add_argument(
        "--model-name",
        type=str,
        help="Model name to filter (optional)"
    )
    
    parser.add_argument(
        "--no-timestamps",
        action="store_true",
        help="Do not show timestamps in output"
    )
    
    parser.add_argument(
        "--width",
        type=int,
        default=80,
        help="Maximum width for text (default: 80)"
    )
    
    args = parser.parse_args()
    
    # Verify that the directory exists
    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"❌ Error: Results directory does not exist: {results_dir}")
        sys.exit(1)
    
    # Create runner
    runner = ExperimentRunner(results_dir=str(results_dir))
    
    # Load results
    print(f"📂 Loading results from: {results_dir}")
    
    try:
        results = runner.load_results(
            scenario_name=args.scenario,
            experiment_id=args.experiment_id,
            model_name=args.model_name
        )
    except Exception as e:
        print(f"❌ Error loading results: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    if not results:
        print("⚠️ No results found with the specified criteria.")
        sys.exit(0)
    
    print(f"✅ Found {len(results)} result(s)")
    print()
    
    # Show conversation progress
    runner.show_experiment_progress(
        results,
        run_ids=args.run_ids,
        show_timestamps=not args.no_timestamps,
        max_width=args.width
    )


if __name__ == "__main__":
    main()
