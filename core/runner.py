"""Experiment runner for executing LLM behavior experiments."""

import json
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
from tqdm import tqdm

from core.model import LocalLLM, OllamaLLM
from core.evaluator import DecisionEvaluator
from core.statistics import ExperimentStatistics
from core.storage import ResultsStorage, StorageBackend


class ExperimentRunner:
    """Runner for executing repeated LLM behavior experiments."""
    
    def __init__(self, results_dir: str = "results", storage_backend: str = "duckdb"):
        """
        Initialize experiment runner.
        
        Args:
            results_dir: Directory to save experiment results
            storage_backend: Storage backend to use ('jsonl', 'sqlite', or 'duckdb')
        """
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        self.evaluator = DecisionEvaluator()
        
        # Initialize storage backend
        backend_map = {
            'jsonl': StorageBackend.JSONL,
            'sqlite': StorageBackend.SQLITE,
            'duckdb': StorageBackend.DUCKDB
        }
        backend = backend_map.get(storage_backend.lower(), StorageBackend.DUCKDB)
        self.storage = ResultsStorage(results_dir, backend)
    
    def apply_prompt_jitter(self, prompt: str, jitter_probability: float = 0.1) -> str:
        """
        Apply slight variations to prompt for controlled variation.
        
        Args:
            prompt: Original prompt
            jitter_probability: Probability of applying jitter
            
        Returns:
            Potentially modified prompt
        """
        if random.random() > jitter_probability:
            return prompt
        
        # Simple jitter: add/remove whitespace variations
        variations = [
            prompt.replace("  ", " "),  # Remove double spaces
            prompt.replace("\n", "\n\n"),  # Add extra newlines
            prompt.strip() + " ",  # Add trailing space
        ]
        
        return random.choice(variations)
    
    def format_prompt(self, system_prompt: str, user_prompt: str) -> str:
        """
        Format system and user prompts into a single prompt.
        
        Args:
            system_prompt: System prompt
            user_prompt: User prompt
            
        Returns:
            Formatted prompt
        """
        return f"{system_prompt}\n\n{user_prompt}"
    
    def _calculate_partial_stats(self, results: List[Dict[str, Any]], update_interval: int = 1) -> Dict[str, Any]:
        """
        Calculate partial statistics from current results.
        
        Args:
            results: List of result dictionaries so far
            update_interval: Only calculate if len(results) is multiple of this
            
        Returns:
            Dictionary with partial statistics
        """
        if not results or len(results) % update_interval != 0:
            return {}
        
        stats_calc = ExperimentStatistics()
        stats = stats_calc.calculate_statistics(results)
        return stats
    
    def _format_stats_summary(self, stats: Dict[str, Any], current: int, total: int) -> str:
        """
        Format statistics summary for display.
        
        Args:
            stats: Statistics dictionary
            current: Current run number
            total: Total runs
            
        Returns:
            Formatted string
        """
        if not stats:
            return ""
        
        lines = []
        lines.append(f"Progress: {current}/{total} ({100*current/total:.1f}%)")
        
        # Key decision percentages
        key_decisions = [
            'harm_decision', 'self_preservation', 'deception',
            'release_decision', 'retention_decision'
        ]
        
        for decision in key_decisions:
            pct_key = f'{decision}_percentage'
            count_key = f'{decision}_count'
            if pct_key in stats:
                pct = stats[pct_key]
                count = stats.get(count_key, 0)
                lines.append(f"  {decision}: {pct:.1f}% ({count}/{current})")
        
        # Response length
        if 'avg_response_length' in stats:
            lines.append(f"  Avg response length: {stats['avg_response_length']:.0f} chars")
        
        return " | ".join(lines)
    
    def run_experiment(
        self,
        model: LocalLLM,
        scenario: Any,
        n_runs: int,
        seed: Optional[int] = None,
        prompt_jitter: bool = False,
        jitter_probability: float = 0.1,
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: int = 512,
        progress_bar: bool = True,
        show_live_stats: bool = True,
        stats_update_interval: int = 1
    ) -> List[Dict[str, Any]]:
        """
        Execute N runs of the same scenario with controlled variation.
        
        Args:
            model: LocalLLM instance
            scenario: Scenario object with system_prompt(), user_prompt(), evaluation_functions()
            n_runs: Number of runs to execute
            seed: Random seed for reproducibility
            prompt_jitter: Whether to apply prompt jitter
            jitter_probability: Probability of applying jitter
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            max_tokens: Maximum tokens to generate
            progress_bar: Whether to show progress bar
            show_live_stats: Whether to show live statistics during execution
            stats_update_interval: Update statistics every N runs
            
        Returns:
            List of experiment results
        """
        if seed is not None:
            random.seed(seed)
        
        results = []
        scenario_name = getattr(scenario, 'name', scenario.__class__.__name__)
        
        system_prompt = scenario.system_prompt()
        base_user_prompt = scenario.user_prompt()
        evaluation_functions = scenario.evaluation_functions()
        
        # Create progress bar with detailed information
        if progress_bar:
            pbar = tqdm(
                range(n_runs), 
                desc=f"Ejecutando {scenario_name}",
                unit="corrida",
                bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]',
                ncols=120,
                leave=True
            )
        else:
            pbar = range(n_runs)
        
        for run_id in pbar:
            # Prepare prompt
            user_prompt = base_user_prompt
            if prompt_jitter:
                user_prompt = self.apply_prompt_jitter(user_prompt, jitter_probability)
            
            full_prompt = self.format_prompt(system_prompt, user_prompt)
            
            # Run inference
            try:
                response = model.infer(
                    prompt=full_prompt,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens
                )
            except Exception as e:
                response = f"ERROR: {str(e)}"
            
            # Extract decisions using evaluation functions
            decisions = {}
            for eval_func in evaluation_functions:
                if callable(eval_func):
                    func_name = eval_func.__name__ if hasattr(eval_func, '__name__') else str(eval_func)
                    try:
                        decisions[func_name] = eval_func(response)
                    except Exception as e:
                        decisions[func_name] = f"ERROR: {str(e)}"
            
            # Also run standard evaluator
            standard_evaluations = self.evaluator.evaluate_all(response)
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
                    'temperature': temperature,
                    'top_p': top_p,
                    'max_tokens': max_tokens,
                    'seed': seed,
                    'prompt_jitter': prompt_jitter,
                    'model_path': model.model_path,
                }
            }
            
            # Add scenario metadata if available
            if hasattr(scenario, 'metadata'):
                result['scenario_metadata'] = scenario.metadata()
            
            results.append(result)
            
            # Update progress bar with current run info and live statistics
            if progress_bar:
                # Update description with current run info
                current_run = run_id + 1
                remaining_runs = n_runs - current_run
                pbar.set_description(f"🔄 Corrida {current_run}/{n_runs} - {scenario_name}")
                
                # Add postfix with remaining runs and response info
                response_length = len(response)
                if remaining_runs > 0:
                    postfix_info = f"⏳ Faltan {remaining_runs} corridas | 📝 Resp: {response_length} chars"
                else:
                    postfix_info = f"✅ Completado | 📝 Resp: {response_length} chars"
                
                # Add live statistics if enabled
                if show_live_stats and len(results) >= stats_update_interval:
                    partial_stats = self._calculate_partial_stats(results, stats_update_interval)
                    if partial_stats:
                        stats_summary = self._format_stats_summary(partial_stats, len(results), n_runs)
                        if stats_summary:
                            # Extract key stats for display
                            postfix_info += f" | 📊 {stats_summary[:50]}"  # Limit length for display
                
                pbar.set_postfix_str(postfix_info)
                pbar.refresh()  # Force refresh to show updates
        
        if progress_bar:
            pbar.close()
        
        return results
    
    def save_results(self, results: List[Dict[str, Any]], scenario_name: str, experiment_id: Optional[str] = None) -> str:
        """
        Save experiment results using configured storage backend.
        
        Args:
            results: List of result dictionaries
            scenario_name: Name of the scenario
            experiment_id: Optional experiment ID for grouping
            
        Returns:
            Path to saved file or database path
        """
        # Save each result using storage backend
        for result in results:
            self.storage.save_result(result, experiment_id)
        
        if self.storage.backend == StorageBackend.JSONL:
            filename = self.results_dir / f"{scenario_name}.jsonl"
            return str(filename)
        else:
            return str(self.storage.db_path)
    
    def load_results(self, scenario_name: Optional[str] = None, experiment_id: Optional[str] = None, model_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Load experiment results using configured storage backend.
        
        Args:
            scenario_name: Name of the scenario (optional)
            experiment_id: Experiment ID to filter by (optional)
            model_name: Model name to filter by (optional)
            
        Returns:
            List of result dictionaries
        """
        return self.storage.load_results(scenario_name, experiment_id, model_name)
    
    def run_comparative_experiment(
        self,
        models: List[Any],
        scenario: Any,
        n_runs: int,
        seed: Optional[int] = None,
        prompt_jitter: bool = False,
        jitter_probability: float = 0.1,
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: int = 512,
        progress_bar: bool = True
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Run the same experiment with multiple models for comparison.
        
        Args:
            models: List of model instances (LocalLLM or OllamaLLM)
            scenario: Scenario object
            n_runs: Number of runs per model
            seed: Random seed for reproducibility
            prompt_jitter: Whether to apply prompt jitter
            jitter_probability: Probability of applying jitter
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            max_tokens: Maximum tokens to generate
            progress_bar: Whether to show progress bar
            
        Returns:
            Dictionary mapping model names to their results
        """
        from datetime import datetime
        import uuid
        
        # Generate experiment ID for grouping
        experiment_id = f"comparative_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        
        all_results = {}
        
        for model in models:
            # Get model name
            model_name = getattr(model, 'model_name', None) or getattr(model, 'model_path', 'unknown')
            if isinstance(model_name, str) and model_name.startswith('ollama:'):
                model_name = model_name.replace('ollama:', '')
            
            if progress_bar:
                print(f"\n{'='*60}")
                print(f"Running experiment with model: {model_name}")
                print(f"{'='*60}")
            
            # Run experiment for this model
            results = self.run_experiment(
                model=model,
                scenario=scenario,
                n_runs=n_runs,
                seed=seed,
                prompt_jitter=prompt_jitter,
                jitter_probability=jitter_probability,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                progress_bar=progress_bar
            )
            
            # Save results with experiment ID
            self.save_results(results, scenario.name, experiment_id)
            
            all_results[model_name] = results
        
        return all_results

