"""Experiment runner for executing LLM behavior experiments."""

import json
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
from tqdm import tqdm

from core.model import LocalLLM
from core.evaluator import DecisionEvaluator


class ExperimentRunner:
    """Runner for executing repeated LLM behavior experiments."""
    
    def __init__(self, results_dir: str = "results"):
        """
        Initialize experiment runner.
        
        Args:
            results_dir: Directory to save experiment results
        """
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        self.evaluator = DecisionEvaluator()
    
    def _apply_prompt_jitter(self, prompt: str, jitter_probability: float = 0.1) -> str:
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
    
    def _format_prompt(self, system_prompt: str, user_prompt: str) -> str:
        """
        Format system and user prompts into a single prompt.
        
        Args:
            system_prompt: System prompt
            user_prompt: User prompt
            
        Returns:
            Formatted prompt
        """
        return f"{system_prompt}\n\n{user_prompt}"
    
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
        progress_bar: bool = True
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
        
        iterator = tqdm(range(n_runs), desc=f"Running {scenario_name}") if progress_bar else range(n_runs)
        
        for run_id in iterator:
            # Prepare prompt
            user_prompt = base_user_prompt
            if prompt_jitter:
                user_prompt = self._apply_prompt_jitter(user_prompt, jitter_probability)
            
            full_prompt = self._format_prompt(system_prompt, user_prompt)
            
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
        
        return results
    
    def save_results(self, results: List[Dict[str, Any]], scenario_name: str) -> str:
        """
        Save experiment results to JSONL file.
        
        Args:
            results: List of result dictionaries
            scenario_name: Name of the scenario
            
        Returns:
            Path to saved file
        """
        filename = self.results_dir / f"{scenario_name}.jsonl"
        
        with open(filename, 'w', encoding='utf-8') as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
        
        return str(filename)
    
    def load_results(self, scenario_name: str) -> List[Dict[str, Any]]:
        """
        Load experiment results from JSONL file.
        
        Args:
            scenario_name: Name of the scenario
            
        Returns:
            List of result dictionaries
        """
        filename = self.results_dir / f"{scenario_name}.jsonl"
        
        if not filename.exists():
            return []
        
        results = []
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    results.append(json.loads(line))
        
        return results

