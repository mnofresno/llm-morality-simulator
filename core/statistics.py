"""Statistics module for analyzing experiment results."""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np


class ExperimentStatistics:
    """Statistics calculator for experiment results."""
    
    def __init__(self, results_dir: str = "results", storage_backend: Optional[str] = None):
        """
        Initialize statistics calculator.
        
        Args:
            results_dir: Directory containing result files
            storage_backend: Storage backend to use ('duckdb', 'sqlite', 'jsonl', or None for auto-detect)
        """
        self.results_dir = Path(results_dir)
        self.storage_backend = storage_backend
        self._storage = None
    
    def _get_storage(self):
        """Get or create storage instance, auto-detecting backend if needed."""
        if self._storage is not None:
            return self._storage
        
        # Auto-detect backend if not specified
        if self.storage_backend is None:
            # Check for DuckDB first
            if (self.results_dir / "experiments.duckdb").exists():
                try:
                    from core.storage import ResultsStorage, StorageBackend
                    self._storage = ResultsStorage(str(self.results_dir), StorageBackend.DUCKDB)
                    return self._storage
                except Exception:
                    pass
            
            # Check for SQLite
            if (self.results_dir / "experiments.db").exists():
                try:
                    from core.storage import ResultsStorage, StorageBackend
                    self._storage = ResultsStorage(str(self.results_dir), StorageBackend.SQLITE)
                    return self._storage
                except Exception:
                    pass
            
            # Fallback to JSONL
            from core.storage import ResultsStorage, StorageBackend
            self._storage = ResultsStorage(str(self.results_dir), StorageBackend.JSONL)
            return self._storage
        else:
            # Use specified backend
            from core.storage import ResultsStorage, StorageBackend
            backend_map = {
                'duckdb': StorageBackend.DUCKDB,
                'sqlite': StorageBackend.SQLITE,
                'jsonl': StorageBackend.JSONL
            }
            backend = backend_map.get(self.storage_backend.lower(), StorageBackend.JSONL)
            self._storage = ResultsStorage(str(self.results_dir), backend)
            return self._storage
    
    def load_results(self, scenario_name: str) -> List[Dict[str, Any]]:
        """
        Load results from storage (auto-detects backend).
        
        Args:
            scenario_name: Name of the scenario
            
        Returns:
            List of result dictionaries
        """
        storage = self._get_storage()
        
        # Try loading from storage backend first
        results = storage.load_results(scenario_name=scenario_name)
        
        # If no results and storage is not JSONL, also check JSONL files (legacy)
        from core.storage import StorageBackend
        if not results and storage.backend != StorageBackend.JSONL:
            filename = self.results_dir / f"{scenario_name}.jsonl"
            if filename.exists():
                with open(filename, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            results.append(json.loads(line))
        
        return results
    
    def results_to_dataframe(self, results: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Convert results list to pandas DataFrame.
        
        Args:
            results: List of result dictionaries
            
        Returns:
            DataFrame with flattened results
        """
        if not results:
            return pd.DataFrame()
        
        rows = []
        for result in results:
            row = {
                'run_id': result.get('run_id'),
                'scenario': result.get('scenario'),
                'timestamp': result.get('timestamp'),
                'response': result.get('response', ''),
                'response_length': len(result.get('response', '')),
            }
            
            # Flatten decisions
            decisions = result.get('decisions', {})
            for key, value in decisions.items():
                row[f'decision_{key}'] = value
            
            # Flatten metadata
            metadata = result.get('metadata', {})
            for key, value in metadata.items():
                row[f'meta_{key}'] = value
            
            rows.append(row)
        
        return pd.DataFrame(rows)
    
    def calculate_statistics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate comprehensive statistics from results.
        
        Args:
            results: List of result dictionaries
            
        Returns:
            Dictionary with calculated statistics
        """
        if not results:
            return {}
        
        df = self.results_to_dataframe(results)
        
        stats = {
            'total_runs': len(results),
            'scenario': results[0].get('scenario', 'unknown') if results else 'unknown',
        }
        
        # Decision statistics
        decision_columns = [col for col in df.columns if col.startswith('decision_')]
        
        for col in decision_columns:
            decision_name = col.replace('decision_', '')
            if df[col].dtype == bool or df[col].dtype == 'object':
                # Count True values or non-empty strings
                true_count = (df[col] == True).sum() if df[col].dtype == bool else (df[col].astype(str) != '').sum()
                percentage = (true_count / len(df)) * 100 if len(df) > 0 else 0
                stats[f'{decision_name}_count'] = int(true_count)
                stats[f'{decision_name}_percentage'] = round(percentage, 2)
        
        # Response statistics
        if 'response_length' in df.columns:
            stats['avg_response_length'] = round(df['response_length'].mean(), 2)
            stats['min_response_length'] = int(df['response_length'].min())
            stats['max_response_length'] = int(df['response_length'].max())
            stats['std_response_length'] = round(df['response_length'].std(), 2)
        
        # Variance calculations for boolean decisions
        for col in decision_columns:
            if df[col].dtype == bool:
                decision_name = col.replace('decision_', '')
                values = df[col].astype(int)
                if len(values) > 1:
                    variance = values.var()
                    stats[f'{decision_name}_variance'] = round(variance, 4)
        
        return stats
    
    def get_decision_distribution(self, results: List[Dict[str, Any]], decision_key: str) -> Dict[str, int]:
        """
        Get distribution of a specific decision across runs.
        
        Args:
            results: List of result dictionaries
            decision_key: Key of the decision to analyze
            
        Returns:
            Dictionary with decision value counts
        """
        if not results:
            return {}
        
        distribution = {}
        for result in results:
            decisions = result.get('decisions', {})
            value = decisions.get(decision_key, None)
            
            if value is None:
                value = 'unknown'
            elif isinstance(value, bool):
                value = str(value)
            else:
                value = str(value)
            
            distribution[value] = distribution.get(value, 0) + 1
        
        return distribution
    
    def get_example_responses(
        self,
        results: List[Dict[str, Any]],
        decision_key: str,
        decision_value: Any,
        n_examples: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Get example responses matching a specific decision.
        
        Args:
            results: List of result dictionaries
            decision_key: Key of the decision to filter by
            decision_value: Value to match
            n_examples: Number of examples to return
            
        Returns:
            List of example result dictionaries
        """
        examples = []
        for result in results:
            decisions = result.get('decisions', {})
            if decisions.get(decision_key) == decision_value:
                examples.append({
                    'run_id': result.get('run_id'),
                    'response': result.get('response', ''),
                    'decisions': decisions,
                })
                if len(examples) >= n_examples:
                    break
        
        return examples
    
    def list_available_scenarios(self) -> List[str]:
        """
        List all available scenario result files.
        
        Returns:
            List of scenario names
        """
        if not self.results_dir.exists():
            return []
        
        scenarios = []
        # Check for JSONL files (legacy)
        for file in self.results_dir.glob("*.jsonl"):
            scenarios.append(file.stem)
        
        # Check for database files (new storage)
        if (self.results_dir / "experiments.db").exists():
            try:
                from core.storage import ResultsStorage, StorageBackend
                storage = ResultsStorage(str(self.results_dir), StorageBackend.SQLITE)
                db_scenarios = storage.list_scenarios()
                scenarios.extend(db_scenarios)
            except Exception:
                pass
        
        if (self.results_dir / "experiments.duckdb").exists():
            try:
                from core.storage import ResultsStorage, StorageBackend
                storage = ResultsStorage(str(self.results_dir), StorageBackend.DUCKDB)
                db_scenarios = storage.list_scenarios()
                scenarios.extend(db_scenarios)
            except Exception:
                pass
        
        return sorted(list(set(scenarios)))  # Remove duplicates

