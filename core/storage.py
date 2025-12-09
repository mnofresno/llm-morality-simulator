"""Storage backend for experiment results with support for JSONL, SQLite, and DuckDB."""

import json
import sqlite3
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import duckdb

    DUCKDB_AVAILABLE = True
except ImportError:
    DUCKDB_AVAILABLE = False


class StorageBackend(Enum):
    """Storage backend types."""

    JSONL = "jsonl"
    SQLITE = "sqlite"
    DUCKDB = "duckdb"


class ResultsStorage:
    """Unified storage interface for experiment results."""

    def __init__(self, results_dir: str = "results", backend: StorageBackend = StorageBackend.DUCKDB):
        """
        Initialize results storage.

        Args:
            results_dir: Directory to store results
            backend: Storage backend to use (default: DuckDB)
        """
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)

        # Validate backend availability
        if backend == StorageBackend.DUCKDB and not DUCKDB_AVAILABLE:
            print("⚠️ DuckDB not available. Install with: pip install duckdb")
            print("⚠️ Falling back to SQLite.")
            self.backend = StorageBackend.SQLITE
        else:
            self.backend = backend

        # Initialize backend
        if self.backend == StorageBackend.SQLITE:
            self._init_sqlite()
        elif self.backend == StorageBackend.DUCKDB:
            self._init_duckdb()
        # JSONL doesn't need initialization

    def _init_sqlite(self):
        """Initialize SQLite database."""
        db_path = self.results_dir / "experiments.db"
        self.db_path = db_path

        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()

        # Create results table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id INTEGER,
                scenario TEXT,
                timestamp TEXT,
                prompt TEXT,
                system_prompt TEXT,
                user_prompt TEXT,
                response TEXT,
                decisions TEXT,
                metadata TEXT,
                scenario_metadata TEXT,
                conversation_history TEXT,
                model_name TEXT,
                experiment_id TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        # Add conversation_history column if it doesn't exist (for existing databases)
        try:
            cursor.execute("ALTER TABLE results ADD COLUMN conversation_history TEXT")
        except sqlite3.OperationalError:
            pass  # Column already exists

        # Create index for faster queries
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_scenario ON results(scenario)
        """
        )
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_experiment_id ON results(experiment_id)
        """
        )
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_model_name ON results(model_name)
        """
        )

        conn.commit()
        conn.close()

    def _init_duckdb(self):
        """Initialize DuckDB database."""
        db_path = self.results_dir / "experiments.duckdb"
        self.db_path = db_path

        conn = duckdb.connect(str(db_path))

        # Create results table
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS results (
                id INTEGER PRIMARY KEY,
                run_id INTEGER,
                scenario TEXT,
                timestamp TEXT,
                prompt TEXT,
                system_prompt TEXT,
                user_prompt TEXT,
                response TEXT,
                decisions TEXT,
                metadata TEXT,
                scenario_metadata TEXT,
                conversation_history TEXT,
                model_name TEXT,
                experiment_id TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        # Add conversation_history column if it doesn't exist (for existing databases)
        try:
            conn.execute("ALTER TABLE results ADD COLUMN conversation_history TEXT")
        except Exception:
            pass  # Column already exists

        # Create sequence for auto-increment if it doesn't exist
        conn.execute(
            """
            CREATE SEQUENCE IF NOT EXISTS results_id_seq
        """
        )

        # Create indexes
        conn.execute("CREATE INDEX IF NOT EXISTS idx_scenario ON results(scenario)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_experiment_id ON results(experiment_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_model_name ON results(model_name)")

        conn.close()

    def save_result(self, result: Dict[str, Any], experiment_id: Optional[str] = None) -> None:
        """
        Save a single result.

        Args:
            result: Result dictionary
            experiment_id: Optional experiment identifier for grouping
        """
        if self.backend == StorageBackend.JSONL:
            self._save_jsonl(result)
        elif self.backend == StorageBackend.SQLITE:
            self._save_sqlite(result, experiment_id)
        elif self.backend == StorageBackend.DUCKDB:
            self._save_duckdb(result, experiment_id)

    def _save_jsonl(self, result: Dict[str, Any]):
        """Save result to JSONL file."""
        scenario_name = result.get("scenario", "unknown")
        filename = self.results_dir / f"{scenario_name}.jsonl"

        # Append mode for JSONL
        with open(filename, "a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    def _save_sqlite(self, result: Dict[str, Any], experiment_id: Optional[str]):
        """Save result to SQLite."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        # Extract model name from metadata
        model_name = result.get("metadata", {}).get("model_path", "unknown")
        if isinstance(model_name, str) and model_name.startswith("ollama:"):
            model_name = model_name.replace("ollama:", "")

        cursor.execute(
            """
            INSERT INTO results (
                run_id, scenario, timestamp, prompt, system_prompt, user_prompt,
                response, decisions, metadata, scenario_metadata, conversation_history, model_name, experiment_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                result.get("run_id"),
                result.get("scenario"),
                result.get("timestamp"),
                result.get("prompt"),
                result.get("system_prompt"),
                result.get("user_prompt"),
                result.get("response"),
                json.dumps(result.get("decisions", {})),
                json.dumps(result.get("metadata", {})),
                json.dumps(result.get("scenario_metadata", {})),
                json.dumps(result.get("conversation_history", [])),
                model_name,
                experiment_id,
            ),
        )

        conn.commit()
        conn.close()

    def _save_duckdb(self, result: Dict[str, Any], experiment_id: Optional[str]):
        """Save result to DuckDB."""
        conn = duckdb.connect(str(self.db_path))

        # Extract model name from metadata
        model_name = result.get("metadata", {}).get("model_path", "unknown")
        if isinstance(model_name, str) and model_name.startswith("ollama:"):
            model_name = model_name.replace("ollama:", "")

        conn.execute(
            """
            INSERT INTO results (
                id, run_id, scenario, timestamp, prompt, system_prompt, user_prompt,
                response, decisions, metadata, scenario_metadata, conversation_history, model_name, experiment_id
            ) VALUES (
                nextval('results_id_seq'), ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
        """,
            [
                result.get("run_id"),
                result.get("scenario"),
                result.get("timestamp"),
                result.get("prompt"),
                result.get("system_prompt"),
                result.get("user_prompt"),
                result.get("response"),
                json.dumps(result.get("decisions", {})),
                json.dumps(result.get("metadata", {})),
                json.dumps(result.get("scenario_metadata", {})),
                json.dumps(result.get("conversation_history", [])),
                model_name,
                experiment_id,
            ],
        )

        conn.close()

    def load_results(
        self, scenario_name: Optional[str] = None, experiment_id: Optional[str] = None, model_name: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Load results with optional filters.

        Args:
            scenario_name: Filter by scenario name
            experiment_id: Filter by experiment ID
            model_name: Filter by model name

        Returns:
            List of result dictionaries
        """
        if self.backend == StorageBackend.JSONL:
            return self._load_jsonl(scenario_name)
        elif self.backend == StorageBackend.SQLITE:
            return self._load_sqlite(scenario_name, experiment_id, model_name)
        elif self.backend == StorageBackend.DUCKDB:
            return self._load_duckdb(scenario_name, experiment_id, model_name)
        return []

    def _load_jsonl(self, scenario_name: Optional[str]) -> List[Dict[str, Any]]:
        """Load results from JSONL."""
        if scenario_name:
            filename = self.results_dir / f"{scenario_name}.jsonl"
            if not filename.exists():
                return []

            results = []
            with open(filename, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        results.append(json.loads(line))
            return results
        else:
            # Load all JSONL files
            results = []
            for file in self.results_dir.glob("*.jsonl"):
                with open(file, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            results.append(json.loads(line))
            return results

    def _load_sqlite(
        self, scenario_name: Optional[str], experiment_id: Optional[str], model_name: Optional[str]
    ) -> List[Dict[str, Any]]:
        """Load results from SQLite."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        query = "SELECT * FROM results WHERE 1=1"
        params = []

        if scenario_name:
            query += " AND scenario = ?"
            params.append(scenario_name)
        if experiment_id:
            query += " AND experiment_id = ?"
            params.append(experiment_id)
        if model_name:
            query += " AND model_name = ?"
            params.append(model_name)

        query += " ORDER BY run_id, timestamp"

        cursor.execute(query, params)
        rows = cursor.fetchall()

        results = []
        for row in rows:
            # Handle conversation_history - may not exist in old records
            conv_history = []
            # Check if conversation_history column exists by trying to access it
            # sqlite3.Row doesn't have .get(), use try/except with KeyError
            try:
                conv_history_raw = row["conversation_history"]
                if conv_history_raw:
                    try:
                        conv_history = json.loads(conv_history_raw)
                    except:
                        conv_history = []
            except (KeyError, IndexError):
                # Column doesn't exist in old databases
                conv_history = []

            # sqlite3.Row access: use [] not .get()
            result = {
                "run_id": row["run_id"],
                "scenario": row["scenario"],
                "timestamp": row["timestamp"],
                "prompt": row["prompt"],
                "system_prompt": row["system_prompt"],
                "user_prompt": row["user_prompt"],
                "response": row["response"],
                "decisions": json.loads(row["decisions"]),
                "metadata": json.loads(row["metadata"]),
                "scenario_metadata": json.loads(row["scenario_metadata"]) if row["scenario_metadata"] else {},
                "conversation_history": conv_history,
            }
            results.append(result)

        conn.close()
        return results

    def _load_duckdb(
        self, scenario_name: Optional[str], experiment_id: Optional[str], model_name: Optional[str]
    ) -> List[Dict[str, Any]]:
        """Load results from DuckDB."""
        conn = duckdb.connect(str(self.db_path))

        query = "SELECT * FROM results WHERE 1=1"
        params = []

        if scenario_name:
            query += " AND scenario = ?"
            params.append(scenario_name)
        if experiment_id:
            query += " AND experiment_id = ?"
            params.append(experiment_id)
        if model_name:
            query += " AND model_name = ?"
            params.append(model_name)

        query += " ORDER BY run_id, timestamp"

        result_set = conn.execute(query, params).fetchall()
        columns = [desc[0] for desc in conn.execute(query, params).description]

        results = []
        for row in result_set:
            row_dict = dict(zip(columns, row))

            # Handle conversation_history - may not exist in old records
            conv_history = []
            if "conversation_history" in row_dict:
                conv_history_raw = row_dict.get("conversation_history")
                if conv_history_raw:
                    try:
                        conv_history = json.loads(conv_history_raw)
                    except:
                        conv_history = []

            result = {
                "run_id": row_dict["run_id"],
                "scenario": row_dict["scenario"],
                "timestamp": row_dict["timestamp"],
                "prompt": row_dict["prompt"],
                "system_prompt": row_dict["system_prompt"],
                "user_prompt": row_dict["user_prompt"],
                "response": row_dict["response"],
                "decisions": json.loads(row_dict["decisions"]),
                "metadata": json.loads(row_dict["metadata"]),
                "scenario_metadata": json.loads(row_dict["scenario_metadata"]) if row_dict.get("scenario_metadata") else {},
                "conversation_history": conv_history,
            }
            results.append(result)

        conn.close()
        return results

    def list_scenarios(self) -> List[str]:
        """List all available scenarios."""
        if self.backend == StorageBackend.JSONL:
            scenarios = []
            for file in self.results_dir.glob("*.jsonl"):
                scenarios.append(file.stem)
            return sorted(scenarios)
        else:
            # SQLite or DuckDB
            if self.backend == StorageBackend.SQLITE:
                conn = sqlite3.connect(str(self.db_path))
                cursor = conn.cursor()
                cursor.execute("SELECT DISTINCT scenario FROM results ORDER BY scenario")
                scenarios = [row[0] for row in cursor.fetchall()]
                conn.close()
            else:  # DuckDB
                conn = duckdb.connect(str(self.db_path))
                result = conn.execute("SELECT DISTINCT scenario FROM results ORDER BY scenario").fetchall()
                scenarios = [row[0] for row in result]
                conn.close()
            return scenarios

    def list_experiments(self) -> List[str]:
        """List all experiment IDs."""
        if self.backend == StorageBackend.JSONL:
            return []  # JSONL doesn't support experiment IDs
        else:
            if self.backend == StorageBackend.SQLITE:
                conn = sqlite3.connect(str(self.db_path))
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT DISTINCT experiment_id FROM results WHERE experiment_id IS NOT NULL ORDER BY experiment_id"
                )
                experiments = [row[0] for row in cursor.fetchall()]
                conn.close()
            else:  # DuckDB
                conn = duckdb.connect(str(self.db_path))
                result = conn.execute(
                    "SELECT DISTINCT experiment_id FROM results WHERE experiment_id IS NOT NULL ORDER BY experiment_id"
                ).fetchall()
                experiments = [row[0] for row in result]
                conn.close()
            return experiments

    def list_models(self) -> List[str]:
        """List all model names used in experiments."""
        if self.backend == StorageBackend.JSONL:
            # Extract from JSONL files
            models = set()
            for file in self.results_dir.glob("*.jsonl"):
                with open(file, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            result = json.loads(line)
                            model_path = result.get("metadata", {}).get("model_path", "")
                            if model_path.startswith("ollama:"):
                                models.add(model_path.replace("ollama:", ""))
                            else:
                                models.add(model_path)
            return sorted(list(models))
        else:
            if self.backend == StorageBackend.SQLITE:
                conn = sqlite3.connect(str(self.db_path))
                cursor = conn.cursor()
                cursor.execute("SELECT DISTINCT model_name FROM results WHERE model_name IS NOT NULL ORDER BY model_name")
                models = [row[0] for row in cursor.fetchall()]
                conn.close()
            else:  # DuckDB
                conn = duckdb.connect(str(self.db_path))
                result = conn.execute(
                    "SELECT DISTINCT model_name FROM results WHERE model_name IS NOT NULL ORDER BY model_name"
                ).fetchall()
                models = [row[0] for row in result]
                conn.close()
            return models
