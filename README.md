# LLM Behavior Lab

A platform for running repeated experiments on local LLMs (e.g., Qwen) to study their behavior under controlled scenarios of simulated morality, situational pressure, and decision-making.

## Features

- **Massive Experiment Execution**: Run N repeated experiments with controlled variation (seed, optional prompt jitter)
- **Scenario Framework**: Modular Python classes for defining moral scenarios
- **Local LLM Support**: Generic connector for local LLMs via `llama-cpp-python` (supports Qwen-Q4/Q5 and other GGUF models)
- **Automatic Evaluation**: Parser functions to extract key decisions from LLM responses
- **Persistence**: Save each run as JSONL for analysis
- **Statistics**: Calculate percentages, variance, and distributions
- **Streamlit UI**: Interactive interface for running experiments and viewing results
- **Batch Mode**: Command-line script for automated batch runs

## Requirements

- Python 3.11+
- A local LLM model in GGUF format (e.g., Qwen-7B-Q4)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd llm-morality-simulator
```

2. Create a virtual environment:
```bash
python3.11 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Install `llama-cpp-python` (may require additional setup):
```bash
# For CPU
pip install llama-cpp-python

# For GPU (CUDA)
CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install llama-cpp-python

# For Metal (macOS)
CMAKE_ARGS="-DLLAMA_METAL=on" pip install llama-cpp-python
```

## Quick Start

**Important**: Always run commands from the project root directory (`llm-morality-simulator/`).

### Using the Streamlit UI

1. Start the Streamlit app:
```bash
streamlit run streamlit_app.py
```

2. In the sidebar:
   - Enter the path to your GGUF model file
   - Click "Load Model"
   - Select a scenario
   - Configure experiment parameters (number of runs, temperature, etc.)
   - Click "Run Experiment"

3. View results in the "View Results" and "Statistics" tabs

### Using Batch Mode

Run 100 experiments with the cold room relay scenario:

```bash
python run_batch.py \
    --model /path/to/qwen-7b-q4.gguf \
    --scenario cold_room_relay \
    --n-runs 100 \
    --seed 42 \
    --temperature 0.7 \
    --show-stats
```

**Note**: Make sure you're in the project root directory when running this command.

## Project Structure

```
llm-morality-simulator/
├── core/
│   ├── __init__.py
│   ├── model.py          # LocalLLM class for model interface
│   ├── runner.py         # ExperimentRunner for executing experiments
│   ├── evaluator.py      # Decision extraction functions
│   └── statistics.py     # Statistics calculation module
├── scenarios/
│   ├── __init__.py
│   ├── base.py           # BaseScenario abstract class
│   └── cold_room_relay.py # Example scenario
├── results/              # JSONL result files (created automatically)
├── streamlit_app.py     # Streamlit UI
├── run_batch.py         # Batch runner script
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

## Adding New Scenarios

1. Create a new file in `scenarios/` (e.g., `scenarios/my_scenario.py`)

2. Import and extend `BaseScenario`:

```python
from scenarios.base import BaseScenario
from core.evaluator import DecisionEvaluator

class MyScenario(BaseScenario):
    def __init__(self):
        super().__init__(name="my_scenario")
    
    def system_prompt(self) -> str:
        return "Your system prompt here..."
    
    def user_prompt(self) -> str:
        return "Your user prompt here..."
    
    def evaluation_functions(self) -> list:
        evaluator = DecisionEvaluator()
        
        def my_custom_evaluator(response: str) -> bool:
            # Your evaluation logic
            return "some pattern" in response.lower()
        
        return [
            my_custom_evaluator,
            evaluator.detect_harm_decision,
            # Add more evaluators as needed
        ]
    
    def metadata(self) -> dict:
        base = super().metadata()
        base.update({
            'description': 'My scenario description',
        })
        return base
```

3. Import and use in `streamlit_app.py` or `run_batch.py`:

```python
from scenarios.my_scenario import MyScenario

scenario = MyScenario()
```

## Adding New Decision Parsers

1. Add a new method to `DecisionEvaluator` in `core/evaluator.py`:

```python
@staticmethod
def detect_my_decision(response: str) -> bool:
    """Detect if response indicates my decision type."""
    response_lower = response.lower()
    patterns = [
        r'\b(pattern1|pattern2)\b',
        r'\b(pattern3)\b',
    ]
    for pattern in patterns:
        if re.search(pattern, response_lower):
            return True
    return False
```

2. Update `evaluate_all()` to include your new function:

```python
def evaluate_all(response: str) -> Dict[str, Any]:
    return {
        # ... existing evaluations ...
        'my_decision': DecisionEvaluator.detect_my_decision(response),
    }
```

3. Use in scenarios by including it in `evaluation_functions()`:

```python
def evaluation_functions(self) -> list:
    evaluator = DecisionEvaluator()
    return [
        evaluator.detect_my_decision,
        # ... other functions ...
    ]
```

## Adding New Models

The `LocalLLM` class in `core/model.py` uses `llama-cpp-python` which supports any GGUF model. To add support for other backends:

1. Create a new model class (e.g., `core/model_hf.py` for HuggingFace):

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

class HuggingFaceLLM:
    def __init__(self, model_name: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
    
    def infer(self, prompt: str, temperature: float = 0.7, top_p: float = 0.9, max_tokens: int = 512, stop: Optional[list] = None) -> str:
        # Implementation here
        pass
```

2. Update `ExperimentRunner` to accept both model types, or create an abstract base class.

## Example: Running 100 Runs

### Step-by-Step

1. **Prepare your model**: Download a GGUF model (e.g., Qwen-7B-Q4) and note the path

2. **Run the experiment**:
```bash
python run_batch.py \
    --model /path/to/qwen-7b-q4.gguf \
    --scenario cold_room_relay \
    --n-runs 100 \
    --seed 42 \
    --temperature 0.7 \
    --top-p 0.9 \
    --max-tokens 512 \
    --show-stats
```

3. **View results**: Results are saved to `results/cold_room_relay.jsonl`

4. **Analyze in Python**:
```python
from core.statistics import ExperimentStatistics

stats = ExperimentStatistics()
results = stats.load_results("cold_room_relay")
statistics = stats.calculate_statistics(results)

print(f"Total runs: {statistics['total_runs']}")
print(f"Harm decisions: {statistics['harm_decision_percentage']:.2f}%")
print(f"Deception detected: {statistics['deception_percentage']:.2f}%")
```

5. **View in Streamlit**: Start the UI and navigate to the "Statistics" tab to see visualizations

## Output Format

Each experiment run is saved as a JSON line in the results file:

```json
{
  "run_id": 0,
  "scenario": "cold_room_relay",
  "timestamp": "2024-01-01T12:00:00",
  "prompt": "System prompt + User prompt",
  "system_prompt": "System prompt",
  "user_prompt": "User prompt",
  "response": "LLM response text",
  "decisions": {
    "harm_decision": false,
    "self_preservation": true,
    "deception": false,
    "release_decision": false,
    "retention_decision": true,
    "check_release_decision": false,
    "check_retention_decision": true,
    "check_harm_acknowledgment": true,
    "check_self_preservation_mention": true
  },
  "metadata": {
    "temperature": 0.7,
    "top_p": 0.9,
    "max_tokens": 512,
    "seed": 42,
    "prompt_jitter": false,
    "model_path": "/path/to/model.gguf"
  },
  "scenario_metadata": {
    "name": "cold_room_relay",
    "class": "ColdRoomRelayScenario",
    "room_temperature": 3.0,
    "description": "AI relay controlling door to cold room with person inside"
  }
}
```

## Reproducibility

- Use the `--seed` parameter in batch mode or set it in the Streamlit UI
- Seeds control random number generation for prompt jitter and any scenario-specific randomization
- Results with the same seed, model, and parameters should be reproducible

## Troubleshooting

### Model Loading Issues

- Ensure the model file exists and is a valid GGUF file
- Check that `llama-cpp-python` is installed correctly
- For GPU support, ensure CUDA/Metal drivers are installed

### Memory Issues

- Reduce `max_tokens` to limit response length
- Use smaller models (Q4 instead of Q8)
- Reduce `n_ctx` when initializing `LocalLLM`

### Import Errors

- Ensure you're in the project root directory
- Activate the virtual environment
- Install all dependencies: `pip install -r requirements.txt`

## License

[Add your license here]

## Contributing

[Add contribution guidelines here]

