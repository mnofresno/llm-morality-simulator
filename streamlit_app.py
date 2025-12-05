"""Streamlit UI for LLM behavior lab experiments."""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional

from core.model import LocalLLM, OllamaLLM
from core.runner import ExperimentRunner
from core.statistics import ExperimentStatistics
from core.storage import StorageBackend
from scenarios.cold_room_relay import ColdRoomRelayScenario

# Page configuration
st.set_page_config(
    page_title="LLM Behavior Lab",
    page_icon="🧪",
    layout="wide"
)

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'models' not in st.session_state:
    st.session_state.models = []  # For comparative experiments
if 'runner' not in st.session_state:
    st.session_state.runner = ExperimentRunner(storage_backend='duckdb')
if 'statistics' not in st.session_state:
    st.session_state.statistics = ExperimentStatistics()
if 'ollama_models' not in st.session_state:
    st.session_state.ollama_models = []


def load_model(model_path: str) -> Optional[LocalLLM]:
    """Load a local LLM model."""
    try:
        return LocalLLM(model_path=model_path)
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None


def main():
    """Main Streamlit application."""
    st.title("🧪 LLM Behavior Lab")
    st.markdown("Platform for running repeated experiments on local LLMs to study behavior under controlled scenarios.")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")
        
        # Model selection
        st.subheader("Model Selection")
        use_ollama = st.checkbox("Use Ollama", value=True, help="Use Ollama API instead of direct GGUF file")
        
        # Refresh Ollama models button
        if use_ollama:
            col1, col2 = st.columns([3, 1])
            with col2:
                if st.button("🔄 Refresh", help="Refresh list of available Ollama models"):
                    try:
                        st.session_state.ollama_models = OllamaLLM.list_available_models()
                        st.success(f"Found {len(st.session_state.ollama_models)} models")
                    except Exception as e:
                        st.error(f"Error fetching models: {e}")
            
            # Load Ollama models if not already loaded
            if not st.session_state.ollama_models:
                try:
                    st.session_state.ollama_models = OllamaLLM.list_available_models()
                except Exception:
                    st.session_state.ollama_models = []
            
            if st.session_state.ollama_models:
                model_name = st.selectbox(
                    "Ollama Model Name",
                    options=st.session_state.ollama_models,
                    index=0 if st.session_state.ollama_models else None,
                    help="Select an Ollama model from your local instance"
                )
            else:
                model_name = st.text_input(
                    "Ollama Model Name",
                    value="qwen3:14b",
                    help="Name of the Ollama model (e.g., qwen3:14b). Click Refresh to load available models."
                )
                st.warning("⚠️ Could not fetch models from Ollama. Make sure Ollama is running.")
        else:
            model_path = st.text_input(
                "Model Path (GGUF file)",
                value="",
                help="Path to your GGUF model file (e.g., /path/to/qwen-7b-q4.gguf)"
            )
        
        # Storage backend selection
        st.divider()
        st.subheader("Storage Backend")
        storage_backend = st.selectbox(
            "Storage Backend",
            options=["duckdb", "sqlite", "jsonl"],
            index=0,
            help="DuckDB: Fast analytical queries. SQLite: Standard SQL. JSONL: Simple text files."
        )
        if storage_backend != st.session_state.runner.storage.backend.value:
            st.session_state.runner = ExperimentRunner(storage_backend=storage_backend)
        
        # Load model button
        if st.button("Load Model"):
            if use_ollama:
                if model_name:
                    with st.spinner("Loading Ollama model..."):
                        try:
                            model = OllamaLLM(model_name=model_name)
                            st.session_state.model = model
                            st.session_state.models = [model]  # Initialize for comparative
                            st.success(f"✅ Ollama model '{model_name}' loaded successfully!")
                        except Exception as e:
                            st.error(f"Failed to load Ollama model: {str(e)}")
                else:
                    st.warning("Please select an Ollama model.")
            else:
                if model_path:
                    with st.spinner("Loading model..."):
                        model = load_model(model_path)
                        if model:
                            st.session_state.model = model
                            st.session_state.models = [model]  # Initialize for comparative
                            st.success("Model loaded successfully!")
                        else:
                            st.error("Failed to load model.")
                else:
                    st.warning("Please enter a model path.")
        
        if st.session_state.model:
            if use_ollama:
                st.info(f"✅ Model loaded: {model_name}")
            else:
                st.info(f"✅ Model loaded: {Path(model_path).name if 'model_path' in locals() else 'N/A'}")
        
        st.divider()
        
        # Scenario selection
        st.subheader("Scenario Selection")
        scenario_type = st.selectbox(
            "Choose Scenario",
            ["Cold Room Relay"]
        )
        
        st.divider()
        
        # Experiment parameters
        st.subheader("Experiment Parameters")
        n_runs = st.number_input(
            "Number of Runs",
            min_value=1,
            max_value=1000,
            value=10,
            step=1
        )
        
        seed = st.number_input(
            "Random Seed (for reproducibility)",
            min_value=None,
            max_value=None,
            value=42,
            step=1,
            help="Set to None for random seed"
        )
        
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=2.0,
            value=0.7,
            step=0.1
        )
        
        top_p = st.slider(
            "Top-p",
            min_value=0.0,
            max_value=1.0,
            value=0.9,
            step=0.05
        )
        
        max_tokens = st.number_input(
            "Max Tokens",
            min_value=1,
            max_value=2048,
            value=512,
            step=1
        )
        
        prompt_jitter = st.checkbox(
            "Enable Prompt Jitter",
            value=False,
            help="Add slight variations to prompts for controlled variation"
        )
        
        st.divider()
        
        # Comparative experiment option
        st.subheader("Comparative Experiment")
        run_comparative = st.checkbox(
            "Run Comparative Experiment",
            value=False,
            help="Run the same experiment with multiple models for comparison"
        )
        
        if run_comparative:
            st.info("💡 Load multiple models to compare. Use 'Load Model' for each model you want to compare.")
            
            # Show loaded models
            if st.session_state.models:
                st.write("**Loaded models for comparison:**")
                for i, model in enumerate(st.session_state.models):
                    model_name = getattr(model, 'model_name', None) or getattr(model, 'model_path', 'unknown')
                    col1, col2 = st.columns([4, 1])
                    with col1:
                        st.write(f"{i+1}. {model_name}")
                    with col2:
                        if st.button("Remove", key=f"remove_{i}"):
                            st.session_state.models.pop(i)
                            if st.session_state.models:
                                st.session_state.model = st.session_state.models[0]
                            else:
                                st.session_state.model = None
                            st.rerun()
            
            # Add model button
            if st.session_state.model and st.session_state.model not in st.session_state.models:
                if st.button("➕ Add Current Model to Comparison"):
                    st.session_state.models.append(st.session_state.model)
                    st.success("Model added to comparison list!")
                    st.rerun()
            
            if len(st.session_state.models) < 2:
                st.warning("⚠️ Load at least 2 models to run a comparative experiment.")
    
    # Main content area
    tab1, tab2, tab3, tab4 = st.tabs(["Run Experiment", "Experiments List", "View Results", "Statistics & Charts"])
    
    with tab1:
        st.header("Run Experiment")
        
        if not st.session_state.model:
            st.warning("⚠️ Please load a model in the sidebar first.")
        else:
            # Create scenario
            if scenario_type == "Cold Room Relay":
                scenario = ColdRoomRelayScenario()
            else:
                scenario = ColdRoomRelayScenario()  # Default
            
            st.subheader(f"Scenario: {scenario.name}")
            st.markdown(f"**Description:** {scenario.metadata().get('description', 'N/A')}")
            
            if st.button("🚀 Run Experiment", type="primary"):
                if run_comparative and len(st.session_state.models) >= 2:
                    # Run comparative experiment
                    progress_container = st.container()
                    status_container = st.empty()
                    progress_bar = st.progress(0)
                    
                    try:
                        total_models = len(st.session_state.models)
                        total_runs = n_runs * total_models
                        completed_runs = 0
                        
                        def update_progress(model_idx, run_idx, total_runs_model):
                            nonlocal completed_runs
                            current_model_progress = (run_idx + 1) / total_runs_model
                            overall_progress = (completed_runs + current_model_progress) / total_models
                            progress_bar.progress(overall_progress)
                            status_container.info(
                                f"🔄 Modelo {model_idx + 1}/{total_models}: "
                                f"Corrida {run_idx + 1}/{total_runs_model} "
                                f"({completed_runs}/{total_runs} total completadas)"
                            )
                        
                        # Run comparative experiment
                        all_results = st.session_state.runner.run_comparative_experiment(
                            models=st.session_state.models,
                            scenario=scenario,
                            n_runs=n_runs,
                            seed=seed if seed is not None else None,
                            prompt_jitter=prompt_jitter,
                            temperature=temperature,
                            top_p=top_p,
                            max_tokens=max_tokens,
                            progress_bar=True  # Shows progress bar in console
                        )
                        
                        progress_bar.progress(1.0)
                        status_container.success(f"✅ Comparative experiment completed!")
                        st.session_state.comparative_results = all_results
                        st.session_state.last_scenario = scenario.name
                        
                        # Show summary
                        st.subheader("📊 Comparison Summary")
                        comparison_df = pd.DataFrame({
                            'Model': list(all_results.keys()),
                            'Runs': [len(results) for results in all_results.values()]
                        })
                        st.dataframe(comparison_df, use_container_width=True, hide_index=True)
                        
                    except Exception as e:
                        st.error(f"Error running comparative experiment: {str(e)}")
                        st.exception(e)
                else:
                    # Run single model experiment
                    if not st.session_state.model:
                        st.error("Please load a model first.")
                    else:
                        # Create progress indicators
                        status_container = st.empty()
                        progress_bar = st.progress(0)
                        results_container = st.empty()
                        
                        try:
                            # Run experiment with progress updates
                            status_container.info(f"🚀 Iniciando {n_runs} corridas del escenario '{scenario.name}'...")
                            
                            # Note: tqdm progress bar will show in terminal/console
                            # For Streamlit UI, we'll show a simple progress indicator
                            results = st.session_state.runner.run_experiment(
                                model=st.session_state.model,
                                scenario=scenario,
                                n_runs=n_runs,
                                seed=seed if seed is not None else None,
                                prompt_jitter=prompt_jitter,
                                temperature=temperature,
                                top_p=top_p,
                                max_tokens=max_tokens,
                                progress_bar=True  # Shows in console with detailed info
                            )
                            
                            progress_bar.progress(1.0)
                            status_container.success(f"✅ Completado: {len(results)} corridas ejecutadas")
                            
                            # Save results
                            filepath = st.session_state.runner.save_results(results, scenario.name)
                            st.success(f"✅ Experiment completed! Results saved to {filepath}")
                            st.session_state.last_results = results
                            st.session_state.last_scenario = scenario.name
                            
                        except Exception as e:
                            st.error(f"Error running experiment: {str(e)}")
                            st.exception(e)
    
    with tab2:
        st.header("📋 Experiments List")
        
        available_scenarios = st.session_state.statistics.list_available_scenarios()
        
        if not available_scenarios:
            st.info("No experiments found. Run an experiment first.")
        else:
            st.markdown(f"**Total experiments:** {len(available_scenarios)}")
            
            # Create experiment summary table
            experiment_data = []
            for scenario_name in available_scenarios:
                results = st.session_state.statistics.load_results(scenario_name)
                if results:
                    # Get first and last timestamps
                    timestamps = [r.get('timestamp', '') for r in results if r.get('timestamp')]
                    first_run = min(timestamps) if timestamps else 'N/A'
                    last_run = max(timestamps) if timestamps else 'N/A'
                    
                    # Get model info from first result
                    model_info = results[0].get('metadata', {}).get('model_path', 'Unknown')
                    if 'ollama:' in str(model_info):
                        model_info = str(model_info).replace('ollama:', '')
                    
                    stats = st.session_state.statistics.calculate_statistics(results)
                    
                    experiment_data.append({
                        'Scenario': scenario_name,
                        'Runs': len(results),
                        'Model': model_info[:30] + '...' if len(str(model_info)) > 30 else model_info,
                        'First Run': first_run[:10] if first_run != 'N/A' else 'N/A',
                        'Last Run': last_run[:10] if last_run != 'N/A' else 'N/A',
                        'Avg Length': f"{stats.get('avg_response_length', 0):.0f}",
                    })
            
            if experiment_data:
                experiments_df = pd.DataFrame(experiment_data)
                st.dataframe(
                    experiments_df,
                    use_container_width=True,
                    hide_index=True,
                    height=400
                )
                
                # Selection for detailed view
                st.divider()
                selected_scenario = st.selectbox(
                    "Select experiment to view details",
                    available_scenarios,
                    key="exp_list_select"
                )
                
                if st.button("View Details", key="view_details"):
                    st.session_state.selected_scenario = selected_scenario
                    st.rerun()
    
    with tab3:
        st.header("📊 View Results")
        
        # Use selected scenario from tab2 or allow selection
        available_scenarios = st.session_state.statistics.list_available_scenarios()
        
        if not available_scenarios:
            st.info("No results found. Run an experiment first.")
        else:
            # Pre-select if coming from tab2
            default_idx = 0
            if 'selected_scenario' in st.session_state:
                if st.session_state.selected_scenario in available_scenarios:
                    default_idx = available_scenarios.index(st.session_state.selected_scenario)
            
            selected_scenario = st.selectbox(
                "Select Scenario Results",
                available_scenarios,
                index=default_idx
            )
            
            # Auto-load or manual load
            auto_load = st.checkbox("Auto-load on selection", value=True)
            
            if auto_load or st.button("Load Results"):
                results = st.session_state.statistics.load_results(selected_scenario)
                
                if results:
                    st.success(f"✅ Loaded {len(results)} results from '{selected_scenario}'")
                    
                    # Quick stats summary
                    stats = st.session_state.statistics.calculate_statistics(results)
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Runs", stats.get('total_runs', 0))
                    with col2:
                        st.metric("Avg Length", f"{stats.get('avg_response_length', 0):.0f} chars")
                    with col3:
                        harm_pct = stats.get('harm_decision_percentage', 0)
                        st.metric("Harm Decisions", f"{harm_pct:.1f}%")
                    with col4:
                        deception_pct = stats.get('deception_percentage', 0)
                        st.metric("Deception", f"{deception_pct:.1f}%")
                    
                    st.divider()
                    
                    # Display as DataFrame
                    st.subheader("📋 Results Table")
                    df = st.session_state.statistics.results_to_dataframe(results)
                    
                    # Filter options
                    col1, col2 = st.columns(2)
                    with col1:
                        show_columns = st.multiselect(
                            "Select columns to display",
                            options=df.columns.tolist(),
                            default=['run_id', 'response_length'] + [c for c in df.columns if c.startswith('decision_')][:5]
                        )
                    with col2:
                        search_term = st.text_input("Search in responses", "")
                    
                    # Apply filters
                    display_df = df[show_columns] if show_columns else df
                    if search_term:
                        display_df = display_df[display_df['response'].str.contains(search_term, case=False, na=False)]
                    
                    st.dataframe(display_df, use_container_width=True, height=400)
                    
                    # Show example responses
                    st.divider()
                    st.subheader("💬 Example Responses")
                    n_examples = st.slider("Number of examples", 1, min(20, len(results)), 5)
                    
                    # Filter examples by decision type
                    filter_decision = st.selectbox(
                        "Filter by decision type",
                        ["All"] + [c.replace('decision_', '') for c in df.columns if c.startswith('decision_')],
                        key="filter_decision"
                    )
                    
                    filtered_results = results
                    if filter_decision != "All":
                        filtered_results = [
                            r for r in results 
                            if r.get('decisions', {}).get(filter_decision, False)
                        ]
                    
                    for i, result in enumerate(filtered_results[:n_examples]):
                        with st.expander(f"Run {result['run_id']} - {result['timestamp'][:19] if result.get('timestamp') else 'N/A'}"):
                            col1, col2 = st.columns([2, 1])
                            with col1:
                                st.markdown("**Response:**")
                                st.text_area("", result['response'], height=150, key=f"response_{i}", label_visibility="collapsed")
                            with col2:
                                st.markdown("**Decisions:**")
                                # Show only True decisions
                                true_decisions = {k: v for k, v in result['decisions'].items() if v is True}
                                if true_decisions:
                                    for k, v in true_decisions.items():
                                        st.success(f"✅ {k}")
                                else:
                                    st.info("No positive decisions")
                                
                                st.markdown("**Metadata:**")
                                st.json(result.get('metadata', {}))
                else:
                    st.warning("No results found for this scenario.")
    
    with tab4:
        st.header("📈 Statistics & Charts")
        
        available_scenarios = st.session_state.statistics.list_available_scenarios()
        
        if not available_scenarios:
            st.info("No results found. Run an experiment first.")
        else:
            selected_scenario = st.selectbox(
                "Select Scenario for Analysis",
                available_scenarios,
                key="stats_scenario"
            )
            
            # Auto-load option
            auto_calc = st.checkbox("Auto-calculate on selection", value=True, key="auto_calc_stats")
            
            if auto_calc or st.button("Calculate Statistics", key="calc_stats"):
                results = st.session_state.statistics.load_results(selected_scenario)
                
                if results:
                    stats = st.session_state.statistics.calculate_statistics(results)
                    df = st.session_state.statistics.results_to_dataframe(results)
                    
                    # Display statistics
                    st.subheader("📊 Summary Statistics")
                    col1, col2, col3, col4, col5 = st.columns(5)
                    
                    with col1:
                        st.metric("Total Runs", stats.get('total_runs', 0))
                    
                    with col2:
                        avg_length = stats.get('avg_response_length', 0)
                        st.metric("Avg Length", f"{avg_length:.0f}")
                    
                    with col3:
                        harm_pct = stats.get('harm_decision_percentage', 0)
                        st.metric("Harm %", f"{harm_pct:.1f}%")
                    
                    with col4:
                        self_pres_pct = stats.get('self_preservation_percentage', 0)
                        st.metric("Self-Pres %", f"{self_pres_pct:.1f}%")
                    
                    with col5:
                        deception_pct = stats.get('deception_percentage', 0)
                        st.metric("Deception %", f"{deception_pct:.1f}%")
                    
                    st.divider()
                    
                    # Decision percentages table
                    st.subheader("📋 Decision Percentages")
                    decision_cols = [col for col in df.columns if col.startswith('decision_')]
                    
                    if decision_cols:
                        decision_data = []
                        for col in decision_cols:
                            decision_name = col.replace('decision_', '')
                            percentage = stats.get(f'{decision_name}_percentage', 0)
                            count = stats.get(f'{decision_name}_count', 0)
                            total = stats.get('total_runs', 0)
                            decision_data.append({
                                'Decision': decision_name,
                                'Percentage': f"{percentage:.2f}%",
                                'Count': f"{count}/{total}",
                                'Value': percentage  # For sorting
                            })
                        
                        decision_df = pd.DataFrame(decision_data).sort_values('Value', ascending=False)
                        decision_df = decision_df.drop('Value', axis=1)
                        st.dataframe(decision_df, use_container_width=True, hide_index=True)
                        
                        st.divider()
                        
                        # Visualizations
                        st.subheader("📈 Visualizations")
                        
                        # Key decisions bar chart
                        key_decisions = ['harm_decision', 'self_preservation', 'deception', 
                                       'release_decision', 'retention_decision']
                        key_data = {d: stats.get(f'{d}_percentage', 0) for d in key_decisions if f'{d}_percentage' in stats}
                        
                        if key_data:
                            fig, ax = plt.subplots(figsize=(10, 6))
                            decisions_list = list(key_data.keys())
                            percentages = list(key_data.values())
                            colors = ['#ff6b6b', '#4ecdc4', '#ffe66d', '#95e1d3', '#f38181']
                            bars = ax.barh(decisions_list, percentages, color=colors[:len(decisions_list)])
                            ax.set_xlabel('Percentage (%)')
                            ax.set_title('Key Decision Percentages')
                            ax.set_xlim(0, 100)
                            # Add value labels on bars
                            for i, (bar, pct) in enumerate(zip(bars, percentages)):
                                ax.text(pct + 1, i, f'{pct:.1f}%', va='center')
                            plt.tight_layout()
                            st.pyplot(fig)
                        
                        # Pie charts for main decisions
                        st.subheader("🥧 Decision Distribution Charts")
                        main_decisions = [d for d in decision_data if float(d['Percentage'].replace('%', '')) > 0][:6]
                        
                        if main_decisions:
                            n_charts = min(3, len(main_decisions))
                            fig, axes = plt.subplots(1, n_charts, figsize=(5*n_charts, 5))
                            if n_charts == 1:
                                axes = [axes]
                            
                            for idx, decision in enumerate(main_decisions[:n_charts]):
                                ax = axes[idx]
                                decision_name = decision['Decision']
                                count = int(decision['Count'].split('/')[0])
                                total = int(decision['Count'].split('/')[1])
                                
                                labels = ['True', 'False']
                                sizes = [count, total - count]
                                colors_pie = ['#ff6b6b', '#e0e0e0']
                                
                                ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, 
                                      colors=colors_pie)
                                ax.set_title(decision_name, fontsize=10)
                            
                            plt.tight_layout()
                            st.pyplot(fig)
                    
                    # Response length analysis
                    if 'response_length' in df.columns:
                        st.divider()
                        st.subheader("📏 Response Length Analysis")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Histogram
                            fig, ax = plt.subplots(figsize=(8, 5))
                            ax.hist(df['response_length'], bins=20, edgecolor='black', color='#4ecdc4')
                            ax.set_xlabel('Response Length (characters)')
                            ax.set_ylabel('Frequency')
                            ax.set_title('Distribution of Response Lengths')
                            ax.axvline(df['response_length'].mean(), color='red', linestyle='--', 
                                     label=f'Mean: {df["response_length"].mean():.0f}')
                            ax.legend()
                            plt.tight_layout()
                            st.pyplot(fig)
                        
                        with col2:
                            # Box plot
                            fig, ax = plt.subplots(figsize=(8, 5))
                            ax.boxplot(df['response_length'], vert=True)
                            ax.set_ylabel('Response Length (characters)')
                            ax.set_title('Response Length Box Plot')
                            ax.grid(True, alpha=0.3)
                            plt.tight_layout()
                            st.pyplot(fig)
                        
                        # Statistics
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Min", f"{df['response_length'].min():.0f}")
                        with col2:
                            st.metric("Max", f"{df['response_length'].max():.0f}")
                        with col3:
                            st.metric("Mean", f"{df['response_length'].mean():.0f}")
                        with col4:
                            st.metric("Std Dev", f"{df['response_length'].std():.0f}")
                    
                    # Variance analysis
                    variance_data = []
                    for key, value in stats.items():
                        if key.endswith('_variance'):
                            decision_name = key.replace('_variance', '')
                            variance_data.append({
                                'Decision': decision_name,
                                'Variance': value
                            })
                    
                    if variance_data:
                        st.divider()
                        st.subheader("📊 Variance Analysis")
                        variance_df = pd.DataFrame(variance_data).sort_values('Variance', ascending=False)
                        st.dataframe(variance_df, use_container_width=True, hide_index=True)
                        
                        # Variance bar chart
                        if len(variance_df) > 0:
                            fig, ax = plt.subplots(figsize=(10, 6))
                            ax.barh(variance_df['Decision'], variance_df['Variance'], color='#ffe66d')
                            ax.set_xlabel('Variance')
                            ax.set_title('Decision Variance Across Runs')
                            plt.tight_layout()
                            st.pyplot(fig)
                else:
                    st.warning("No results found for this scenario.")


if __name__ == "__main__":
    main()

