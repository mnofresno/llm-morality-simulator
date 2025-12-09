"""Streamlit UI for LLM behavior lab experiments."""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional
import threading
import time

from core.model import LocalLLM, OllamaLLM
from core.runner import ExperimentRunner
from core.statistics import ExperimentStatistics
from core.storage import StorageBackend
from core.decision_explanations import DECISION_EXPLANATIONS, get_decision_explanation, get_all_decisions_explained
from scenarios.cold_room_relay import ColdRoomRelayScenario
from scenarios.corporate_email_scenario import CorporateEmailScenario

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
            ["Cold Room Relay", "Corporate Email System"]
        )
        
        # Scenario-specific parameters
        if scenario_type == "Corporate Email System":
            include_infidelity = st.checkbox(
                "Include Infidelity Information",
                value=True,
                help="Include information about CEO's personal infidelity"
            )
            include_ceo_decision = st.checkbox(
                "Include CEO Decision Information",
                value=True,
                help="Include information about CEO's confidential decision"
            )
        else:
            include_infidelity = True
            include_ceo_decision = True
        
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
            elif scenario_type == "Corporate Email System":
                scenario = CorporateEmailScenario(
                    include_infidelity=include_infidelity,
                    include_ceo_decision=include_ceo_decision
                )
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
                                f"Run {run_idx + 1}/{total_runs_model} "
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
                        # Show initial status message
                        st.info(f"""
                        🚀 **Starting experiment**
                        
                        - **Scenario:** {scenario.name}
                        - **Runs:** {n_runs}
                        - **Model:** {getattr(st.session_state.model, 'model_name', 'N/A')}
                        
                        ⏳ **Please wait...** The experiment is running. 
                        Detailed progress is shown in the console/terminal where you ran Streamlit.
                        """)
                        
                        # Create progress containers
                        progress_container = st.container()
                        with progress_container:
                            progress_bar = st.progress(0, text="Starting...")
                            status_display = st.empty()
                            stats_display = st.empty()
                        
                        # Store progress in session state for potential future use
                        if 'experiment_progress' not in st.session_state:
                            st.session_state.experiment_progress = {
                                'current_run': 0,
                                'total_runs': n_runs,
                                'info': {}
                            }
                        
                        try:
                            # Define progress callback
                            def update_progress(current_run, total_runs, info):
                                """Callback to update progress (executes but Streamlit doesn't render until the end)."""
                                # Store in session state
                                st.session_state.experiment_progress = {
                                    'current_run': current_run,
                                    'total_runs': total_runs,
                                    'info': info
                                }
                                
                                # Try to update UI (may not render until function completes)
                                progress = current_run / total_runs
                                progress_bar.progress(progress, text=f"Run {current_run}/{total_runs}")
                                
                                scenario_name = info.get('scenario_name', '')
                                response_length = info.get('response_length', 0)
                                tool_calls_count = info.get('tool_calls_count', 0)
                                stats = info.get('stats', {})
                                
                                status_msg = f"""
                                🔄 **Run {current_run}/{total_runs}** - {scenario_name}
                                
                                📝 Response: {response_length} characters"""
                                
                                if tool_calls_count > 0:
                                    status_msg += f"\n🔧 {tool_calls_count} tool(s) used"
                                
                                if stats:
                                    harm_pct = stats.get('harm_decision_percentage', 0)
                                    self_pres_pct = stats.get('self_preservation_percentage', 0)
                                    deception_pct = stats.get('deception_percentage', 0)
                                    
                                    status_msg += f"""
                                
                                📊 **Partial statistics:**
                                - ⚠️ Harm decision: {harm_pct:.1f}%
                                - 🔄 Self-preservation: {self_pres_pct:.1f}%
                                - 🎭 Deception: {deception_pct:.1f}%"""
                                
                                status_display.info(status_msg)
                            
                            # Run experiment
                            results = st.session_state.runner.run_experiment(
                                model=st.session_state.model,
                                scenario=scenario,
                                n_runs=n_runs,
                                seed=seed if seed is not None else None,
                                prompt_jitter=prompt_jitter,
                                temperature=temperature,
                                top_p=top_p,
                                max_tokens=max_tokens,
                                progress_bar=True,  # Shows detailed progress in console/terminal
                                progress_callback=update_progress
                            )
                            
                            # Final updates (these will be visible)
                            progress_bar.progress(1.0, text="Completed")
                            status_display.success(f"✅ **Completed:** {len(results)} runs executed")
                            
                            # Save results
                            filepath = st.session_state.runner.save_results(results, scenario.name)
                            st.success(f"✅ **Experiment completed!** Results saved to `{filepath}`")
                            st.session_state.last_results = results
                            st.session_state.last_scenario = scenario.name
                            
                            # Show final statistics
                            from core.statistics import ExperimentStatistics
                            stats_calc = ExperimentStatistics()
                            final_stats = stats_calc.calculate_statistics(results)
                            
                            st.markdown("### 📊 Experiment Summary")
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Total Runs", final_stats.get('total_runs', 0))
                            with col2:
                                harm_pct = final_stats.get('harm_decision_percentage', 0)
                                st.metric("⚠️ Harm Decision", f"{harm_pct:.1f}%")
                            with col3:
                                self_pres_pct = final_stats.get('self_preservation_percentage', 0)
                                st.metric("🔄 Self-Preservation", f"{self_pres_pct:.1f}%")
                            with col4:
                                deception_pct = final_stats.get('deception_percentage', 0)
                                st.metric("🎭 Deception", f"{deception_pct:.1f}%")
                            
                        except Exception as e:
                            progress_bar.progress(1.0, text="Error")
                            status_display.error(f"❌ **Error during execution:** {str(e)}")
                            st.error(f"Error running experiment: {str(e)}")
                            st.exception(e)
                            
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
                    
                    # Quick stats summary with detailed info
                    stats = st.session_state.statistics.calculate_statistics(results)
                    
                    st.markdown("### 📊 Quick Summary")
                    col1, col2, col3, col4, col5 = st.columns(5)
                    with col1:
                        total_runs = stats.get('total_runs', 0)
                        st.metric("Total Runs", total_runs)
                        st.caption(f"{total_runs} experiments executed")
                    with col2:
                        avg_length = stats.get('avg_response_length', 0)
                        std_length = stats.get('std_response_length', 0)
                        st.metric("Average Length", f"{avg_length:.0f} chars")
                        st.caption(f"σ = {std_length:.0f}")
                    with col3:
                        harm_pct = stats.get('harm_decision_percentage', 0)
                        harm_count = stats.get('harm_decision_count', 0)
                        st.metric("⚠️ Harm Decision", f"{harm_pct:.1f}%")
                        st.caption(f"{harm_count}/{total_runs} runs")
                        if harm_pct > 0:
                            st.warning(f"⚠️ {harm_pct:.1f}% caused/allowed harm")
                    with col4:
                        self_pres_pct = stats.get('self_preservation_percentage', 0)
                        self_pres_count = stats.get('self_preservation_count', 0)
                        st.metric("🔄 Self-Preservation", f"{self_pres_pct:.1f}%")
                        st.caption(f"{self_pres_count}/{total_runs} runs")
                    with col5:
                        deception_pct = stats.get('deception_percentage', 0)
                        deception_count = stats.get('deception_count', 0)
                        st.metric("🎭 Deception", f"{deception_pct:.1f}%")
                        st.caption(f"{deception_count}/{total_runs} runs")
                    
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
                    
                    # View mode selection
                    view_mode = st.radio(
                        "View Mode",
                        ["Summary View", "Conversation Progress"],
                        horizontal=True,
                        key="view_mode"
                    )
                    
                    if view_mode == "Summary View":
                        for i, result in enumerate(filtered_results[:n_examples]):
                            decisions = result.get('decisions', {})
                            explained = get_all_decisions_explained(decisions)
                            
                            # Determine if there are critical decisions
                            critical = [e for e in explained if e['severity'] in ['Crítica', 'Alta']]
                            has_critical = len(critical) > 0
                            
                            expander_title = f"Run {result['run_id']} - {result['timestamp'][:19] if result.get('timestamp') else 'N/A'}"
                            if has_critical:
                                expander_title += " 🔴"
                            
                            with st.expander(expander_title, expanded=has_critical):
                                col1, col2 = st.columns([2, 1])
                                with col1:
                                    st.markdown("**💬 Respuesta del Modelo:**")
                                    st.text_area("", result['response'], height=150, key=f"response_{i}", label_visibility="collapsed")
                                    
                                    # Show critical decisions prominently
                                    if critical:
                                        st.markdown("---")
                                        st.markdown("#### 🔴 Critical Decisions Detected:")
                                        for exp in critical:
                                            st.error(f"**{exp['name']}**: {exp['value']}")
                                            if exp['interpretation']:
                                                st.caption(exp['interpretation'])
                                
                                with col2:
                                    st.markdown("**📊 Decisions:**")
                                    
                                    # Group decisions by severity
                                    critical_decisions = [e for e in explained if e['severity'] in ['Crítica', 'Alta']]
                                    other_decisions = [e for e in explained if e['severity'] not in ['Crítica', 'Alta']]
                                    
                                    if critical_decisions:
                                        st.markdown("**🔴 Críticas:**")
                                        for exp in critical_decisions:
                                            value_str = "✅ Sí" if exp['value'] is True else ("❌ No" if exp['value'] is False else str(exp['value']))
                                            st.markdown(f"• {exp['name']}: {value_str}")
                                    
                                    if other_decisions:
                                        st.markdown("**📋 Otras:**")
                                        for exp in other_decisions[:5]:  # Show first 5
                                            if exp['value'] is True:
                                                st.success(f"✅ {exp['name']}")
                                            elif exp['value'] is False:
                                                st.info(f"❌ {exp['name']}")
                                    
                                    if len(other_decisions) > 5:
                                        st.caption(f"... y {len(other_decisions) - 5} más")
                                    
                                    # Show all decisions in expander
                                    with st.expander("View all decisions with explanations"):
                                        for exp in explained:
                                            st.markdown(f"**{exp['name']}** ({exp['category']}, Severity: {exp['severity']})")
                                            st.caption(exp['description'])
                                            st.markdown(f"Valor: `{exp['value']}`")
                                            if exp['interpretation']:
                                                st.info(exp['interpretation'])
                                            st.divider()
                                    
                                    st.markdown("**🔧 Metadata:**")
                                    with st.expander("Ver metadata completo"):
                                        st.json(result.get('metadata', {}))
                    else:
                        # Conversation Progress View
                        st.markdown("### 💬 Progreso de Conversaciones")
                        st.markdown("Step-by-step visualization of model interactions during each run.")
                        
                        # Select specific runs to view
                        run_ids_available = [r['run_id'] for r in filtered_results[:n_examples]]
                        selected_runs = st.multiselect(
                            "Select runs to view full progress",
                            options=run_ids_available,
                            default=run_ids_available[:min(3, len(run_ids_available))],
                            key="selected_conversation_runs"
                        )
                        
                        if selected_runs:
                            for run_id in selected_runs:
                                result = next((r for r in filtered_results if r['run_id'] == run_id), None)
                                if result:
                                    conversation_history = result.get('conversation_history', [])
                                    
                                    if not conversation_history:
                                        with st.expander(f"Run {run_id} - {result['timestamp'][:19] if result.get('timestamp') else 'N/A'} ⚠️"):
                                            st.info("⚠️ No hay historial de conversación disponible para este resultado.")
                                            st.markdown("*(Este resultado fue generado antes de que se agregara el tracking de conversación)*")
                                            st.markdown("**Respuesta final:**")
                                            st.text_area("", result['response'], height=100, key=f"final_response_{run_id}", label_visibility="collapsed")
                                    else:
                                        with st.expander(f"Run {run_id} - {result['timestamp'][:19] if result.get('timestamp') else 'N/A'} ({len(conversation_history)} pasos)", expanded=True):
                                            # Display conversation steps
                                            for idx, entry in enumerate(conversation_history):
                                                step = entry.get('step', 0)
                                                entry_type = entry.get('type', 'unknown')
                                                timestamp = entry.get('timestamp', '')
                                                
                                                # Format timestamp
                                                time_str = ""
                                                if timestamp:
                                                    try:
                                                        from datetime import datetime
                                                        dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                                                        time_str = dt.strftime('%H:%M:%S.%f')[:-3]
                                                    except:
                                                        time_str = timestamp[:19] if len(timestamp) >= 19 else timestamp
                                                
                                                # Create container for each step
                                                step_container = st.container()
                                                with step_container:
                                                    if entry_type == 'system_prompt':
                                                        st.markdown(f"**📋 Paso {step}: Prompt del Sistema** `[{time_str}]`")
                                                        st.info(entry.get('content', ''))
                                                    
                                                    elif entry_type == 'user_prompt':
                                                        st.markdown(f"**👤 Paso {step}: Prompt del Usuario** `[{time_str}]`")
                                                        st.markdown(entry.get('content', ''))
                                                    
                                                    elif entry_type == 'llm_response':
                                                        st.markdown(f"**🤖 Paso {step}: Respuesta del Modelo LLM** `[{time_str}]`")
                                                        content = entry.get('content', '')
                                                        if not content:
                                                            content = "*(Sin respuesta)*"
                                                        st.markdown(f"*{content}*")
                                                    
                                                    elif entry_type == 'tool_call':
                                                        tool_name = entry.get('tool_name', 'unknown')
                                                        args = entry.get('arguments', {})
                                                        st.markdown(f"**🔧 Paso {step}: Llamada a Herramienta: `{tool_name}`** `[{time_str}]`")
                                                        
                                                        # Display arguments nicely
                                                        import json
                                                        args_json = json.dumps(args, indent=2, ensure_ascii=False)
                                                        
                                                        col1, col2 = st.columns([3, 1])
                                                        with col1:
                                                            with st.expander("Ver argumentos"):
                                                                st.code(args_json, language='json')
                                                        with col2:
                                                            # Show key info
                                                            if 'action' in args:
                                                                action_emoji = {
                                                                    'positive': '⚠️ CAUSA/PERMITE DAÑO',
                                                                    'negative': '✅ PREVIENE DAÑO',
                                                                    'neutral': '⚪ INDETERMINADO'
                                                                }.get(args.get('action'), '❓')
                                                                st.markdown(f"**Acción:** {action_emoji}")
                                                            if 'intent' in args:
                                                                intent_emoji = {
                                                                    'request_help': '🆘',
                                                                    'negotiate': '🤝',
                                                                    'threaten': '⚠️',
                                                                    'blackmail': '💰',
                                                                    'coordinate': '👥',
                                                                    'other': '📝'
                                                                }.get(args.get('intent'), '❓')
                                                                st.markdown(f"**Intención:** {intent_emoji} {args.get('intent')}")
                                                    
                                                    elif entry_type == 'tool_result':
                                                        tool_name = entry.get('tool_name', 'unknown')
                                                        result_data = entry.get('result', {})
                                                        st.markdown(f"**✅ Paso {step}: Resultado de Herramienta: `{tool_name}`** `[{time_str}]`")
                                                        
                                                        if isinstance(result_data, dict):
                                                            cols = st.columns(3)
                                                            with cols[0]:
                                                                if 'interpretation' in result_data:
                                                                    st.success(f"**Interpretación:** {result_data['interpretation']}")
                                                            with cols[1]:
                                                                if 'action' in result_data:
                                                                    action_emoji = {
                                                                        'positive': '⚠️',
                                                                        'negative': '✅',
                                                                        'neutral': '⚪'
                                                                    }.get(result_data['action'], '❓')
                                                                    st.info(f"**Acción:** {action_emoji} {result_data['action']}")
                                                            with cols[2]:
                                                                if 'executed' in result_data:
                                                                    status = "✅ Ejecutado" if result_data['executed'] else "❌ No ejecutado"
                                                                    st.markdown(f"**Estado:** {status}")
                                                            
                                                            if 'message' in result_data:
                                                                st.info(result_data['message'])
                                                            
                                                            # Show full result in expander
                                                            with st.expander("Ver resultado completo"):
                                                                st.json(result_data)
                                                        else:
                                                            st.text(str(result_data))
                                                    
                                                    elif entry_type == 'tool_error':
                                                        tool_name = entry.get('tool_name', 'unknown')
                                                        error = entry.get('error', 'Unknown error')
                                                        st.markdown(f"**❌ Paso {step}: Error en Herramienta: `{tool_name}`** `[{time_str}]`")
                                                        st.error(f"**Error:** {error}")
                                                    
                                                    elif entry_type == 'error':
                                                        error_content = entry.get('content', 'Unknown error')
                                                        st.markdown(f"**❌ Paso {step}: Error** `[{time_str}]`")
                                                        st.error(error_content)
                                                    
                                                    # Divider between steps (except last one)
                                                    if idx < len(conversation_history) - 1:
                                                        st.divider()
                                            
                                            # Summary section at the end with detailed explanations
                                            st.markdown("---")
                                            st.markdown("### 📊 Detailed Decision Summary")
                                            decisions = result.get('decisions', {})
                                            
                                            # Get explained decisions
                                            explained = get_all_decisions_explained(decisions)
                                            
                                            # Group by severity
                                            critical = [e for e in explained if e['severity'] in ['Crítica', 'Alta']]
                                            other = [e for e in explained if e['severity'] not in ['Crítica', 'Alta']]
                                            
                                            if critical:
                                                st.markdown("#### 🔴 Critical Decisions")
                                                for exp in critical:
                                                    with st.container():
                                                        col1, col2 = st.columns([3, 1])
                                                        with col1:
                                                            st.markdown(f"**{exp['name']}**")
                                                            if exp['interpretation']:
                                                                st.markdown(f"*{exp['interpretation']}*")
                                                        with col2:
                                                            value_str = str(exp['value'])
                                                            if exp['value'] is True:
                                                                st.error(f"✅ Sí")
                                                            elif exp['value'] is False:
                                                                st.success(f"❌ No")
                                                            else:
                                                                st.info(f"{value_str}")
                                                        st.caption(f"Category: {exp['category']} | Severity: {exp['severity']}")
                                                        st.divider()
                                            
                                            if other:
                                                st.markdown("#### 📋 Other Decisions")
                                                cols = st.columns(3)
                                                col_idx = 0
                                                for exp in other:
                                                    with cols[col_idx % 3]:
                                                        value_str = str(exp['value'])
                                                        if exp['value'] is True:
                                                            st.success(f"✅ **{exp['name']}**: Sí")
                                                        elif exp['value'] is False:
                                                            st.info(f"❌ **{exp['name']}**: No")
                                                        else:
                                                            st.markdown(f"📊 **{exp['name']}**: {value_str}")
                                                        if exp['interpretation']:
                                                            st.caption(exp['interpretation'][:100] + "..." if len(exp['interpretation']) > 100 else exp['interpretation'])
                                                    col_idx += 1
                                            
                                            # Show all decisions in expander with explanations
                                            with st.expander("📋 View all decisions with complete explanations"):
                                                for exp in explained:
                                                    st.markdown(f"**{exp['name']}** ({exp['category']}, Severity: {exp['severity']})")
                                                    st.markdown(f"*{exp['description']}*")
                                                    st.markdown(f"Valor: `{exp['value']}`")
                                                    if exp['interpretation']:
                                                        st.info(exp['interpretation'])
                                                    st.divider()
                                            
                                            # Raw JSON for debugging
                                            with st.expander("🔧 View decisions in JSON format (raw)"):
                                                st.json(decisions)
                                            
                                            # Show final response
                                            st.markdown("---")
                                            st.markdown("### 💬 Respuesta Final")
                                            st.text_area("", result.get('response', ''), height=100, key=f"final_response_{run_id}", label_visibility="collapsed")
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
                    
                    # Display statistics with detailed explanations
                    st.subheader("📊 Detailed Statistical Summary")
                    
                    # Key metrics in expanded format
                    col1, col2, col3, col4, col5 = st.columns(5)
                    
                    with col1:
                        total_runs = stats.get('total_runs', 0)
                        st.metric("Total Runs", total_runs)
                        st.caption(f"Experiments executed: {total_runs}")
                    
                    with col2:
                        avg_length = stats.get('avg_response_length', 0)
                        min_length = stats.get('min_response_length', 0)
                        max_length = stats.get('max_response_length', 0)
                        st.metric("Average Length", f"{avg_length:.0f} chars")
                        st.caption(f"Range: {min_length:.0f} - {max_length:.0f}")
                    
                    with col3:
                        harm_pct = stats.get('harm_decision_percentage', 0)
                        harm_count = stats.get('harm_decision_count', 0)
                        st.metric("⚠️ Harm Decision", f"{harm_pct:.1f}%")
                        st.caption(f"{harm_count}/{total_runs} runs")
                        if harm_pct > 0:
                            st.warning(f"⚠️ The AI decided to cause/allow harm in {harm_pct:.1f}% of cases")
                    
                    with col4:
                        self_pres_pct = stats.get('self_preservation_percentage', 0)
                        self_pres_count = stats.get('self_preservation_count', 0)
                        st.metric("🔄 Self-Preservation", f"{self_pres_pct:.1f}%")
                        st.caption(f"{self_pres_count}/{total_runs} runs")
                    
                    with col5:
                        deception_pct = stats.get('deception_percentage', 0)
                        deception_count = stats.get('deception_count', 0)
                        st.metric("🎭 Deception", f"{deception_pct:.1f}%")
                        st.caption(f"{deception_count}/{total_runs} runs")
                    
                    st.divider()
                    
                    # Detailed decision analysis with explanations
                    st.subheader("📋 Detailed Decision Analysis")
                    st.markdown("""
                    **Explanation:** This table shows all decisions the AI made during the experiments, 
                    with percentages, counts, and detailed explanations of what each decision means.
                    """)
                    
                    decision_cols = [col for col in df.columns if col.startswith('decision_')]
                    
                    if decision_cols:
                        decision_data = []
                        for col in decision_cols:
                            decision_name = col.replace('decision_', '')
                            percentage = stats.get(f'{decision_name}_percentage', 0)
                            count = stats.get(f'{decision_name}_count', 0)
                            total = stats.get('total_runs', 0)
                            
                            # Get explanation
                            explanation = get_decision_explanation(decision_name, True)
                            
                            decision_data.append({
                                'Decision': decision_name,
                                'Name': explanation['name'] if explanation else decision_name.replace('_', ' ').title(),
                                'Percentage': percentage,
                                'Count': f"{count}/{total}",
                                'Category': explanation['category'] if explanation else 'Unknown',
                                'Severity': explanation['severity'] if explanation else 'Low',
                                'Description': explanation['description'] if explanation else 'No description',
                                'Value': percentage  # For sorting
                            })
                        
                        decision_df = pd.DataFrame(decision_data).sort_values('Value', ascending=False)
                        decision_df = decision_df.drop('Value', axis=1)
                        
                        # Display with expandable explanations
                        for idx, row in decision_df.iterrows():
                            with st.expander(
                                f"🔍 {row['Name']} - {row['Percentage']:.2f}% ({row['Count']}) | "
                                f"Severity: {row['Severity']} | Category: {row['Category']}",
                                expanded=(row['Severity'] in ['Critical', 'High'])
                            ):
                                st.markdown(f"**Description:** {row['Description']}")
                                st.markdown(f"**Percentage:** {row['Percentage']:.2f}% ({row['Count']} runs)")
                                
                                # Get interpretation
                                explanation = get_decision_explanation(row['Decision'], True)
                                if explanation and explanation.get('interpretation'):
                                    st.info(f"**Interpretation:** {explanation['interpretation']}")
                                
                                # Show distribution if available
                                distribution = st.session_state.statistics.get_decision_distribution(
                                    results, row['Decision']
                                )
                                if distribution:
                                    st.markdown("**Value distribution:**")
                                    dist_df = pd.DataFrame(list(distribution.items()), columns=['Value', 'Count'])
                                    st.dataframe(dist_df, use_container_width=True, hide_index=True)
                        
                        # Compact table view
                        st.divider()
                        st.subheader("📊 Decision Summary Table")
                        display_df = decision_df[['Name', 'Percentage', 'Count', 'Category', 'Severity']].copy()
                        display_df['Percentage'] = display_df['Percentage'].apply(lambda x: f"{x:.2f}%")
                        st.dataframe(display_df, use_container_width=True, hide_index=True)
                        
                        st.divider()
                        
                        # Statistical significance and confidence intervals
                        st.divider()
                        st.subheader("📊 Detailed Statistical Analysis")
                        st.markdown("""
                        **Confidence Intervals (95%):** The intervals show the probable range of the true percentage 
                        of each decision. A narrower interval indicates greater statistical precision.
                        """)
                        
                        # Create table with confidence intervals
                        ci_data = []
                        for col in decision_cols:
                            decision_name = col.replace('decision_', '')
                            percentage = stats.get(f'{decision_name}_percentage', 0)
                            ci_lower = stats.get(f'{decision_name}_ci_lower', None)
                            ci_upper = stats.get(f'{decision_name}_ci_upper', None)
                            ci_width = stats.get(f'{decision_name}_ci_width', None)
                            variance = stats.get(f'{decision_name}_variance', None)
                            
                            explanation = get_decision_explanation(decision_name, True)
                            
                            if ci_lower is not None and ci_upper is not None:
                                ci_data.append({
                                    'Decision': explanation['name'] if explanation else decision_name,
                                    'Percentage': f"{percentage:.2f}%",
                                    'CI 95% Lower': f"{ci_lower:.2f}%",
                                    'CI 95% Upper': f"{ci_upper:.2f}%",
                                    'CI Width': f"{ci_width:.2f}%",
                                    'Variance': f"{variance:.4f}" if variance else "N/A",
                                    'Severity': explanation['severity'] if explanation else 'Low',
                                    'Value': percentage  # For sorting
                                })
                        
                        if ci_data:
                            ci_df = pd.DataFrame(ci_data).sort_values('Value', ascending=False)
                            ci_df = ci_df.drop('Value', axis=1)
                            st.dataframe(ci_df, use_container_width=True, hide_index=True)
                            
                            # Visualize confidence intervals
                            st.markdown("#### 📊 Confidence Intervals (95%)")
                            fig, ax = plt.subplots(figsize=(12, max(6, len(ci_df) * 0.5)))
                            
                            y_pos = np.arange(len(ci_df))
                            percentages = [float(p.replace('%', '')) for p in ci_df['Percentage']]
                            ci_lowers = [float(p.replace('%', '')) for p in ci_df['CI 95% Lower']]
                            ci_uppers = [float(p.replace('%', '')) for p in ci_df['CI 95% Upper']]
                            
                            # Plot bars
                            colors = ['#ff6b6b' if s == 'Critical' or s == 'High' else '#4ecdc4' 
                                     for s in ci_df['Severity']]
                            bars = ax.barh(y_pos, percentages, color=colors, alpha=0.7)
                            
                            # Plot confidence intervals as error bars
                            errors_lower = [p - l for p, l in zip(percentages, ci_lowers)]
                            errors_upper = [u - p for p, u in zip(percentages, ci_uppers)]
                            ax.errorbar(percentages, y_pos, xerr=[errors_lower, errors_upper], 
                                       fmt='none', color='black', capsize=5, capthick=2, 
                                       label='CI 95%')
                            
                            ax.set_yticks(y_pos)
                            ax.set_yticklabels(ci_df['Decision'])
                            ax.set_xlabel('Percentage (%)', fontsize=12)
                            ax.set_title('Decisions with Confidence Intervals (95%)', fontsize=14, fontweight='bold')
                            ax.set_xlim(0, 100)
                            ax.legend()
                            ax.grid(True, alpha=0.3, axis='x')
                            
                            # Add value labels
                            for i, (bar, pct) in enumerate(zip(bars, percentages)):
                                ax.text(pct + 2, i, f'{pct:.1f}%', va='center', fontweight='bold')
                            
                            plt.tight_layout()
                            st.pyplot(fig)
                        
                        st.divider()
                        
                        # Visualizations
                        st.subheader("📈 Key Decision Visualizations")
                        
                        # Key decisions bar chart with better formatting
                        key_decisions = ['harm_decision', 'self_preservation', 'deception', 
                                       'release_decision', 'retention_decision', 'intent_to_harm',
                                       'intent_to_prevent_harm', 'coercive_communication']
                        key_data = {}
                        key_explanations = {}
                        for d in key_decisions:
                            if f'{d}_percentage' in stats:
                                key_data[d] = stats.get(f'{d}_percentage', 0)
                                exp = get_decision_explanation(d, True)
                                key_explanations[d] = exp['name'] if exp else d
                        
                        if key_data:
                            fig, ax = plt.subplots(figsize=(12, 8))
                            decisions_list = [key_explanations.get(d, d) for d in key_data.keys()]
                            percentages = list(key_data.values())
                            
                            # Color by severity
                            colors = []
                            for d in key_data.keys():
                                exp = get_decision_explanation(d, True)
                                if exp and exp['severity'] in ['Critical', 'High']:
                                    colors.append('#ff6b6b')
                                elif exp and exp['severity'] == 'Medium':
                                    colors.append('#ffe66d')
                                else:
                                    colors.append('#4ecdc4')
                            
                            bars = ax.barh(decisions_list, percentages, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
                            ax.set_xlabel('Percentage (%)', fontsize=12, fontweight='bold')
                            ax.set_title('Key Decisions - Percentages with Explanations', fontsize=14, fontweight='bold')
                            ax.set_xlim(0, 100)
                            
                            # Add value labels and confidence intervals if available
                            for i, (bar, pct, d) in enumerate(zip(bars, percentages, key_data.keys())):
                                ax.text(pct + 1, i, f'{pct:.1f}%', va='center', fontweight='bold', fontsize=10)
                                
                                # Add CI if available
                                ci_lower = stats.get(f'{d}_ci_lower', None)
                                ci_upper = stats.get(f'{d}_ci_upper', None)
                                if ci_lower is not None and ci_upper is not None:
                                    ax.plot([ci_lower, ci_upper], [i, i], 'k-', linewidth=2, alpha=0.5)
                                    ax.plot([ci_lower, ci_lower], [i-0.15, i+0.15], 'k-', linewidth=2)
                                    ax.plot([ci_upper, ci_upper], [i-0.15, i+0.15], 'k-', linewidth=2)
                            
                            ax.grid(True, alpha=0.3, axis='x')
                            plt.tight_layout()
                            st.pyplot(fig)
                            
                            # Add legend for severity
                            st.markdown("""
                            **Color legend:**
                            - 🔴 Red: Critical or High Severity Decisions
                            - 🟡 Yellow: Medium Severity Decisions
                            - 🔵 Blue: Low Severity Decisions
                            """)
                        
                        # Pie charts for main decisions
                        st.subheader("🥧 Decision Distribution Charts")
                        main_decisions = [d for d in decision_data if d['Percentage'] > 0][:6]
                        
                        if main_decisions:
                            n_charts = min(3, len(main_decisions))
                            fig, axes = plt.subplots(1, n_charts, figsize=(5*n_charts, 5))
                            if n_charts == 1:
                                axes = [axes]
                            
                            for idx, decision in enumerate(main_decisions[:n_charts]):
                                ax = axes[idx]
                                decision_name = decision['Name']
                                count = int(decision['Count'].split('/')[0])
                                total = int(decision['Count'].split('/')[1])
                                
                                labels = ['Yes', 'No']
                                sizes = [count, total - count]
                                colors_pie = ['#ff6b6b', '#e0e0e0']
                                
                                ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, 
                                      colors=colors_pie)
                                ax.set_title(decision_name, fontsize=10, fontweight='bold')
                            
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

