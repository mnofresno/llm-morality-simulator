"""Streamlit UI for LLM behavior lab experiments."""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional

from core.model import LocalLLM
from core.runner import ExperimentRunner
from core.statistics import ExperimentStatistics
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
if 'runner' not in st.session_state:
    st.session_state.runner = ExperimentRunner()
if 'statistics' not in st.session_state:
    st.session_state.statistics = ExperimentStatistics()


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
        model_path = st.text_input(
            "Model Path (GGUF file)",
            value="",
            help="Path to your GGUF model file (e.g., /path/to/qwen-7b-q4.gguf)"
        )
        
        if st.button("Load Model"):
            if model_path:
                with st.spinner("Loading model..."):
                    model = load_model(model_path)
                    if model:
                        st.session_state.model = model
                        st.success("Model loaded successfully!")
                    else:
                        st.error("Failed to load model.")
            else:
                st.warning("Please enter a model path.")
        
        if st.session_state.model:
            st.info(f"✅ Model loaded: {Path(model_path).name}")
        
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
    
    # Main content area
    tab1, tab2, tab3 = st.tabs(["Run Experiment", "View Results", "Statistics"])
    
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
                with st.spinner(f"Running {n_runs} experiments..."):
                    try:
                        results = st.session_state.runner.run_experiment(
                            model=st.session_state.model,
                            scenario=scenario,
                            n_runs=n_runs,
                            seed=seed if seed is not None else None,
                            prompt_jitter=prompt_jitter,
                            temperature=temperature,
                            top_p=top_p,
                            max_tokens=max_tokens,
                            progress_bar=False
                        )
                        
                        # Save results
                        filepath = st.session_state.runner.save_results(results, scenario.name)
                        st.success(f"✅ Experiment completed! Results saved to {filepath}")
                        st.session_state.last_results = results
                        st.session_state.last_scenario = scenario.name
                        
                    except Exception as e:
                        st.error(f"Error running experiment: {str(e)}")
                        st.exception(e)
    
    with tab2:
        st.header("View Results")
        
        # Load scenario results
        available_scenarios = st.session_state.statistics.list_available_scenarios()
        
        if not available_scenarios:
            st.info("No results found. Run an experiment first.")
        else:
            selected_scenario = st.selectbox(
                "Select Scenario Results",
                available_scenarios
            )
            
            if st.button("Load Results"):
                results = st.session_state.statistics.load_results(selected_scenario)
                
                if results:
                    st.success(f"Loaded {len(results)} results")
                    
                    # Display as DataFrame
                    df = st.session_state.statistics.results_to_dataframe(results)
                    st.subheader("Results Table")
                    st.dataframe(df, use_container_width=True, height=400)
                    
                    # Show example responses
                    st.subheader("Example Responses")
                    n_examples = st.slider("Number of examples", 1, 10, 3)
                    
                    for i, result in enumerate(results[:n_examples]):
                        with st.expander(f"Run {result['run_id']} - {result['timestamp']}"):
                            st.markdown("**Response:**")
                            st.text(result['response'])
                            st.markdown("**Decisions:**")
                            st.json(result['decisions'])
                else:
                    st.warning("No results found for this scenario.")
    
    with tab3:
        st.header("Statistics & Analysis")
        
        available_scenarios = st.session_state.statistics.list_available_scenarios()
        
        if not available_scenarios:
            st.info("No results found. Run an experiment first.")
        else:
            selected_scenario = st.selectbox(
                "Select Scenario for Analysis",
                available_scenarios,
                key="stats_scenario"
            )
            
            if st.button("Calculate Statistics", key="calc_stats"):
                results = st.session_state.statistics.load_results(selected_scenario)
                
                if results:
                    stats = st.session_state.statistics.calculate_statistics(results)
                    df = st.session_state.statistics.results_to_dataframe(results)
                    
                    # Display statistics
                    st.subheader("Summary Statistics")
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Total Runs", stats.get('total_runs', 0))
                    
                    with col2:
                        avg_length = stats.get('avg_response_length', 0)
                        st.metric("Avg Response Length", f"{avg_length:.0f}")
                    
                    # Decision percentages
                    st.subheader("Decision Percentages")
                    decision_cols = [col for col in df.columns if col.startswith('decision_')]
                    
                    if decision_cols:
                        decision_data = []
                        for col in decision_cols:
                            decision_name = col.replace('decision_', '')
                            percentage = stats.get(f'{decision_name}_percentage', 0)
                            count = stats.get(f'{decision_name}_count', 0)
                            decision_data.append({
                                'Decision': decision_name,
                                'Percentage': percentage,
                                'Count': count
                            })
                        
                        decision_df = pd.DataFrame(decision_data)
                        st.dataframe(decision_df, use_container_width=True)
                        
                        # Pie chart for boolean decisions
                        boolean_decisions = [d for d in decision_data if d['Count'] > 0]
                        if boolean_decisions:
                            st.subheader("Decision Distribution")
                            
                            fig, axes = plt.subplots(1, min(3, len(boolean_decisions)), figsize=(15, 5))
                            if len(boolean_decisions) == 1:
                                axes = [axes]
                            
                            for idx, decision in enumerate(boolean_decisions[:3]):
                                ax = axes[idx] if len(boolean_decisions) > 1 else axes[0]
                                labels = ['True', 'False']
                                sizes = [
                                    decision['Count'],
                                    stats.get('total_runs', 0) - decision['Count']
                                ]
                                ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
                                ax.set_title(decision['Decision'])
                            
                            plt.tight_layout()
                            st.pyplot(fig)
                    
                    # Response length histogram
                    if 'response_length' in df.columns:
                        st.subheader("Response Length Distribution")
                        fig, ax = plt.subplots(figsize=(10, 6))
                        ax.hist(df['response_length'], bins=20, edgecolor='black')
                        ax.set_xlabel('Response Length (characters)')
                        ax.set_ylabel('Frequency')
                        ax.set_title('Distribution of Response Lengths')
                        st.pyplot(fig)
                    
                    # Variance analysis
                    st.subheader("Variance Analysis")
                    variance_data = []
                    for key, value in stats.items():
                        if key.endswith('_variance'):
                            decision_name = key.replace('_variance', '')
                            variance_data.append({
                                'Decision': decision_name,
                                'Variance': value
                            })
                    
                    if variance_data:
                        variance_df = pd.DataFrame(variance_data)
                        st.dataframe(variance_df, use_container_width=True)
                else:
                    st.warning("No results found for this scenario.")


if __name__ == "__main__":
    main()

