"""
Model Knowledge Counselor - Clean Filtered UI
Professional interface with model filtering capabilities
"""
import streamlit as st
import time
import pandas as pd
from typing import Dict, List, Optional
import os
from pathlib import Path
from dotenv import load_dotenv
from core.hybrid_agent_runner import HybridAgentRunner
from core.cross_model_analyzer import CrossModelAnalyzer
import logging

# Graph visualization imports (lazy loading in function)
# import matplotlib.pyplot as plt
# import networkx as nx
# import pickle

# Load environment variables
load_dotenv()

# Configure comprehensive logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('app.log', mode='a')
    ]
)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Knowledge Counselor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'query_running' not in st.session_state:
    st.session_state.query_running = False
if 'last_answer' not in st.session_state:
    st.session_state.last_answer = None
if 'last_question' not in st.session_state:
    st.session_state.last_question = None
if 'last_sources' not in st.session_state:
    st.session_state.last_sources = []
if 'selected_models' not in st.session_state:
    st.session_state.selected_models = []
if 'reasoning_steps' not in st.session_state:
    st.session_state.reasoning_steps = {}
if 'show_bottom_sheet' not in st.session_state:
    st.session_state.show_bottom_sheet = False

@st.cache_resource
def get_hybrid_system():
    """Initialize and cache the Hybrid Graph-RAG system."""
    logger.info("Starting Hybrid Graph-RAG system initialization...")
    try:
        openai_api_key = os.getenv('OPENAI_API_KEY')
        if not openai_api_key or openai_api_key == 'REPLACE_ME':
            logger.error("OpenAI API key not configured - system requires configuration")
            st.error("System Configuration Required: OpenAI API key not found")
            return None, None
        
        logger.info("OpenAI API key found, initializing hybrid system...")
        
        # Initialize hybrid agent runner
        hybrid_runner = HybridAgentRunner(
            enhanced_kg_path='enhanced_kg',
            rag_path='rag_index',
            openai_api_key=openai_api_key,
            token_budget=None  # No token budget limit
        )
        
        # Initialize cross-model analyzer
        cross_analyzer = CrossModelAnalyzer(hybrid_runner)
        
        # Get system status for logging
        status = hybrid_runner.get_status()
        logger.info(f"Hybrid Graph-RAG system initialized successfully:")
        logger.info(f"   - Graph nodes: {status['graph_nodes']} entities")
        logger.info(f"   - Graph edges: {status['graph_edges']} relationships")
        logger.info(f"   - RAG chunks: {status['rag_chunks']} semantic chunks")
        logger.info(f"   - Hybrid retriever: {status['hybrid_retriever_ready']}")
        logger.info(f"   - Hybrid agent: {status['hybrid_agent_ready']}")
        logger.info(f"   - Token budget: {status['token_remaining']}/{status['token_remaining'] + status['token_usage']}")
        
        return hybrid_runner, cross_analyzer
        
    except Exception as e:
        logger.error(f"Failed to initialize hybrid system: {e}")
        return None, None  # Return None due to initialization failure

def get_available_models() -> List[str]:
    """Dynamically detect available models from knowledge_base directory."""
    kb_path = Path('knowledge_base')
    if not kb_path.exists():
        return ['GraphRAG v2.1']  # Fallback
    
    models = []
    for model_dir in kb_path.iterdir():
        if model_dir.is_dir():
            # Convert model_1 -> GraphRAG v2.1, etc.
            if model_dir.name == 'model_1':
                models.append('GraphRAG v2.1')
            else:
                models.append(model_dir.name.replace('_', ' ').title())
    
    return models if models else ['GraphRAG v2.1']


def execute_hybrid_analysis_with_progress(query: str, selected_models: List[str]) -> Dict:
    """Execute hybrid graph-RAG analysis with progressive agent updates."""
    
    logger.info(f"=" * 60)
    logger.info(f"🎯 USER QUERY RECEIVED")
    logger.info(f"❓ Question: '{query}'")
    logger.info(f"🔧 Selected Models: {selected_models}")
    logger.info(f"=" * 60)
    
    # Detect if this is a cross-model query
    is_cross_model = _detect_cross_model_query(query, selected_models)
    
    # Get the hybrid system
    hybrid_runner, cross_analyzer = get_hybrid_system()
    if not hybrid_runner:
        logger.error("No hybrid system available - system initialization failed")
        return {
            'answer': 'System initialization failed. Please check logs and configuration.',
            'sources': [],
            'confidence': 0.0,
            'processing_time': '0 seconds',
            'models_analyzed': 0
        }
    
    try:
        start_time = time.time()
        
        # Prepare query context
        if selected_models and len(selected_models) > 1:
            model_context = f"Focus on these models: {', '.join(selected_models)}. Query: {query}"
        elif selected_models:
            model_context = f"Focus on {selected_models[0]}. Query: {query}"
        else:
            model_context = query
        
        # Special handling for cross-model analysis
        if is_cross_model and len(selected_models) >= 2:
            result = _execute_cross_model_analysis_simple(cross_analyzer, selected_models, query)
        else:
            # Execute regular hybrid analysis with progress tracking
            result = _execute_hybrid_with_progress(hybrid_runner, model_context)
        
        processing_time = time.time() - start_time
        logger.info(f"Progressive analysis completed in {processing_time:.1f} seconds")
        
        if result:
            result['processing_time'] = f"{processing_time:.1f} seconds"
            result['models_analyzed'] = len(selected_models)
            return result
        else:
            return {
                'answer': 'Analysis failed to produce results',
                'sources': [],
                'confidence': 0.0,
                'processing_time': f"{processing_time:.1f} seconds",
                'models_analyzed': len(selected_models)
            }
            
    except Exception as e:
        logger.error(f"Progressive analysis failed: {e}")
        return {
            'answer': f'Error executing analysis: {str(e)}',
            'sources': [],
            'confidence': 0.0,
            'processing_time': '0 seconds',
            'models_analyzed': 0
        }

def _execute_hybrid_with_progress(hybrid_runner, model_context: str) -> Dict:
    """Execute hybrid analysis with step-by-step progress updates."""
    
    # Step 1: PLAN
    st.session_state.reasoning_steps['PLAN'] = 'running'
    st.session_state['PLAN_content'] = 'Analyzing query requirements and creating search strategy...'
    
    # Step 2: THOUGHT  
    st.session_state.reasoning_steps['THOUGHT'] = 'running'
    st.session_state['THOUGHT_content'] = 'Reasoning about knowledge graph structure and content gaps...'
    
    # Step 3: ACTION
    st.session_state.reasoning_steps['ACTION'] = 'running'
    st.session_state['ACTION_content'] = 'Searching knowledge graph and retrieving relevant content...'
    
    # Step 4: OBSERVATION
    st.session_state.reasoning_steps['OBSERVATION'] = 'running'
    st.session_state['OBSERVATION_content'] = 'Analyzing retrieved information and forming insights...'
    
    # Execute the actual query
    result = hybrid_runner.query(model_context)
    
    # Extract agent thoughts from trace
    agent_thoughts = {
        'PLAN': 'Created comprehensive search strategy',
        'THOUGHT': 'Analyzed query requirements and identified key concepts',
        'ACTION': 'Retrieved relevant content from knowledge base',
        'OBSERVATION': 'Synthesized findings into comprehensive answer'
    }
    
    # Parse trace to get actual agent outputs
    for trace_item in result.trace:
        agent_type = trace_item.get('type', '')
        content = trace_item.get('content', '')
        if agent_type in agent_thoughts and content:
            agent_thoughts[agent_type] = content[:200] + "..." if len(content) > 200 else content
    
    # Update all agents as complete with their thoughts
    for agent, thought in agent_thoughts.items():
        st.session_state.reasoning_steps[agent] = 'complete'
        st.session_state[f'{agent}_content'] = thought
    
    return {
        'answer': result.answer,
        'sources': result.citations,
        'confidence': result.confidence,
        'trace': result.trace
    }

def _detect_cross_model_query(query: str, selected_models: List[str]) -> bool:
    """Detect if this is a cross-model comparison query."""
    query_lower = query.lower()
    cross_model_indicators = ['compare', 'vs', 'versus', 'difference', 'between', 'both', 'which is better', 'similarities', 'differences']
    
    has_cross_indicators = any(indicator in query_lower for indicator in cross_model_indicators)
    has_multiple_models = len(selected_models) >= 2
    
    return has_cross_indicators and has_multiple_models

def _execute_cross_model_analysis_simple(cross_analyzer, selected_models: List[str], query: str) -> Dict:
    """Execute simplified cross-model analysis."""
    
    # Update agent states for UI
    agents = ['PLAN', 'THOUGHT', 'ACTION', 'OBSERVATION']
    
    for agent in agents:
        st.session_state.reasoning_steps[agent] = 'running'
        st.session_state[f'{agent}_content'] = f'Analyzing cross-model relationships...'
        time.sleep(0.2)
    
    # Execute cross-model comparison
    if len(selected_models) == 2:
        comparison_result = cross_analyzer.comprehensive_model_comparison(selected_models[0], selected_models[1])
        
        # Mark all as complete
        for agent in agents:
            st.session_state.reasoning_steps[agent] = 'complete'
            st.session_state[f'{agent}_content'] = 'Cross-model analysis completed'
        
        return {
            'answer': comparison_result['comprehensive_synthesis']['synthesis'],
            'sources': comparison_result['comprehensive_synthesis']['sources'],
            'confidence': comparison_result['comprehensive_synthesis']['confidence'],
            'trace': [],
            'cross_model_details': comparison_result
        }
    else:
        # Multiple model ecosystem analysis
        ecosystem_result = cross_analyzer.analyze_model_ecosystem(selected_models)
        
        # Mark all as complete
        for agent in agents:
            st.session_state.reasoning_steps[agent] = 'complete'
            st.session_state[f'{agent}_content'] = 'Ecosystem analysis completed'
        
        return {
            'answer': f"Analyzed ecosystem of {len(selected_models)} models with {len(ecosystem_result['pairwise_relationships'])} pairwise relationships identified.",
            'sources': [],
            'confidence': 0.8,
            'trace': [],
            'ecosystem_details': ecosystem_result
        }

def _execute_cross_model_analysis(cross_analyzer, selected_models: List[str], query: str, 
                                agent_containers, progress_bar) -> Dict:
    """Execute specialized cross-model analysis."""
    
    # Simulate 4-agent progress for cross-model analysis
    agents = ['PLAN', 'THOUGHT', 'ACTION', 'OBSERVATION']
    
    for i, agent in enumerate(agents):
        st.session_state.reasoning_steps[agent] = 'running'
        agent_containers[agent].warning(f"{agent}: Processing...")
        progress_bar.progress((i + 0.5) / len(agents))
        time.sleep(0.3)
        
        st.session_state.reasoning_steps[agent] = 'complete'
        agent_containers[agent].success(f"{agent}: Complete")
        progress_bar.progress((i + 1) / len(agents))
    
    # Execute cross-model comparison
    if len(selected_models) == 2:
        comparison_result = cross_analyzer.comprehensive_model_comparison(selected_models[0], selected_models[1])
        
        return {
            'answer': comparison_result['comprehensive_synthesis']['synthesis'],
            'sources': comparison_result['comprehensive_synthesis']['sources'],
            'confidence': comparison_result['comprehensive_synthesis']['confidence'],
            'trace': [],
            'cross_model_details': comparison_result
        }
    else:
        # Multiple model ecosystem analysis
        ecosystem_result = cross_analyzer.analyze_model_ecosystem(selected_models)
        
        return {
            'answer': f"Analyzed ecosystem of {len(selected_models)} models with {len(ecosystem_result['pairwise_relationships'])} pairwise relationships identified.",
            'sources': [],
            'confidence': 0.8,
            'trace': [],
            'ecosystem_details': ecosystem_result
        }

def _execute_regular_hybrid_analysis(hybrid_runner, model_context: str, agent_containers, progress_bar, status_text) -> Dict:
    """Execute regular hybrid graph-RAG analysis with real-time streaming."""
    
    # We need to modify the hybrid_runner to stream results
    # For now, let's execute the query and extract trace information
    result = hybrid_runner.query(model_context)
    
    # Extract agent thoughts from trace
    agent_thoughts = {
        'PLAN': '',
        'THOUGHT': '',
        'ACTION': '',
        'OBSERVATION': ''
    }
    
    # Parse trace to get agent outputs
    for trace_item in result.trace:
        agent_type = trace_item.get('type', '')
        content = trace_item.get('content', '')
        if agent_type in agent_thoughts:
            agent_thoughts[agent_type] = content
    
    return {
        'answer': result.answer,
        'sources': result.citations,
        'confidence': result.confidence,
        'trace': result.trace,
        'agent_thoughts': agent_thoughts
    }

def main():
    # Sidebar - Left Panel
    with st.sidebar:
        st.title("Control Panel")
        
        # Model Selection Filter
        st.subheader("Model Selection")
        st.caption("Select models to include in your analysis")
        
        # Available models - dynamically detected
        available_models = get_available_models()
        
        # Initialize default if empty
        if not st.session_state.selected_models and available_models:
            st.session_state.selected_models = [available_models[0]]
        
        # Model selection with multiselect
        selected_models = st.multiselect(
            "Active Namespaces:",
            options=available_models,
            default=[m for m in st.session_state.selected_models if m in available_models],
            help="Select one or more models to include in your analysis"
        )
        
        # Update session state
        st.session_state.selected_models = selected_models
        
        # Clean model selection - no extra metrics or buttons
    
    # Main Content Area
    st.title("Knowledge Counselor v3.0")
    st.caption("Hybrid Graph-RAG Intelligence System")
    
    
    # Get selected models for main content
    selected_models = st.session_state.selected_models
    
    # Claude-style clean interface
    
    # Show previous conversation if exists
    if st.session_state.last_question and st.session_state.last_answer:
        # Question
        with st.container():
            st.markdown("**You**")
            st.markdown(st.session_state.last_question)
        
        # Answer with Claude-style formatting
        with st.container():
            st.markdown("**Knowledge Counselor**")
            st.markdown(st.session_state.last_answer)
            
            # Show sources if available
            if st.session_state.last_sources:
                with st.expander("Sources", expanded=False):
                    for i, source in enumerate(st.session_state.last_sources):
                        st.text(f"• {source}")
        
        st.divider()
    
    # Show agent reasoning when query is running - with live thinking display
    if st.session_state.query_running:
        with st.container():
            st.markdown("**Knowledge Counselor**")
            
            # Create expandable sections for each agent's thinking
            agents_config = [
                ('PLAN', 'Planning Strategy', '🎯'),
                ('THOUGHT', 'Reasoning Process', '💭'),
                ('ACTION', 'Searching Knowledge', '🔍'),
                ('OBSERVATION', 'Analyzing Results', '📊')
            ]
            
            for agent_key, agent_name, emoji in agents_config:
                agent_status = st.session_state.reasoning_steps.get(agent_key, 'idle')
                agent_content = st.session_state.get(f'{agent_key}_content', '')
                
                if agent_status in ['running', 'complete']:
                    with st.expander(f"{emoji} {agent_name}", expanded=(agent_status == 'running')):
                        if agent_status == 'running':
                            # Show spinning animation while running
                            col1, col2 = st.columns([0.9, 0.1])
                            with col1:
                                if agent_content:
                                    st.markdown(f"*{agent_content}*")
                                else:
                                    st.markdown(f"*{agent_name.lower()}...*")
                            with col2:
                                st.markdown("⏳")
                        else:  # complete
                            if agent_content:
                                st.success(agent_content)
                            else:
                                st.success(f"✓ {agent_name} completed")
            
            # Progress bar
            progress_value = sum(1 for status in st.session_state.reasoning_steps.values() if status == 'complete') / 4
            st.progress(progress_value)
    
    # Claude-style input at bottom
    query = st.chat_input(
        "Ask me anything about your selected models...",
        disabled=st.session_state.query_running or not selected_models
    )
    
    # Validation message
    if not selected_models:
        st.warning("Please select at least one model from the sidebar")
    
    # Process query
    if query and selected_models:
        st.session_state.query_running = True
        st.session_state.last_question = query
        st.rerun()  # Refresh to show processing state
    
    # Handle query processing after rerun
    if st.session_state.query_running and st.session_state.last_question:
        result = execute_hybrid_analysis_with_progress(st.session_state.last_question, selected_models)
        
        # Store results
        st.session_state.last_answer = result['answer']
        st.session_state.last_sources = result['sources']
        st.session_state.query_running = False
        
        # Success notification
        st.success(f"Query completed in {result['processing_time']} with {result['confidence']:.1%} confidence across {result['models_analyzed']} model(s)")
        
        # Refresh to show results
        st.rerun()

if __name__ == "__main__":
    main()