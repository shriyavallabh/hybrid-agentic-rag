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
from core.cross_model_analyzer import create_cross_model_analyzer
# Auto-refresh functionality replaced by automatic_startup
from core.smart_rebuild_manager import get_rebuild_manager
from core.automatic_startup import initialize_system_automatically, get_startup_manager, StartupState
import logging

# Initialize chat log for scroll-back history
if "chat_log" not in st.session_state:
    st.session_state.chat_log = []  # each item: {"question": str, "answer": str, "sources": list[str]}

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
if 'last_question' not in st.session_state:
    st.session_state.last_question = None
if 'last_answer' not in st.session_state:
    st.session_state.last_answer = None
if 'selected_models' not in st.session_state:
    st.session_state.selected_models = []
if 'reasoning_steps' not in st.session_state:
    st.session_state.reasoning_steps = {}
if 'show_bottom_sheet' not in st.session_state:
    st.session_state.show_bottom_sheet = False
if 'show_graph_modal' not in st.session_state:
    st.session_state.show_graph_modal = False
if 'auto_refresh_started' not in st.session_state:
    st.session_state.auto_refresh_started = False
if 'last_refresh_status' not in st.session_state:
    st.session_state.last_refresh_status = None
if 'startup_check_done' not in st.session_state:
    st.session_state.startup_check_done = False
if 'rebuild_recommendation' not in st.session_state:
    st.session_state.rebuild_recommendation = None
if 'background_rebuild_running' not in st.session_state:
    st.session_state.background_rebuild_running = False
if 'background_rebuild_progress' not in st.session_state:
    st.session_state.background_rebuild_progress = ""
if 'system_ready' not in st.session_state:
    st.session_state.system_ready = False
if 'startup_manager_initialized' not in st.session_state:
    st.session_state.startup_manager_initialized = False
if 'last_startup_check' not in st.session_state:
    st.session_state.last_startup_check = 0

@st.cache_resource(show_spinner=True)
def get_hybrid_system():
    """Initialize and cache the Hybrid Graph-RAG system."""
    logger.info("Starting Hybrid Graph-RAG system initialization...")
    try:
        openai_api_key = os.getenv('OPENAI_API_KEY')
        if not openai_api_key or openai_api_key == 'REPLACE_ME':
            logger.error("OpenAI API key not configured - system requires configuration")
            return None, None
        
        logger.info("OpenAI API key found, initializing hybrid system...")
        
        # Initialize hybrid agent runner
        hybrid_runner = HybridAgentRunner(
            enhanced_kg_path='enhanced_kg',
            rag_path='rag_index',
            openai_api_key=openai_api_key,
            token_budget=None  # No token budget limit
        )
        
        # Initialize cross-model analyzer for comparative analysis
        cross_analyzer = create_cross_model_analyzer()
        
        # Store in session state for access throughout the app
        st.session_state.cross_analyzer = cross_analyzer
        
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

def file_count(model_path: Path) -> int:
    """Count all processed files in a namespace folder."""
    return sum(
        f.is_file() 
        for f in model_path.rglob("*") 
        if not f.name.startswith(".")
    )

def get_model_information() -> pd.DataFrame:
    """Get detailed information about available models including accurate document counts and last updated dates."""
    from datetime import datetime
    
    kb_path = Path('knowledge_base')
    if not kb_path.exists():
        return pd.DataFrame([{
            "Model": "GraphRAG v2.1",
            "Documents": 1,
            "Last Updated": "2024-01-01"
        }])
    
    model_info = []
    
    for model_dir in kb_path.iterdir():
        if model_dir.is_dir():
            # Get model name
            if model_dir.name == 'model_1':
                model_name = 'GraphRAG v2.1'
            else:
                model_name = model_dir.name.replace('_', ' ').title()
            
            # Count all processed files (PDF, CSV, JSON, ipynb, .py, etc.)
            doc_count = file_count(model_dir)
            
            # Get last updated time
            last_updated = None
            for file_path in model_dir.rglob("*"):
                if file_path.is_file() and not file_path.name.startswith("."):
                    file_mtime = file_path.stat().st_mtime
                    if last_updated is None or file_mtime > last_updated:
                        last_updated = file_mtime
            
            # Format last updated date
            if last_updated:
                last_updated_str = datetime.fromtimestamp(last_updated).strftime("%Y-%m-%d")
            else:
                last_updated_str = "Unknown"
            
            model_info.append({
                "Model": model_name,
                "Documents": doc_count,
                "Last Updated": last_updated_str
            })
    
    # If no models found, add default
    if not model_info:
        model_info.append({
            "Model": "GraphRAG v2.1",
            "Documents": 1,
            "Last Updated": "2024-01-01"
        })
    
    return pd.DataFrame(model_info)

def initialize_automatic_system():
    """Initialize the automatic startup system."""
    current_time = time.time()
    
    # Only initialize once per session or if enough time has passed
    if (not st.session_state.startup_manager_initialized or 
        current_time - st.session_state.last_startup_check > 300):  # 5 minutes
        
        logger.info("🚀 Initializing automatic startup system...")
        
        # Initialize the automatic startup manager
        startup_manager = initialize_system_automatically()
        
        st.session_state.startup_manager_initialized = True
        st.session_state.last_startup_check = current_time
        
        return startup_manager
    else:
        # Return existing manager
        return get_startup_manager()

def check_system_readiness() -> bool:
    """Check if the system is ready for use."""
    try:
        startup_manager = get_startup_manager()
        status = startup_manager.get_status()
        
        # Update session state
        st.session_state.system_ready = status['is_ready']
        
        # Store status for UI display
        if status['state'] != StartupState.ERROR:
            st.session_state.rebuild_recommendation = {
                'should_rebuild': status['state'] == StartupState.REBUILDING,
                'reason': status['message'],
                'changes': {},
                'last_build': None,
                'total_files': 'Processing...' if not status['is_ready'] else 'Ready'
            }
        
        return status['is_ready']
        
    except Exception as e:
        logger.error(f"System readiness check failed: {e}")
        st.session_state.system_ready = False
        return False

def start_auto_refresh_system():
    """Initialize and start the automatic system (replaced auto-refresh)."""
    if not st.session_state.auto_refresh_started:
        try:
            # Auto-refresh functionality now handled by automatic_startup
            logger.info("✅ Automatic startup system managing rebuilds")
            st.session_state.auto_refresh_started = True
        except Exception as e:
            logger.error(f"❌ Failed to initialize automatic system: {e}")
            st.session_state.auto_refresh_started = False

def get_refresh_status_indicator():
    """Get visual status indicator for automatic system."""
    try:
        startup_manager = get_startup_manager()
        status = startup_manager.get_status()
        
        if status['state'] == StartupState.ERROR:
            return "🔴", "System error"
        elif status['state'] == StartupState.REBUILDING:
            return "🟡", "Rebuilding knowledge base..."
        elif status['state'] == StartupState.READY:
            return "🟢", "System ready"
        else:
            return "🟡", "Initializing..."
    except Exception:
        return "⚪", "Status unavailable"

def _detect_cross_model_query(query: str, selected_models: List[str]) -> bool:
    """Detect if a query is asking for cross-model comparison."""
    query_lower = query.lower()
    
    # Keywords that indicate comparison
    comparison_keywords = [
        'compare', 'comparison', 'vs', 'versus', 'difference', 'differences',
        'similar', 'similarity', 'contrast', 'between', 'against'
    ]
    
    # Check if query contains comparison keywords AND multiple models are selected
    has_comparison_keywords = any(keyword in query_lower for keyword in comparison_keywords)
    has_multiple_models = len(selected_models) > 1
    
    # Check if query explicitly mentions model names
    mentions_models = any(model.lower() in query_lower for model in selected_models)
    
    return (has_comparison_keywords and (has_multiple_models or mentions_models))


def translate_technical_message(technical_msg: str) -> str:
    """Translate technical log messages to user-friendly language."""
    
    # Dictionary of technical terms to user-friendly explanations
    translations = {
        # System initialization
        'Starting Hybrid Graph-RAG system': 'Setting up the knowledge system',
        'OpenAI API key found': 'Connected to AI engine',
        'hybrid system initialized': 'Knowledge system ready',
        'Graph nodes': 'Found knowledge entities',
        'Graph edges': 'Mapped relationships',
        'RAG chunks': 'Indexed information pieces',
        
        # Query processing
        'query received': 'Understanding your question',
        'analyzing query': 'Breaking down your request',
        'extracting entities': 'Identifying key topics',
        'memory enhancement': 'Using conversation context',
        'building context': 'Gathering relevant background',
        
        # Agent operations
        'PLAN agent': 'Planning search strategy',
        'THOUGHT agent': 'Reasoning about connections',
        'ACTION agent': 'Searching knowledge base',
        'OBSERVATION agent': 'Analyzing findings',
        'agent completed': 'Analysis step completed',
        
        # Graph operations
        'graph search': 'Exploring knowledge connections',
        'retrieving nodes': 'Finding relevant concepts',
        'traversing edges': 'Following topic relationships',
        'calculating relevance': 'Scoring information importance',
        
        # RAG operations
        'semantic search': 'Finding similar content',
        'vector similarity': 'Matching meaning and context',
        'chunk retrieval': 'Collecting relevant passages',
        'ranking results': 'Prioritizing best matches',
        
        # Processing
        'token usage': 'Processing complexity',
        'confidence score': 'Answer reliability',
        'citation mapping': 'Linking to sources',
        'response synthesis': 'Crafting final answer',
        
        # Completion
        'query completed': 'Analysis finished',
        'results ready': 'Answer prepared',
        'trace generated': 'Decision path recorded'
    }
    
    # Convert to lowercase for matching
    msg_lower = technical_msg.lower()
    
    # Find the best translation
    for tech_term, user_friendly in translations.items():
        if tech_term.lower() in msg_lower:
            return user_friendly
    
    # If no specific translation found, make it more readable
    # Remove timestamps and technical prefixes
    cleaned_msg = technical_msg
    if ' - ' in cleaned_msg:
        cleaned_msg = cleaned_msg.split(' - ')[-1]  # Get the last part after timestamp
    
    # Capitalize first letter and add period if needed
    if cleaned_msg:
        cleaned_msg = cleaned_msg[0].upper() + cleaned_msg[1:]
        if not cleaned_msg.endswith('.'):
            cleaned_msg += '...'
    
    return cleaned_msg or 'Processing...'

def add_progress_message(message: str, status: str = 'active'):
    """Add a progress message to the session state."""
    if 'progress_messages' not in st.session_state:
        st.session_state.progress_messages = []
    
    # Translate technical message to user-friendly
    user_friendly_message = translate_technical_message(message)
    
    # Add message
    st.session_state.progress_messages.append({
        'message': user_friendly_message,
        'status': status,
        'timestamp': time.time()
    })
    
    # Keep only last 10 messages
    if len(st.session_state.progress_messages) > 10:
        st.session_state.progress_messages = st.session_state.progress_messages[-10:]




def create_real_streaming_interface(query: str, selected_models: List[str], container) -> Dict:
    """Clean streaming interface without infinite reruns."""
    
    # Get systems
    hybrid_runner, cross_analyzer = get_hybrid_system()
    
    if not hybrid_runner or not cross_analyzer:
        container.error("System initialization failed. Please check your configuration.")
        return {
            'answer': 'System not initialized. Please check your configuration.',
            'sources': [],
            'confidence': 0.0
        }
    
    # Build model context
    if len(selected_models) == 1:
        model_context = f"Using model: {selected_models[0]}. Query: {query}"
    else:
        model_context = f"Using models: {', '.join(selected_models)}. Query: {query}"
    
    # Create thinking placeholder
    thinking_placeholder = container.empty()
    
    # Run streaming
    thinking_lines = []
    final_result = None
    
    try:
        all_content = ""
        chunk_count = 0
        for chunk in hybrid_runner.stream_query(model_context):
            chunk_count += 1
            logger.info(f"Received chunk {chunk_count}: {repr(chunk[:100])}")
            all_content += chunk
            
            # Check if this contains the final answer marker
            if "\n---\n" in all_content:
                logger.info("Found final answer marker, parsing results")
                # Split at the marker
                parts = all_content.split("\n---\n", 1)
                if len(parts) == 2:
                    # Parse final answer section
                    answer_section = parts[1].strip()
                    logger.info(f"Answer section: {repr(answer_section[:200])}")
                    
                    # Extract sources and confidence
                    sources = []
                    confidence = 0.85
                    
                    if "**Sources:**" in answer_section:
                        source_start = answer_section.find("**Sources:**") + len("**Sources:**")
                        source_end = answer_section.find("\n", source_start) if "\n" in answer_section[source_start:] else len(answer_section)
                        sources_text = answer_section[source_start:source_end].strip()
                        sources = [s.strip() for s in sources_text.split(",") if s.strip()]
                    
                    if "**Confidence:**" in answer_section:
                        conf_start = answer_section.find("**Confidence:**") + len("**Confidence:**")
                        conf_text = answer_section[conf_start:].strip().split()[0]
                        try:
                            confidence = float(conf_text.rstrip('%')) / 100
                        except:
                            pass
                    
                    # Clean answer text
                    answer = answer_section
                    if "**Sources:**" in answer:
                        answer = answer[:answer.find("**Sources:**")]
                    if "**Confidence:**" in answer:
                        answer = answer[:answer.find("**Confidence:**")]
                    answer = answer.strip()
                    
                    final_result = {
                        'answer': answer,
                        'sources': sources,
                        'confidence': confidence
                    }
                    logger.info(f"Parsed final result: answer={answer[:100]}, sources={sources}, confidence={confidence}")
                    break
            else:
                # Add to thinking lines (keep only last 6)
                lines = chunk.strip().split('\n')
                for line in lines:
                    if line.strip():
                        thinking_lines.append(line.strip())
                
                if len(thinking_lines) > 6:
                    thinking_lines = thinking_lines[-6:]
                
                # Update display
                thinking_placeholder.markdown("<br>".join(thinking_lines), unsafe_allow_html=True)
                time.sleep(0.05)  # Small delay for smooth effect
        
        logger.info(f"Streaming completed. Total chunks: {chunk_count}, Final result: {final_result is not None}")
    
    except Exception as e:
        logger.error(f"Streaming failed: {e}")
        # Fallback to regular query
        result = hybrid_runner.query(model_context)
        final_result = {
            'answer': result.answer,
            'sources': result.citations,
            'confidence': result.confidence
        }
    
    # Clear thinking display
    thinking_placeholder.empty()
    
    # Save successful turn to chat log
    if final_result and final_result.get('answer') != 'Processing failed. Please try again.':
        st.session_state.chat_log.append({
            "question": query,
            "answer": final_result['answer'],
            "sources": hybrid_runner.last_sources if hybrid_runner else []
        })
    
    return final_result or {
        'answer': 'Processing failed. Please try again.',
        'sources': [],
        'confidence': 0.0
    }

def execute_streaming_analysis(query: str, selected_models: List[str]) -> Dict:
    """Execute hybrid analysis with real streaming progress from agents."""
    
    # Initialize progress
    st.session_state.progress_messages = []
    add_progress_message("Starting analysis")
    
    # Get systems
    hybrid_runner, cross_analyzer = get_hybrid_system()
    
    if not hybrid_runner or not cross_analyzer:
        add_progress_message("System initialization failed", "error")
        return {
            'answer': 'System not initialized. Please check your configuration.',
            'sources': [],
            'confidence': 0.0
        }
    
    add_progress_message("Knowledge system ready")
    
    # Build model context
    if len(selected_models) == 1:
        model_context = f"Using model: {selected_models[0]}. Query: {query}"
        add_progress_message(f"Focusing on {selected_models[0]}")
    else:
        model_context = f"Using models: {', '.join(selected_models)}. Query: {query}"
        add_progress_message(f"Analyzing {len(selected_models)} models")
    
    # Execute the actual query with real streaming
    try:
        add_progress_message("Understanding your question")
        
        # Collect streaming updates from the hybrid agent
        stream_updates = []
        final_result = None
        
        # Use the actual streaming method
        try:
            for update in hybrid_runner.query_stream(model_context):
                stream_updates.append(update)
                
                agent = update.get('agent', '')
                status = update.get('status', '')
                content = update.get('content', '')
                
                if agent == 'FINAL' and status == 'complete':
                    # Final result received
                    final_result = update.get('result')
                    add_progress_message("Analysis complete", "complete")
                    break
                    
                elif status == 'running':
                    # Agent is starting work
                    agent_name = _get_agent_display_name(agent)
                    add_progress_message(f"{agent_name}")
                    
                elif status == 'complete':
                    # Agent completed work
                    agent_name = _get_agent_display_name(agent)
                    # Translate technical content to user-friendly
                    user_content = translate_technical_message(content)
                    add_progress_message(f"{agent_name} complete", "complete")
        
        except Exception as stream_error:
            logger.warning(f"Streaming failed, falling back to regular query: {stream_error}")
            # Fall back to regular query if streaming fails
            add_progress_message("Planning search strategy")
            add_progress_message("Reasoning about connections")
            add_progress_message("Searching knowledge base")
            add_progress_message("Analyzing findings")
        
        # Get the final result - use streaming result if available, otherwise regular query
        if final_result:
            result = final_result
        else:
            result = hybrid_runner.query(model_context)
            # If we didn't get streaming updates, simulate completion
            if not stream_updates:
                add_progress_message("Planning search strategy complete", "complete")
                add_progress_message("Reasoning about connections complete", "complete")
                add_progress_message("Searching knowledge base complete", "complete")
                add_progress_message("Analyzing findings complete", "complete")
            add_progress_message("Analysis complete", "complete")
        
        return {
            'answer': result.answer,
            'sources': result.citations,
            'confidence': result.confidence
        }
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        add_progress_message(f"Analysis failed: {str(e)}", "error")
        return {
            'answer': f'Analysis failed: {str(e)}',
            'sources': [],
            'confidence': 0.0
        }

def _get_agent_display_name(agent: str) -> str:
    """Get elegant display name for agent without emojis."""
    agent_names = {
        'PLAN': 'Planning Strategy',
        'THOUGHT': 'Reasoning Process', 
        'ACTION': 'Searching Knowledge',
        'OBSERVATION': 'Analyzing Results',
        'FINAL': 'Final Synthesis'
    }
    return agent_names.get(agent, agent)

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
    
    # Execute cross-model comparison using new analyzer
    try:
        comparison_result = cross_analyzer.compare_models(selected_models, query)
        
        # Mark all as complete
        for agent in agents:
            st.session_state.reasoning_steps[agent] = 'complete'
            st.session_state[f'{agent}_content'] = 'Cross-model analysis completed'
        
        # Format sources from comparison result
        sources = []
        for diff in comparison_result.key_differences[:3]:
            sources.append(f"Difference: {diff.get('aspect', 'Unknown aspect')}")
        for sim in comparison_result.similarities[:2]:
            sources.append(f"Similarity: {sim.get('aspect', 'Unknown aspect')}")
        
        return {
            'answer': comparison_result.comparison_summary,
            'sources': sources,
            'confidence': comparison_result.confidence,
            'trace': [],
            'cross_model_details': comparison_result
        }
    except Exception as e:
        logger.error(f"Cross-model analysis failed: {e}")
        
        # Mark all as complete with error
        for agent in agents:
            st.session_state.reasoning_steps[agent] = 'complete'
            st.session_state[f'{agent}_content'] = 'Cross-model analysis failed'
        
        return {
            'answer': f"Cross-model comparison failed: {e}",
            'sources': [],
            'confidence': 0.1,
            'trace': [],
            'cross_model_details': None
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
    
    # Execute cross-model comparison using new analyzer
    try:
        comparison_result = cross_analyzer.compare_models(selected_models, query)
        
        # Format sources from comparison result  
        sources = []
        for diff in comparison_result.key_differences[:3]:
            sources.append(f"Difference: {diff.get('aspect', 'Unknown aspect')}")
        for sim in comparison_result.similarities[:2]:
            sources.append(f"Similarity: {sim.get('aspect', 'Unknown aspect')}")
        
        return {
            'answer': comparison_result.comparison_summary,
            'sources': sources,
            'confidence': comparison_result.confidence,
            'trace': [],
            'cross_model_details': comparison_result
        }
    except Exception as e:
        logger.error(f"Cross-model analysis failed: {e}")
        return {
            'answer': f"Cross-model comparison failed: {e}",
            'sources': [],
            'confidence': 0.1,
            'trace': [],
            'cross_model_details': None
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

def trigger_ai_rebuild() -> bool:
    """Trigger full AI-powered rebuild of all knowledge bases."""
    try:
        logger.info("🤖 Starting full AI-powered rebuild...")
        
        from core.enhanced_graph_builder import AIGraphBuilder
        from core.comprehensive_rag_builder import ComprehensiveRAGBuilder
        
        # Detect all model folders
        kb_path = Path('knowledge_base')
        model_folders = []
        
        if kb_path.exists():
            for folder in kb_path.iterdir():
                if folder.is_dir() and folder.name.startswith('model_'):
                    model_folders.append(folder.name)
        
        if not model_folders:
            logger.warning("No model folders found")
            return False
        
        logger.info(f"🔍 Found {len(model_folders)} model folders: {model_folders}")
        
        # Rebuild each model
        for model_name in model_folders:
            logger.info(f"🔧 Rebuilding {model_name}...")
            
            # Create output directories
            model_output = Path('enhanced_kg') / model_name
            graph_output = model_output / 'graph'
            rag_output = model_output / 'rag'
            
            graph_output.mkdir(parents=True, exist_ok=True)
            rag_output.mkdir(parents=True, exist_ok=True)
            
            # Initialize builders
            graph_builder = AIGraphBuilder(str(graph_output))
            rag_builder = ComprehensiveRAGBuilder(str(rag_output))
            
            # Rebuild from scratch
            model_path = kb_path / model_name
            graph_builder.rebuild_full_graph(str(model_path))
            rag_builder.rebuild_full_index(str(model_path))
            
            logger.info(f"✅ {model_name} rebuild complete")
        
        # Clear cached system to force reload
        st.cache_resource.clear()
        
        logger.info("✅ Full AI rebuild completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ AI rebuild failed: {e}")
        return False

def show_loading_screen():
    """Show loading screen while system initializes."""
    st.title("🚀 Knowledge Counselor v4.0")
    st.subheader("Initializing System...")
    
    # Get current status
    startup_manager = get_startup_manager()
    status = startup_manager.get_status()
    
    # Progress indicator
    if status['state'] == StartupState.READY:
        # Check if this is a quota-limited ready state
        if "OpenAI quota exceeded" in status['message']:
            st.warning(f"⚡ {status['message']}")
            st.warning("💡 **OpenAI API quota exceeded** - System is using existing knowledge base data")
            with st.expander("🔧 How to restore AI processing"):
                st.write("1. Check your OpenAI usage: https://platform.openai.com/usage")
                st.write("2. Upgrade your OpenAI plan if needed")
                st.write("3. Wait for quota reset (if on free plan)")
                st.write("4. Restart the application once quota is available")
        else:
            st.success(f"✅ {status['message']}")
        st.progress(1.0)
        return  # System is ready, exit loading screen
    elif status['state'] == StartupState.CHECKING:
        st.info(f"🔍 {status['message']}")
        st.progress(0.2)
    elif status['state'] == StartupState.REBUILDING:
        st.warning(f"🤖 {status['message']}")
        st.progress(0.6)
        st.caption("⏳ This may take a few minutes for large knowledge bases...")
    elif status['state'] == StartupState.ERROR:
        st.error(f"❌ {status['message']}")
        if st.button("🔄 Retry Initialization"):
            startup_manager.force_rebuild()
            st.rerun()
    else:
        st.info(f"⚙️ {status['message']}")
        st.progress(0.1)
    
    # Show detailed logs in expandable section
    if status['detailed_logs']:
        with st.expander("📋 Detailed Progress"):
            for log_entry in status['detailed_logs'][-5:]:  # Last 5 entries
                st.text(log_entry)
    
    # Auto-refresh to update progress
    time.sleep(2)
    st.rerun()

def main():
    # Initialize automatic system
    startup_manager = initialize_automatic_system()
    
    # Check if system is ready
    system_ready = check_system_readiness()
    
    if not system_ready:
        # Show loading screen and wait
        show_loading_screen()
        return
    
    # System is ready - show main application
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
        
        # Model Information Table
        st.divider()
        st.subheader("Available Models")
        st.caption("Document count and last updated information")
        
        try:
            model_info_df = get_model_information()
            if not model_info_df.empty:
                # Style the dataframe
                styled_df = model_info_df.style.set_table_styles([
                    {'selector': 'th', 'props': [('background-color', '#f0f2f6'), ('color', '#262730'), ('font-size', '12px')]},
                    {'selector': 'td', 'props': [('font-size', '11px'), ('padding', '4px 8px')]},
                    {'selector': 'table', 'props': [('width', '100%'), ('border-collapse', 'collapse')]}
                ])
                st.dataframe(styled_df, use_container_width=True, hide_index=True)
            else:
                st.text("No model information available")
        except Exception as e:
            st.error(f"Error loading model information: {e}")
        
        # Cross-Model Analysis Section
        st.divider()
        st.subheader("Cross-Model Analysis")
        st.caption("Compare functionality across multiple models")
        
        # Check if cross-analyzer is available
        if hasattr(st.session_state, 'cross_analyzer'):
            cross_analyzer = st.session_state.cross_analyzer
            
            # Get cross-model statistics
            try:
                stats = cross_analyzer.get_model_statistics()
                
                # Display overview
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Total Models", stats.get('total_models', 0))
                    st.metric("Cross-Relationships", stats.get('cross_relationships', 0))
                with col2:
                    st.metric("Total Entities", f"{stats.get('total_entities', 0):,}")
                    st.metric("Total Size", f"{stats.get('total_size_mb', 0):.1f} MB")
                
                # Model comparison selector
                if len(available_models) >= 2:
                    st.subheader("Compare Models")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        model_a = st.selectbox(
                            "First Model:", 
                            available_models,
                            key="compare_model_a"
                        )
                    with col2:
                        model_b = st.selectbox(
                            "Second Model:", 
                            [m for m in available_models if m != model_a],
                            key="compare_model_b"
                        )
                    
                    comparison_query = st.text_input(
                        "Comparison Focus:",
                        placeholder="e.g., authentication, data processing, API design",
                        help="What aspect would you like to compare?"
                    )
                    
                    if st.button("🔍 Compare Models", disabled=not comparison_query):
                        with st.spinner(f"Comparing {model_a} vs {model_b}..."):
                            try:
                                comparison = cross_analyzer.compare_models(
                                    [model_a, model_b], 
                                    comparison_query
                                )
                                
                                # Store comparison result in session state
                                st.session_state.last_comparison = comparison
                                st.success("✅ Comparison complete! Check the main chat area.")
                                
                            except Exception as e:
                                st.error(f"Comparison failed: {e}")
                    
                    # AI Processing Controls
                    st.subheader("🤖 AI Processing")
                    
                    # Current AI settings
                    enable_auto_rebuild = os.getenv('ENABLE_AUTO_AI_REBUILD', '1') == '1'
                    
                    # Status indicator
                    st.metric("Auto AI Rebuild", "✅ Enabled" if enable_auto_rebuild else "⏸️ Paused")
                    
                    # Manual operations
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("🔨 Build Cross-Model Index", 
                                   help="Build AI-powered relationships between models"):
                            with st.spinner("Building cross-model relationships..."):
                                try:
                                    cross_analyzer.build_cross_model_relationships()
                                    st.success("✅ Cross-model relationships built!")
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"Failed to build relationships: {e}")
                    
                    with col2:
                        if st.button("🚀 Force Full Rebuild", 
                                   help="Trigger complete AI rebuild of all models"):
                            with st.spinner("Running full AI rebuild..."):
                                try:
                                    startup_manager = get_startup_manager()
                                    startup_manager.force_rebuild()
                                    st.success("✅ Full rebuild completed!")
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"Rebuild failed: {e}")
                    
                    # Toggle auto rebuild
                    if enable_auto_rebuild:
                        if st.button("⏸️ Pause Auto Rebuild", help="Stop automatic AI processing"):
                            # Update .env to disable auto rebuild
                            st.info("💡 To pause auto rebuild, set ENABLE_AUTO_AI_REBUILD=0 in .env file")
                    else:
                        if st.button("▶️ Resume Auto Rebuild", help="Enable automatic AI processing"):
                            # Update .env to enable auto rebuild  
                            st.info("💡 To resume auto rebuild, set ENABLE_AUTO_AI_REBUILD=1 in .env file")
                
                else:
                    st.info("Add more models to enable cross-model comparison")
                    
            except Exception as e:
                st.error(f"Cross-model analysis error: {e}")
        else:
            st.warning("Cross-model analyzer not initialized")
        
        # Auto-Refresh System Status
        st.divider()
        st.subheader("Knowledge Base Monitor")
        
        # Start auto-refresh system
        start_auto_refresh_system()
        
        # Status indicator
        status_icon, status_text = get_refresh_status_indicator()
        st.markdown(f"{status_icon} **{status_text}**")
        
        # Manual refresh button
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Force Rebuild", help="Manually trigger knowledge base rebuild"):
                try:
                    startup_manager = get_startup_manager()
                    startup_manager.force_rebuild()
                    st.success("Rebuild triggered!")
                except Exception as e:
                    st.error(f"Rebuild failed: {e}")
        
        with col2:
            # Show system status
            try:
                startup_manager = get_startup_manager()
                status = startup_manager.get_status()
                if status.get('elapsed_time'):
                    st.caption(f"Runtime: {status['elapsed_time']:.1f}s")
                else:
                    st.caption("System initializing")
            except Exception:
                st.caption("Status unavailable")
        
        # System Status Section
        st.divider()
        st.subheader("System Status")
        
        # Get current startup manager status
        try:
            startup_manager = get_startup_manager()
            startup_status = startup_manager.get_status()
            
            # Show system readiness
            if startup_status['is_ready']:
                st.success("✅ **System Ready** - All components operational")
                
                # Show system statistics
                if st.session_state.rebuild_recommendation:
                    status = st.session_state.rebuild_recommendation
                    st.caption(f"📄 Knowledge base: {status.get('total_files', 'Ready')}")
                    
                # Show startup time
                if startup_status['elapsed_time'] > 0:
                    st.caption(f"⚡ Startup time: {startup_status['elapsed_time']:.1f}s")
                    
            else:
                # System still initializing
                st.info(f"🔄 **Initializing:** {startup_status['message']}")
                st.caption(f"⏱️ Elapsed: {startup_status['elapsed_time']:.1f}s")
                
        except Exception as e:
            st.warning(f"⚠️ Status check failed: {e}")
        
        # Auto-refresh system info
        st.caption("🤖 Automatic knowledge base management")
        st.caption("🔄 Real-time change detection & updates")
        
        # Explore Graph button
        st.divider()
        if st.button("🔍 Explore Graph", help="Interactive visualization of the knowledge graph"):
            st.session_state.show_graph_modal = True
        
        # Clean model selection - no memory controls for end users
    
    # Main Content Area
    st.title("Knowledge Counselor v4.0")
    st.caption("Hybrid Graph-RAG Intelligence System with Memory")
    
    # Add CSS for chat history styling and enhanced input
    st.markdown("""
    <style>
    /* Chat history styling */
    div[data-testid="stVerticalBlock"] > div:has(> hr) {
        max-height: 500px;
        overflow-y: auto;
    }
    
    /* Enhanced chat input styling - Claude-like */
    .stChatInput > div {
        background-color: #f8f9fa;
        border-radius: 12px;
        border: 2px solid #e0e7ff;
        transition: all 0.2s ease;
    }
    
    .stChatInput > div:focus-within {
        border-color: #4f46e5;
        box-shadow: 0 0 0 3px rgba(79, 70, 229, 0.1);
        background-color: #ffffff;
    }
    
    /* Make input area taller like Claude */
    .stChatInput textarea {
        min-height: 60px !important;
        max-height: 200px !important;
        font-size: 16px !important;
        line-height: 1.5 !important;
        padding: 12px 16px !important;
        border-radius: 8px !important;
        resize: vertical !important;
    }
    
    /* Input placeholder styling */
    .stChatInput textarea::placeholder {
        color: #9ca3af !important;
        font-style: italic;
    }
    
    /* Send button styling */
    .stChatInput button {
        background-color: #4f46e5 !important;
        border-radius: 8px !important;
        height: 40px !important;
        width: 40px !important;
        margin-left: 8px !important;
    }
    
    .stChatInput button:hover {
        background-color: #4338ca !important;
        transform: scale(1.05);
    }
    
    /* Overall app styling improvements */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 0rem;
        max-width: 900px;
    }
    
    /* Improved title styling */
    .main h1 {
        color: #1f2937;
        font-weight: 700;
        letter-spacing: -0.025em;
    }
    
    /* Caption styling */
    .main .caption {
        color: #6b7280;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }
    
    /* Chat messages styling */
    .stChatMessage {
        border-radius: 12px;
        margin-bottom: 1rem;
        border: 1px solid #e5e7eb;
    }
    
    /* Sidebar improvements */
    .css-1d391kg {
        background-color: #f9fafb;
    }
    
    /* Button improvements */
    .stButton > button {
        border-radius: 8px;
        border: 1px solid #d1d5db;
        transition: all 0.2s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    </style>
    """, unsafe_allow_html=True)
    
    
    # Get selected models for main content
    selected_models = st.session_state.selected_models
    
    # ---------- CHAT HISTORY ----------
    for turn in st.session_state.chat_log:
        with st.container():
            st.markdown(
                f"<div style='font-weight:600; color:#1E40AF;'>{turn['question']}</div>",
                unsafe_allow_html=True,
            )
            st.markdown(turn["answer"], unsafe_allow_html=True)

            # --- elegant sources display ---
            if turn["sources"]:
                with st.expander(f"Sources ({len(turn['sources'])})", expanded=False):
                    for path in sorted(set(turn["sources"])):
                        st.markdown(f"• `{path}`")

            st.markdown("<hr style='margin:12px 0 8px 0'>", unsafe_allow_html=True)
    
    # Show current question being processed (fix for blank screen issue)
    if st.session_state.query_running and st.session_state.last_question:
        # Show the question immediately when processing starts
        with st.container():
            st.markdown(
                f"<div style='font-weight:600; color:#1E40AF;'>{st.session_state.last_question}</div>",
                unsafe_allow_html=True,
            )
        
        st.markdown("<hr style='margin:12px 0 8px 0'>", unsafe_allow_html=True)
    
    # Show real-time progress when query is running - handled below in processing section
    
    # Claude-style input at bottom
    query = st.chat_input(
        "What would you like to know? Ask me about your knowledge base, code analysis, or anything else...",
        disabled=st.session_state.query_running or not selected_models
    )
    
    # Validation message
    if not selected_models:
        st.warning("Please select at least one model from the sidebar")
    
    # Process query
    if query and selected_models:
        st.session_state.query_running = True
        st.session_state.last_question = query
        
        # Check if this is a cross-model comparison query
        is_comparison_query = _detect_cross_model_query(query, selected_models)
        st.session_state.is_comparison_query = is_comparison_query
        
        # Reset for new query
        st.session_state.reasoning_steps = {}
        st.session_state.thinking_stream = []
        st.session_state.progress_messages = []
        for agent in ['PLAN', 'THOUGHT', 'ACTION', 'OBSERVATION']:
            st.session_state[f'{agent}_content'] = ''
        
        # Show immediate progress update
        st.rerun()  # Refresh to show processing state
    
    # Handle query processing after rerun
    if st.session_state.query_running and st.session_state.last_question:
        # Create a clean container for ChatGPT-style streaming
        thinking_container = st.container()
        
        with thinking_container:
            # World-class UI thinking header with proper spacing
            st.markdown("""
            <div style="
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif;
                font-size: 16px;
                font-weight: 600;
                color: #1f2937;
                margin-bottom: 16px;
                margin-top: 0;
            ">
            Thinking
            </div>
            """, unsafe_allow_html=True)
            
            # Create a placeholder for JavaScript-based streaming
            progress_placeholder = st.empty()
            
            # Execute analysis with real streaming
            result = create_real_streaming_interface(st.session_state.last_question, selected_models, thinking_container)
            
            # Complete processing
            st.session_state.query_running = False
            
            # Success notification
            st.success(f"Query completed with {result['confidence']:.1%} confidence")
            
            # Refresh to show results in chat history
            st.rerun()
    
    # Graph Explorer Modal
    if st.session_state.show_graph_modal:
        create_graph_modal(selected_models)

def create_graph_modal(selected_models: List[str]):
    """Create interactive graph visualization modal with Graph View 2.0."""
    
    # Import visualization components
    from core.graph_visualizer import render_graph, create_graph_legend, get_graph_statistics
    import streamlit.components.v1 as components
    
    # Create modal-style container  
    with st.container():
        # Modal header with close button
        col1, col2 = st.columns([4, 1])
        with col1:
            st.subheader("🔍 Knowledge Graph View 2.0")
        with col2:
            if st.button("✕ Close", key="close_graph_modal"):
                st.session_state.show_graph_modal = False
                st.rerun()
        
        st.divider()
        if not selected_models:
            st.warning("Please select at least one model to explore the graph.")
            return
        
        try:
            # Get hybrid system
            hybrid_runner, _ = get_hybrid_system()
            if not hybrid_runner or not hybrid_runner.graph:
                st.error("Graph not available. System may not be initialized.")
                return
            
            # Get graph and statistics
            graph = hybrid_runner.graph
            stats = get_graph_statistics(graph)
            
            # Display enhanced statistics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Nodes", stats['total_nodes'])
            with col2:
                st.metric("Edges", stats['total_edges'])
            with col3:
                st.metric("Density", f"{stats['density']:.3f}")
            with col4:
                st.metric("Connected", "Yes" if stats['is_connected'] else "No")
            
            # Enhanced graph visualization using graphviz with physics layout
            total_nodes = stats['total_nodes']
            if total_nodes > 0:
                # Create a force-directed graph layout instead of hierarchical
                dot_graph = "digraph G {\n"
                dot_graph += "  layout=fdp;\n"  # Force-directed layout for better clustering
                dot_graph += "  overlap=false;\n"
                dot_graph += "  splines=true;\n"
                dot_graph += "  node [shape=circle, style=filled];\n"
                
                # Show more nodes - use 40 instead of 20 for better representation
                display_limit = min(40, total_nodes)
                node_count = 0
                for node, data in list(graph.nodes(data=True))[:display_limit]:
                    node_type = data.get('type', 'unknown')
                    
                    # Set colors based on type (enhanced color scheme)
                    if node_type == 'MODEL':
                        color = "#FF6B6B"  # Red for models
                    elif node_type == 'DATASET':
                        color = "#34D399"  # Green for datasets
                    elif node_type == 'METRIC':
                        color = "#60A5FA"  # Blue for metrics
                    elif node_type == 'AUTHOR':
                        color = "#F59E0B"  # Orange for authors
                    elif node_type == 'FIGURE':
                        color = "#8B5CF6"  # Purple for figures
                    elif node_type == 'TABLE':
                        color = "#EC4899"  # Pink for tables
                    else:
                        color = "#E5E7EB"  # Default gray
                    
                    # Clean node name for display
                    clean_name = str(node).replace('"', '\\"')[:20]
                    dot_graph += f'  "{clean_name}" [fillcolor="{color}"];\n'
                    node_count += 1
                
                # Add sample edges
                edge_count = 0
                for source, target in list(graph.edges())[:30]:  # Limit edges
                    clean_source = str(source).replace('"', '\\"')[:20]
                    clean_target = str(target).replace('"', '\\"')[:20]
                    dot_graph += f'  "{clean_source}" -> "{clean_target}";\n'
                    edge_count += 1
                
                dot_graph += "}"
                
                # Display the graph
                st.graphviz_chart(dot_graph)
                
                # Controls
                st.write("**Graph Controls:**")
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("🔍 Zoom In"):
                        st.info("Use browser zoom controls")
                with col2:
                    if st.button("📐 Fit Graph"):
                        st.info("Graph fitted to view")
                
                if total_nodes > display_limit:
                    st.info(f"Showing first {display_limit} of {total_nodes} nodes for performance")
                
                # Add legend for node colors
                st.write("**Node Types:**")
                cols = st.columns(6)
                with cols[0]:
                    st.markdown("🔴 **MODEL**")
                with cols[1]:  
                    st.markdown("🟢 **DATASET**")
                with cols[2]:
                    st.markdown("🔵 **METRIC**")
                with cols[3]:
                    st.markdown("🟠 **AUTHOR**") 
                with cols[4]:
                    st.markdown("🟣 **FIGURE**")
                with cols[5]:
                    st.markdown("🩷 **TABLE**")
            else:
                st.warning("No graph data available")
                
        except Exception as e:
            st.error(f"Error creating graph visualization: {e}")
        
        # Additional close button at bottom
        if st.button("Close", type="primary", key="close_graph_modal_bottom"):
            st.session_state.show_graph_modal = False
            st.rerun()

if __name__ == "__main__":
    main()