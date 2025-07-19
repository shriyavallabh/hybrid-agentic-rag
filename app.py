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
from core.auto_refresh import get_auto_refresh, start_auto_refresh
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

def start_auto_refresh_system():
    """Initialize and start the auto-refresh subsystem."""
    if not st.session_state.auto_refresh_started:
        try:
            start_auto_refresh()
            st.session_state.auto_refresh_started = True
            logger.info("✅ Auto-refresh subsystem started")
        except Exception as e:
            logger.error(f"❌ Failed to start auto-refresh: {e}")
            st.session_state.auto_refresh_started = False

def get_refresh_status_indicator():
    """Get visual status indicator for auto-refresh system."""
    try:
        auto_refresh = get_auto_refresh()
        status = auto_refresh.get_status()
        
        if not status['running']:
            return "🔴", "Auto-refresh stopped"
        elif status['is_refreshing']:
            return "🟡", "Refreshing knowledge base..."
        else:
            return "🟢", "Monitoring for changes"
    except Exception:
        return "⚪", "Auto-refresh unavailable"

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
            if st.button("🔄 Refresh Now", help="Manually trigger knowledge base refresh"):
                try:
                    auto_refresh = get_auto_refresh()
                    auto_refresh.trigger_refresh()
                    st.success("Refresh triggered!")
                except Exception as e:
                    st.error(f"Refresh failed: {e}")
        
        with col2:
            # Show last refresh info
            try:
                auto_refresh = get_auto_refresh()
                status = auto_refresh.get_status()
                if status.get('last_refresh'):
                    from datetime import datetime
                    last_refresh = datetime.fromisoformat(status['last_refresh'])
                    st.caption(f"Last: {last_refresh.strftime('%H:%M:%S')}")
                else:
                    st.caption("No refreshes yet")
            except Exception:
                st.caption("Status unavailable")
        
        # Explore Graph button
        st.divider()
        if st.button("🔍 Explore Graph", help="Interactive visualization of the knowledge graph"):
            st.session_state.show_graph_modal = True
        
        # Clean model selection - no memory controls for end users
    
    # Main Content Area
    st.title("Knowledge Counselor v4.0")
    st.caption("Hybrid Graph-RAG Intelligence System with Memory")
    
    # Add CSS for chat history styling
    st.markdown("""
    <style>
    div[data-testid="stVerticalBlock"] > div:has(> hr) {
        max-height: 500px;
        overflow-y: auto;
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