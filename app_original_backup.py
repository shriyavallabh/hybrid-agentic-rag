"""
Model Knowledge Counselor
A Streamlit app for querying banking models using Graph Counselor agents.
"""
import os
import time
import base64
from pathlib import Path
from typing import Dict, Any, Optional

import streamlit as st
import networkx as nx
from pyvis.network import Network
import tempfile
from dotenv import load_dotenv

from core.graph_builder import GraphBuilder
from core.agent_runner import AgentRunner

# Load environment variables
load_dotenv()

# Page config
st.set_page_config(
    page_title="Knowledge Counselor",
    page_icon="🦉",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for Material You theme
st.markdown("""
<style>
/* Import Google Fonts */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=Cascadia+Code:wght@400;700&display=swap');

/* Base theme */
:root {
    --primary-coral: #FF6B6B;
    --primary-mint: #4ECDC4;
    --primary-violet: #667EEA;
    --secondary-violet: #764BA2;
    --canvas: #F7F8FA;
    --surface: #FFFFFF;
    --mint-highlight: #D1FAE5;
    --text-primary: #1A202C;
    --text-secondary: #718096;
    --shadow: 0 1px 3px rgba(0,0,0,0.12), 0 1px 2px rgba(0,0,0,0.24);
    --shadow-hover: 0 4px 6px rgba(0,0,0,0.1), 0 2px 4px rgba(0,0,0,0.1);
}

/* Global styles */
.main .block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1200px;
}

.stApp {
    background-color: var(--canvas);
    font-family: 'Inter', sans-serif;
}

/* Header */
.header-container {
    background: var(--surface);
    height: 80px;
    padding: 0 2rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    box-shadow: var(--shadow);
    margin: -2rem -2rem 2rem -2rem;
    border-radius: 0 0 16px 16px;
}

.header-left {
    display: flex;
    align-items: center;
    gap: 1rem;
}

.owl-icon {
    width: 56px;
    height: 56px;
    background: linear-gradient(135deg, var(--primary-violet), var(--secondary-violet));
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 24px;
    color: white;
}

.header-title {
    font-size: 24px;
    font-weight: 600;
    color: var(--text-primary);
    margin: 0;
}

.status-dots {
    display: flex;
    gap: 8px;
}

.status-dot {
    width: 12px;
    height: 12px;
    border-radius: 50%;
    transition: all 0.3s ease;
}

.status-green { background-color: #10B981; }
.status-amber { background-color: #F59E0B; }
.status-red { background-color: #EF4444; }

/* Cards */
.card {
    background: var(--surface);
    border-radius: 16px;
    padding: 1.5rem;
    box-shadow: var(--shadow);
    margin-bottom: 1rem;
    transition: all 0.15s ease;
}

.card:hover {
    transform: translateY(-2px);
    box-shadow: var(--shadow-hover);
    border: 2px solid var(--primary-mint);
}

.card-title {
    font-size: 18px;
    font-weight: 600;
    color: var(--text-primary);
    margin-bottom: 1rem;
}

/* Models card */
.model-item {
    padding: 0.75rem;
    border-radius: 8px;
    margin-bottom: 0.5rem;
    cursor: pointer;
    transition: all 0.2s ease;
    border: 1px solid transparent;
}

.model-item:hover {
    background-color: var(--mint-highlight);
}

.model-item.highlighted {
    background-color: var(--mint-highlight);
    border-color: var(--primary-mint);
}

/* Citation pills */
.citation-pill {
    display: inline-block;
    background: var(--primary-coral);
    color: white;
    padding: 0.25rem 0.75rem;
    border-radius: 12px;
    font-size: 12px;
    margin: 0.25rem;
    font-weight: 500;
}

/* Reasoning chips */
.reasoning-chip {
    display: inline-block;
    padding: 0.5rem 1rem;
    border-radius: 8px;
    font-size: 14px;
    font-weight: 500;
    margin: 0.25rem;
    color: white;
}

.chip-plan { background: var(--primary-violet); }
.chip-thought { background: var(--primary-coral); }
.chip-action { background: var(--primary-mint); color: var(--text-primary); }
.chip-observation { background: #10B981; }

/* Bottom sheet */
.bottom-sheet {
    position: fixed;
    bottom: 0;
    left: 0;
    right: 0;
    height: 25vh;
    background: var(--surface);
    border-radius: 16px 16px 0 0;
    box-shadow: 0 -4px 6px rgba(0,0,0,0.1);
    padding: 1.5rem;
    transform: translateY(100%);
    transition: transform 0.3s ease;
    z-index: 1000;
}

.bottom-sheet.open {
    transform: translateY(0);
}

.fab {
    position: fixed;
    bottom: 2rem;
    right: 2rem;
    width: 56px;
    height: 56px;
    background: linear-gradient(135deg, var(--primary-coral), #FF8E8E);
    border-radius: 50%;
    border: none;
    cursor: pointer;
    box-shadow: var(--shadow-hover);
    display: flex;
    align-items: center;
    justify-content: center;
    color: white;
    font-size: 24px;
    z-index: 1001;
    transition: all 0.3s ease;
}

.fab:hover {
    transform: scale(1.1);
}

/* Answer display */
.answer-box {
    background: var(--canvas);
    border-radius: 12px;
    padding: 1rem;
    margin-bottom: 1rem;
    border-left: 4px solid var(--primary-mint);
}

.unverifiable {
    border-left-color: var(--primary-coral);
    background: #FEF2F2;
}

/* Responsive */
@media (max-width: 1024px) {
    .main .block-container {
        padding-left: 1rem;
        padding-right: 1rem;
    }
    
    .header-container {
        margin-left: -1rem;
        margin-right: -1rem;
        padding: 0 1rem;
    }
}

/* Code font for technical content */
code, pre, .code-content {
    font-family: 'Cascadia Code', monospace;
}
</style>
""", unsafe_allow_html=True)


def create_owl_icon():
    """Create an SVG owl icon."""
    return """
    <svg width="56" height="56" viewBox="0 0 56 56" fill="none">
        <circle cx="28" cy="28" r="28" fill="url(#gradient)"/>
        <circle cx="20" cy="22" r="4" fill="white"/>
        <circle cx="36" cy="22" r="4" fill="white"/>
        <circle cx="20" cy="22" r="2" fill="#667EEA"/>
        <circle cx="36" cy="22" r="2" fill="#667EEA"/>
        <path d="M22 32 Q28 38 34 32" stroke="white" stroke-width="2" fill="none"/>
        <defs>
            <linearGradient id="gradient" x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" style="stop-color:#667EEA"/>
                <stop offset="100%" style="stop-color:#764BA2"/>
            </linearGradient>
        </defs>
    </svg>
    """


@st.cache_resource
def init_system():
    """Initialize the system and check dependencies."""
    demo_mode = os.getenv("DEMO_MODE", "0") == "1"
    
    if demo_mode:
        # Load demo bundle (to be created)
        kg_path = "demo_bundle"
        if not Path(kg_path).exists():
            st.error("Demo mode enabled but demo_bundle not found")
            st.stop()
    else:
        kg_path = "kg_bundle"
    
    try:
        # Initialize agent runner
        runner = AgentRunner(
            kg_path=kg_path,
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            token_budget=int(os.getenv("TOKEN_BUDGET", 200000))
        )
        return runner, demo_mode
    except Exception as e:
        st.error(f"System initialization failed: {e}")
        st.stop()


def build_or_refresh_kg():
    """Build or refresh the knowledge graph."""
    if os.getenv("DEMO_MODE", "0") == "1":
        st.info("Demo mode - skipping KG build")
        return
    
    with st.spinner("Building knowledge graph..."):
        try:
            builder = GraphBuilder(
                knowledge_base_path="knowledge_base",
                output_path="kg_bundle",
                openai_api_key=os.getenv("OPENAI_API_KEY")
            )
            manifest = builder.build()
            st.success(f"KG built: {manifest['node_count']} nodes, {manifest['edge_count']} edges")
            st.experimental_rerun()
        except Exception as e:
            st.error(f"KG build failed: {e}")


def render_header(status: Dict[str, Any]):
    """Render the header with status dots."""
    # Determine status dot colors
    dot1_color = "status-green" if status.get("faiss_loaded") else "status-red"
    dot2_color = "status-green" if status.get("graph_loaded") else "status-red" 
    dot3_color = "status-green" if status.get("llm_connected") else "status-red"
    
    if status.get("budget_exceeded"):
        dot3_color = "status-red"
    
    header_html = f"""
    <div class="header-container">
        <div class="header-left">
            <div class="owl-icon">
                {create_owl_icon()}
            </div>
            <h1 class="header-title">Knowledge Counselor</h1>
        </div>
        <div class="status-dots">
            <div class="status-dot {dot1_color}" title="Vector Store"></div>
            <div class="status-dot {dot2_color}" title="Knowledge Graph"></div>
            <div class="status-dot {dot3_color}" title="LLM Status"></div>
        </div>
    </div>
    """
    st.markdown(header_html, unsafe_allow_html=True)


def render_models_card(status: Dict[str, Any], highlighted_models: set = None):
    """Render the models card."""
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<h3 class="card-title">📊 Models</h3>', unsafe_allow_html=True)
    
    # Search field
    search_term = st.text_input("Search models...", key="model_search", label_visibility="collapsed")
    
    # Model list (simplified for now)
    models = ["model_1", "fraud_score_v3", "loan_pd_model"]  # Will be dynamic later
    
    st.markdown('<div style="height: 200px; overflow-y: auto;">', unsafe_allow_html=True)
    for model in models:
        if not search_term or search_term.lower() in model.lower():
            highlight_class = "highlighted" if highlighted_models and model in highlighted_models else ""
            st.markdown(f'''
                <div class="model-item {highlight_class}">
                    <strong>{model}</strong><br>
                    <small style="color: var(--text-secondary);">Banking model • {status.get("node_count", 0)} entities</small>
                </div>
            ''', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


def render_references_card(citations: list = None):
    """Render the references card."""
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<h3 class="card-title">📚 References</h3>', unsafe_allow_html=True)
    
    if citations:
        st.markdown('<div style="margin-bottom: 1rem;">', unsafe_allow_html=True)
        for citation in citations:
            filename = Path(citation).name if citation else "Unknown"
            st.markdown(f'<span class="citation-pill">{filename}</span>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        if st.button("📄 Download Answer + Reasoning as PDF", key="download_pdf"):
            st.info("PDF download feature coming soon!")
    else:
        st.markdown('<p style="color: var(--text-secondary);">No references available</p>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)


def render_reasoning_card(trace: list = None, graph_subview: nx.Graph = None):
    """Render the reasoning card."""
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<h3 class="card-title">🧠 Reasoning</h3>', unsafe_allow_html=True)
    
    if trace:
        # Reasoning chips
        st.markdown('<div style="margin-bottom: 1rem;">', unsafe_allow_html=True)
        step_counts = {"PLAN": 0, "THOUGHT": 0, "ACTION": 0, "OBSERVATION": 0}
        for step in trace:
            step_type = step.get("type", "").upper()
            if step_type in step_counts:
                step_counts[step_type] += 1
        
        for step_type, count in step_counts.items():
            if count > 0:
                chip_class = f"chip-{step_type.lower()}"
                st.markdown(f'<span class="reasoning-chip {chip_class}">{step_type} ({count})</span>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Mini graph thumbnail
        if graph_subview and graph_subview.number_of_nodes() > 0:
            st.markdown("**Graph Traversal:**")
            
            # Create simple network visualization
            net = Network(height="200px", width="100%", bgcolor="#F7F8FA")
            
            # Add nodes
            for node_id, node_data in graph_subview.nodes(data=True):
                label = node_data.get('label', node_id)[:20]
                node_type = node_data.get('type', 'Unknown')
                color = {
                    'Model': '#667EEA',
                    'Dataset': '#4ECDC4',
                    'Metric': '#FF6B6B',
                    'CodeEntity': '#10B981'
                }.get(node_type, '#9CA3AF')
                
                net.add_node(node_id, label=label, color=color, title=f"{node_type}: {label}")
            
            # Add edges
            for source, target, edge_data in graph_subview.edges(data=True):
                edge_type = edge_data.get('type', '')
                net.add_edge(source, target, title=edge_type)
            
            # Save and display
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp_file:
                    net.save_graph(tmp_file.name)
                    with open(tmp_file.name, 'r') as f:
                        html_content = f.read()
                    st.components.v1.html(html_content, height=220)
                    os.unlink(tmp_file.name)
            except Exception as e:
                st.error(f"Graph visualization error: {e}")
        
        # Full screen reasoning button
        if st.button("🔍 View Full Reasoning Trace", key="full_reasoning"):
            st.session_state.show_reasoning_modal = True
    else:
        st.markdown('<p style="color: var(--text-secondary);">No reasoning trace available</p>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)


def render_qa_interface(runner: AgentRunner):
    """Render the Q&A interface."""
    st.markdown("## 💬 Ask About Your Models")
    
    # Display current answer if available
    if "current_answer" in st.session_state:
        answer = st.session_state.current_answer
        citations = st.session_state.get("current_citations", [])
        
        # Answer box
        unverifiable_class = "unverifiable" if answer.startswith("[UNVERIFIABLE]") else ""
        st.markdown(f'''
            <div class="answer-box {unverifiable_class}">
                <strong>Answer:</strong><br>
                {answer}
            </div>
        ''', unsafe_allow_html=True)
        
        if citations:
            st.caption(f"Sources: {', '.join([Path(c).name for c in citations])}")
    
    # Question input
    question = st.text_input(
        "Ask about your models...",
        placeholder="Which dataset does the loan PD model use?",
        key="question_input"
    )
    
    col1, col2 = st.columns([1, 4])
    with col1:
        ask_button = st.button("🔍 Ask", type="primary", use_container_width=True)
    
    if ask_button and question.strip():
        with st.spinner("Agent is thinking..."):
            try:
                result = runner.query(question)
                
                # Store results in session state
                st.session_state.current_answer = result.answer
                st.session_state.current_citations = result.citations
                st.session_state.current_trace = result.trace
                st.session_state.current_graph = result.graph_subview
                
                st.experimental_rerun()
                
            except Exception as e:
                st.error(f"Query failed: {e}")


def main():
    """Main application."""
    # Initialize system
    runner, demo_mode = init_system()
    status = runner.get_status()
    
    # Check if KG needs building
    if not demo_mode and not status.get("graph_loaded"):
        st.warning("Knowledge graph not found. Please build it first.")
        if st.button("🔨 Build Knowledge Graph", type="primary"):
            build_or_refresh_kg()
        st.stop()
    
    # Render header
    render_header(status)
    
    # Main content area
    col1, col2, col3 = st.columns([1, 1, 1])
    
    # Get highlighted models from current answer
    highlighted_models = set()
    if "current_citations" in st.session_state:
        for citation in st.session_state.current_citations:
            # Extract model name from path
            parts = Path(citation).parts
            for part in parts:
                if part.startswith("model_"):
                    highlighted_models.add(part)
    
    with col1:
        render_models_card(status, highlighted_models)
    
    with col2:
        citations = st.session_state.get("current_citations", [])
        render_references_card(citations)
    
    with col3:
        trace = st.session_state.get("current_trace", [])
        graph_subview = st.session_state.get("current_graph")
        render_reasoning_card(trace, graph_subview)
    
    # Q&A Interface
    st.markdown("---")
    render_qa_interface(runner)
    
    # Refresh KG button in sidebar
    with st.sidebar:
        st.markdown("### System")
        if st.button("🔄 Refresh Knowledge Graph"):
            build_or_refresh_kg()
        
        st.markdown(f"""
        **Status:**
        - Nodes: {status.get('node_count', 0)}
        - Edges: {status.get('edge_count', 0)}
        - Tokens used: {status.get('token_usage', 0):,}
        - Remaining: {status.get('token_remaining', 0):,}
        """)
        
        if demo_mode:
            st.info("🚀 Demo Mode Active")
    
    # Reasoning modal
    if st.session_state.get("show_reasoning_modal"):
        st.markdown("### Full Reasoning Trace")
        trace = st.session_state.get("current_trace", [])
        
        for i, step in enumerate(trace):
            step_type = step.get("type", "").upper()
            content = step.get("content", "")
            
            with st.expander(f"{i+1}. {step_type}", expanded=i==0):
                st.markdown(f"**{step_type}:** {content}")
        
        if st.button("Close"):
            st.session_state.show_reasoning_modal = False
            st.experimental_rerun()


if __name__ == "__main__":
    main()