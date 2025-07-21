#!/usr/bin/env python3
"""
Graph Visualization Module
Provides graph rendering and visualization capabilities for the Knowledge Counselor
"""

import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pickle
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from collections import Counter
import plotly.graph_objects as go
import plotly.express as px

logger = logging.getLogger(__name__)


def get_graph_statistics(selected_models: List[str]) -> Dict[str, Any]:
    """Get statistics for selected models' graphs (alias for compatibility)."""
    return get_graph_stats(selected_models)


def get_graph_stats(selected_models: List[str]) -> Dict[str, Any]:
    """Get statistics for selected models' graphs."""
    stats = {
        'total_nodes': 0,
        'total_edges': 0,
        'models': {},
        'node_types': Counter(),
        'edge_types': Counter()
    }
    
    for model_id in selected_models:
        graph_path = Path("enhanced_kg") / model_id / "graph" / "enhanced_graph.pkl"
        
        if graph_path.exists():
            try:
                with open(graph_path, 'rb') as f:
                    graph = pickle.load(f)
                
                model_stats = {
                    'nodes': graph.number_of_nodes(),
                    'edges': graph.number_of_edges(),
                    'node_types': Counter(),
                    'edge_types': Counter()
                }
                
                # Count node types
                for node_id in graph.nodes():
                    node_data = graph.nodes[node_id]
                    node_type = node_data.get('type', 'Unknown')
                    model_stats['node_types'][node_type] += 1
                    stats['node_types'][node_type] += 1
                
                # Count edge types
                for edge in graph.edges(data=True):
                    edge_type = edge[2].get('type', 'Unknown')
                    model_stats['edge_types'][edge_type] += 1
                    stats['edge_types'][edge_type] += 1
                
                stats['models'][model_id] = model_stats
                stats['total_nodes'] += model_stats['nodes']
                stats['total_edges'] += model_stats['edges']
                
            except Exception as e:
                logger.error(f"Failed to load graph for {model_id}: {e}")
                stats['models'][model_id] = {
                    'nodes': 0,
                    'edges': 0,
                    'error': str(e)
                }
    
    return stats


def create_graph_legend(node_types: Counter, edge_types: Counter) -> None:
    """Create a legend for graph visualization."""
    st.subheader("📊 Graph Legend")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Node Types:**")
        for node_type, count in node_types.most_common():
            color = get_node_color(node_type)
            st.write(f"🔸 {node_type}: {count}")
    
    with col2:
        st.write("**Edge Types:**")
        for edge_type, count in edge_types.most_common():
            st.write(f"🔗 {edge_type}: {count}")


def get_node_color(node_type: str) -> str:
    """Get color for node type."""
    color_map = {
        'Class': '#FF6B6B',
        'Function': '#4ECDC4',
        'Module': '#45B7D1',
        'Concept': '#96CEB4',
        'Configuration': '#FFEAA7',
        'File': '#DDA0DD',
        'CodeExample': '#98D8C8',
        'Unknown': '#95A5A6'
    }
    return color_map.get(node_type, '#95A5A6')


def render_graph_matplotlib(graph: nx.MultiDiGraph, max_nodes: int = 100) -> plt.Figure:
    """Render graph using matplotlib (for smaller graphs)."""
    # Sample nodes if graph is too large
    if graph.number_of_nodes() > max_nodes:
        # Get top nodes by degree
        node_degrees = dict(graph.degree())
        top_nodes = sorted(node_degrees.items(), key=lambda x: x[1], reverse=True)[:max_nodes]
        sampled_nodes = [node for node, _ in top_nodes]
        graph = graph.subgraph(sampled_nodes).copy()
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Position nodes using spring layout
    pos = nx.spring_layout(graph, k=1, iterations=50)
    
    # Group nodes by type for coloring
    node_types = {}
    for node_id in graph.nodes():
        node_data = graph.nodes[node_id]
        node_type = node_data.get('type', 'Unknown')
        if node_type not in node_types:
            node_types[node_type] = []
        node_types[node_type].append(node_id)
    
    # Draw nodes by type
    for node_type, nodes in node_types.items():
        color = get_node_color(node_type)
        nx.draw_networkx_nodes(
            graph, pos, 
            nodelist=nodes,
            node_color=color,
            node_size=300,
            alpha=0.8,
            ax=ax
        )
    
    # Draw edges
    nx.draw_networkx_edges(
        graph, pos,
        edge_color='gray',
        alpha=0.5,
        arrows=True,
        arrowsize=10,
        ax=ax
    )
    
    # Add labels for important nodes
    important_nodes = {}
    node_degrees = dict(graph.degree())
    top_nodes = sorted(node_degrees.items(), key=lambda x: x[1], reverse=True)[:20]
    
    for node_id, _ in top_nodes:
        node_data = graph.nodes[node_id]
        important_nodes[node_id] = node_data.get('name', str(node_id))[:15]
    
    nx.draw_networkx_labels(
        graph, pos,
        labels=important_nodes,
        font_size=8,
        ax=ax
    )
    
    ax.set_title(f'Knowledge Graph ({graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges)', 
                 fontsize=14, fontweight='bold')
    ax.axis('off')
    
    # Create legend
    legend_elements = []
    for node_type in node_types.keys():
        color = get_node_color(node_type)
        legend_elements.append(mpatches.Patch(color=color, label=node_type))
    
    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))
    
    plt.tight_layout()
    return fig


def render_graph_plotly(graph: nx.MultiDiGraph, max_nodes: int = 200) -> go.Figure:
    """Render graph using plotly (interactive)."""
    # Sample nodes if graph is too large
    if graph.number_of_nodes() > max_nodes:
        node_degrees = dict(graph.degree())
        top_nodes = sorted(node_degrees.items(), key=lambda x: x[1], reverse=True)[:max_nodes]
        sampled_nodes = [node for node, _ in top_nodes]
        graph = graph.subgraph(sampled_nodes).copy()
    
    # Position nodes
    pos = nx.spring_layout(graph, k=1, iterations=50)
    
    # Prepare node data
    node_x = []
    node_y = []
    node_text = []
    node_color = []
    node_size = []
    
    for node_id in graph.nodes():
        x, y = pos[node_id]
        node_x.append(x)
        node_y.append(y)
        
        node_data = graph.nodes[node_id]
        node_type = node_data.get('type', 'Unknown')
        node_name = node_data.get('name', str(node_id))
        
        # Node info for hover
        degree = graph.degree(node_id)
        node_text.append(f"{node_name}<br>Type: {node_type}<br>Connections: {degree}")
        
        # Color by type
        color_map = {
            'Class': 0, 'Function': 1, 'Module': 2, 'Concept': 3,
            'Configuration': 4, 'File': 5, 'CodeExample': 6, 'Unknown': 7
        }
        node_color.append(color_map.get(node_type, 7))
        
        # Size by degree
        node_size.append(max(10, min(50, degree * 3)))
    
    # Prepare edge data
    edge_x = []
    edge_y = []
    
    for edge in graph.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    
    # Create edge trace
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines'
    )
    
    # Create node trace
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        text=node_text,
        marker=dict(
            showscale=True,
            colorscale='Viridis',
            color=node_color,
            size=node_size,
            colorbar=dict(
                thickness=15,
                len=0.5,
                x=1.05,
                title="Node Type"
            ),
            line=dict(width=2)
        )
    )
    
    # Create figure
    fig = go.Figure(data=[edge_trace, node_trace],
                   layout=go.Layout(
                       title=f'Interactive Knowledge Graph ({graph.number_of_nodes()} nodes)',
                       titlefont_size=16,
                       showlegend=False,
                       hovermode='closest',
                       margin=dict(b=20,l=5,r=5,t=40),
                       annotations=[
                           dict(
                               text="Hover over nodes for details",
                               showarrow=False,
                               xref="paper", yref="paper",
                               x=0.005, y=-0.002,
                               xanchor="left", yanchor="bottom",
                               font=dict(color="#888", size=12)
                           )
                       ],
                       xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                       yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                   )
    
    return fig


def render_graph(selected_models: List[str], visualization_type: str = "plotly") -> Optional[Any]:
    """Main function to render graph for selected models."""
    if not selected_models:
        st.warning("Please select at least one model to visualize.")
        return None
    
    # Load and combine graphs
    combined_graph = nx.MultiDiGraph()
    
    for model_id in selected_models:
        graph_path = Path("enhanced_kg") / model_id / "graph" / "enhanced_graph.pkl"
        
        if graph_path.exists():
            try:
                with open(graph_path, 'rb') as f:
                    graph = pickle.load(f)
                
                # Add nodes with model prefix
                for node_id in graph.nodes():
                    new_node_id = f"{model_id}::{node_id}"
                    node_data = graph.nodes[node_id].copy()
                    node_data['model'] = model_id
                    combined_graph.add_node(new_node_id, **node_data)
                
                # Add edges with model prefix
                for edge in graph.edges(data=True):
                    new_source = f"{model_id}::{edge[0]}"
                    new_target = f"{model_id}::{edge[1]}"
                    combined_graph.add_edge(new_source, new_target, **edge[2])
                
                logger.info(f"Loaded graph for {model_id}: {graph.number_of_nodes()} nodes")
                
            except Exception as e:
                logger.error(f"Failed to load graph for {model_id}: {e}")
                st.error(f"Failed to load graph for {model_id}: {e}")
    
    if combined_graph.number_of_nodes() == 0:
        st.warning("No graph data found for selected models.")
        return None
    
    # Render based on type
    if visualization_type == "plotly":
        return render_graph_plotly(combined_graph)
    else:
        return render_graph_matplotlib(combined_graph)


def create_network_stats_dashboard(stats: Dict[str, Any]) -> None:
    """Create a dashboard showing network statistics."""
    st.subheader("📊 Network Statistics")
    
    # Overall stats
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Nodes", f"{stats['total_nodes']:,}")
    
    with col2:
        st.metric("Total Edges", f"{stats['total_edges']:,}")
    
    with col3:
        if stats['total_nodes'] > 0:
            density = stats['total_edges'] / (stats['total_nodes'] * (stats['total_nodes'] - 1))
            st.metric("Network Density", f"{density:.4f}")
        else:
            st.metric("Network Density", "0")
    
    with col4:
        node_types = len(stats['node_types'])
        st.metric("Node Types", node_types)
    
    # Per-model breakdown
    if len(stats['models']) > 1:
        st.subheader("📈 Per-Model Breakdown")
        
        model_data = []
        for model_id, model_stats in stats['models'].items():
            if 'error' not in model_stats:
                model_data.append({
                    'Model': model_id,
                    'Nodes': model_stats['nodes'],
                    'Edges': model_stats['edges']
                })
        
        if model_data:
            import pandas as pd
            df = pd.DataFrame(model_data)
            st.dataframe(df, use_container_width=True)
    
    # Node type distribution
    if stats['node_types']:
        st.subheader("🔸 Node Type Distribution")
        
        # Create pie chart
        labels = list(stats['node_types'].keys())
        values = list(stats['node_types'].values())
        
        fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.3)])
        fig.update_layout(
            title="Distribution of Node Types",
            annotations=[dict(text='Node Types', x=0.5, y=0.5, font_size=20, showarrow=False)]
        )
        st.plotly_chart(fig, use_container_width=True)


# Compatibility functions for the main app
def create_graph_visualization(selected_models: List[str]) -> None:
    """Create complete graph visualization interface."""
    st.subheader("🌐 Knowledge Graph Visualization")
    
    if not selected_models:
        st.info("Select models from the sidebar to visualize their knowledge graphs.")
        return
    
    # Get graph statistics
    stats = get_graph_stats(selected_models)
    
    # Create stats dashboard
    create_network_stats_dashboard(stats)
    
    # Visualization options
    viz_type = st.radio(
        "Visualization Type:",
        ["plotly", "matplotlib"],
        index=0,
        help="Plotly provides interactive graphs, matplotlib provides static graphs"
    )
    
    # Render graph
    if st.button("🔄 Generate Visualization"):
        with st.spinner("Generating graph visualization..."):
            if viz_type == "plotly":
                fig = render_graph(selected_models, "plotly")
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
            else:
                fig = render_graph(selected_models, "matplotlib")
                if fig:
                    st.pyplot(fig)
    
    # Create legend
    if stats['node_types'] or stats['edge_types']:
        create_graph_legend(stats['node_types'], stats['edge_types'])


if __name__ == "__main__":
    # Test the module
    test_models = ["model_1"]
    stats = get_graph_stats(test_models)
    print(f"Graph stats: {stats}")