#!/usr/bin/env python3
"""
Graph View 2.0 - Readable, Interactive, Insightful
PyVis-based graph visualization with enhanced UX
"""
import textwrap
import networkx as nx
from pyvis.network import Network
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)

def render_graph(g: nx.MultiDiGraph, height_px: int = 750) -> str:
    """
    Render NetworkX graph as interactive PyVis HTML with enhanced UX.
    
    Args:
        g: NetworkX MultiDiGraph to visualize
        height_px: Height of the visualization canvas
        
    Returns:
        HTML string for embedding in Streamlit components.html
    """
    logger.info(f"🎨 Rendering graph with {g.number_of_nodes()} nodes, {g.number_of_edges()} edges")
    
    # 1. Canvas sizing & basic setup
    net = Network(
        height=f"{height_px}px",
        width="100%",
        bgcolor="#FFFFFF",
        font_color="#111",
        directed=True,
        notebook=False,
    )
    
    # Enhanced physics for better node distribution
    net.barnes_hut(
        gravity=-3000,           # Stronger repulsion prevents clustering
        central_gravity=0.3,     # Gentle pull toward center
        spring_length=230,       # Longer springs for breathing room
        spring_strength=0.04,    # Softer springs prevent rigidity
        damping=0.09            # Smooth movement, prevents oscillation
    )
    
    # Consistent colour palette matching schema
    colour_map = {
        "MODEL":    "#FF6B6B",   # coral red
        "DATASET":  "#34D399",   # mint green  
        "METRIC":   "#60A5FA",   # sky blue
        "FIGURE":   "#A78BFA",   # violet purple
        "TABLE":    "#EC4899",   # hot pink
        "AUTHOR":   "#F59E0B",   # amber orange
        "_default": "#D1D5DB",   # slate gray
    }
    
    # 2. Node creation with legible labels
    for n, attrs in g.nodes(data=True):
        node_type = attrs.get("type", "_default")
        
        # Create readable label with ellipsis for long names
        label = textwrap.shorten(str(n), width=22, placeholder="…")
        
        # Enhanced tooltip with node details
        node_name = attrs.get('name', str(n))
        node_details = attrs.get('description', attrs.get('details', '–'))
        tooltip = f"<b>{node_name}</b><br>{node_details}"
        
        # Dynamic sizing based on node degree (connectivity)
        degree = g.degree(n)
        node_size = 18 + min(4 * degree, 30)  # Cap at reasonable size
        
        net.add_node(
            n,
            label=label,
            title=tooltip,
            color=colour_map.get(node_type, colour_map["_default"]),
            size=node_size,
            font={'size': 16, 'color': '#333333'},
            borderWidth=2,
            borderWidthSelected=4
        )
    
    # 3. Edge styling & curved arrows
    for u, v, data in g.edges(data=True):
        edge_label = data.get("label", data.get("type", ""))
        edge_title = data.get("description", f"{edge_label}: {u} → {v}")
        
        # Edge thickness based on weight or importance
        edge_weight = data.get("weight", 1)
        edge_width = max(1.4, min(edge_weight * 2, 6))  # Scale between 1.4 and 6
        
        net.add_edge(
            u, v,
            title=edge_title,
            label=edge_label if len(edge_label) < 15 else "",  # Only show short labels
            width=edge_width,
            arrows="to",
            arrowStrikethrough=False,
            smooth={
                "type": "curvedCW", 
                "roundness": 0.15
            },
            color={'color': '#848484', 'highlight': '#FF6B6B'},
            selectionWidth=3
        )
    
    # 4. Quality-of-life UX tweaks
    # Left-panel physics controls
    net.show_buttons(filter_=['physics'])
    
    # Enhanced interaction options
    net.set_options("""
        const options = {
            physics: {
                stabilization: {iterations: 150},
                barnesHut: {
                    gravitationalConstant: -3000,
                    centralGravity: 0.3,
                    springLength: 230,
                    springConstant: 0.04,
                    damping: 0.09,
                    avoidOverlap: 0.1
                }
            },
            interaction: {
                hover: true,
                tooltipDelay: 120,
                navigationButtons: true,
                keyboard: true,
                multiselect: true,
                selectConnectedEdges: false
            },
            nodes: {
                borderWidth: 2,
                borderWidthSelected: 4,
                chosen: {
                    node: function(values, id, selected, hovering) {
                        values.borderWidth = 4;
                        values.color = '#FF6B6B';
                    }
                }
            },
            edges: {
                smooth: {
                    type: 'curvedCW',
                    roundness: 0.15
                },
                chosen: {
                    edge: function(values, id, selected, hovering) {
                        values.width = values.width * 2;
                        values.color = '#FF6B6B';
                    }
                }
            }
        }
    """)
    
    logger.info("✅ Graph rendering complete")
    return net.generate_html()

def create_graph_legend() -> str:
    """Create HTML legend for graph node types."""
    legend_html = """
    <div style="
        font-family: Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        margin-bottom: 12px;
        padding: 8px 12px;
        background: #f8fafc;
        border-radius: 6px;
        border: 1px solid #e2e8f0;
        font-size: 14px;
        line-height: 1.5;
    ">
        <div style="font-weight: 600; margin-bottom: 6px; color: #374151;">Node Types:</div>
        <div style="display: flex; flex-wrap: wrap; gap: 12px;">
            <span style="display: flex; align-items: center; gap: 4px;">
                <div style="width:12px; height:12px; border-radius:6px; background:#FF6B6B;"></div>
                <span style="color: #6b7280;">Model</span>
            </span>
            <span style="display: flex; align-items: center; gap: 4px;">
                <div style="width:12px; height:12px; border-radius:6px; background:#34D399;"></div>
                <span style="color: #6b7280;">Dataset</span>
            </span>
            <span style="display: flex; align-items: center; gap: 4px;">
                <div style="width:12px; height:12px; border-radius:6px; background:#60A5FA;"></div>
                <span style="color: #6b7280;">Metric</span>
            </span>
            <span style="display: flex; align-items: center; gap: 4px;">
                <div style="width:12px; height:12px; border-radius:6px; background:#A78BFA;"></div>
                <span style="color: #6b7280;">Figure</span>
            </span>
            <span style="display: flex; align-items: center; gap: 4px;">
                <div style="width:12px; height:12px; border-radius:6px; background:#EC4899;"></div>
                <span style="color: #6b7280;">Table</span>
            </span>
            <span style="display: flex; align-items: center; gap: 4px;">
                <div style="width:12px; height:12px; border-radius:6px; background:#F59E0B;"></div>
                <span style="color: #6b7280;">Author</span>
            </span>
        </div>
        <div style="margin-top: 6px; font-size: 12px; color: #9ca3af;">
            💡 Hover nodes for details • Drag to explore • Scroll to zoom • Use controls to toggle physics
        </div>
    </div>
    """
    return legend_html

def get_graph_statistics(g: nx.MultiDiGraph) -> Dict[str, int]:
    """Get comprehensive graph statistics for display."""
    node_types = {}
    for _, data in g.nodes(data=True):
        node_type = data.get('type', 'Unknown')
        node_types[node_type] = node_types.get(node_type, 0) + 1
    
    edge_types = {}
    for _, _, data in g.edges(data=True):
        edge_type = data.get('type', 'Unknown')
        edge_types[edge_type] = edge_types.get(edge_type, 0) + 1
    
    return {
        'total_nodes': g.number_of_nodes(),
        'total_edges': g.number_of_edges(),
        'node_types': node_types,
        'edge_types': edge_types,
        'density': nx.density(g),
        'is_connected': nx.is_weakly_connected(g) if g.is_directed() else nx.is_connected(g)
    }

def compute_doc_count(model_path) -> int:
    """
    Compute accurate document count for a model directory.
    Counts all files recursively, excluding hidden files.
    """
    from pathlib import Path
    model_path = Path(model_path)
    return sum(
        1 for f in model_path.rglob("*")
        if f.is_file() and not f.name.startswith('.')
    )