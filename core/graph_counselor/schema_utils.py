"""
Simple dataclasses for Graph Counselor runtime
"""
from dataclasses import dataclass
from typing import List, Dict, Any, Optional


@dataclass
class AgentStep:
    """Represents a single step in the agent reasoning process"""
    step_type: str  # PLAN, THOUGHT, ACTION, OBSERVATION
    content: str
    timestamp: float
    
    
@dataclass
class AgentTrace:
    """Full reasoning trace from the agent"""
    steps: List[AgentStep]
    final_answer: str
    citations: List[str]
    confidence: float = 0.0
    

@dataclass
class QueryResult:
    """Result from querying the knowledge graph"""
    answer: str
    trace: List[Dict[str, Any]]
    citations: List[str]
    graph_subview: Any  # networkx.Graph
    confidence: float = 0.0
    

@dataclass
class Node:
    """Knowledge graph node"""
    id: str
    type: str
    label: str
    details: str
    page: Optional[int] = None
    namespace: Optional[str] = None
    

@dataclass
class Edge:
    """Knowledge graph edge"""
    source: str
    target: str
    type: str
    confidence: float
    source_page: Optional[int] = None