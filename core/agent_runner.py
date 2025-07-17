"""
Agent Runner
Wraps Graph Counselor 4-agent loop for knowledge graph querying.
"""
import os
import json
import pickle
import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple
import time

import networkx as nx
import faiss
import numpy as np
from openai import OpenAI
import tiktoken

from .graph_counselor.schema_utils import QueryResult, AgentStep, AgentTrace

logger = logging.getLogger(__name__)


class TokenTracker:
    """Track OpenAI API token usage and enforce budget."""
    
    def __init__(self, budget: int, state_path: str = "runtime_state/token_usage.json"):
        self.budget = budget
        self.state_path = Path(state_path)
        self.state_path.parent.mkdir(exist_ok=True)
        self.tokenizer = tiktoken.encoding_for_model("gpt-4o")
        
    def _load_usage(self) -> Dict[str, Any]:
        """Load token usage state."""
        if self.state_path.exists():
            with open(self.state_path, 'r') as f:
                return json.load(f)
        return {"month": "", "tokens": 0}
    
    def _save_usage(self, usage: Dict[str, Any]):
        """Save token usage state."""
        with open(self.state_path, 'w') as f:
            json.dump(usage, f, indent=2)
    
    def get_current_usage(self) -> Tuple[int, int]:
        """Get current month usage and remaining budget."""
        usage = self._load_usage()
        current_month = time.strftime("%Y-%m")
        
        if usage["month"] != current_month:
            usage = {"month": current_month, "tokens": 0}
            self._save_usage(usage)
        
        return usage["tokens"], self.budget - usage["tokens"]
    
    def add_usage(self, tokens: int):
        """Add token usage and check budget."""
        usage = self._load_usage()
        current_month = time.strftime("%Y-%m")
        
        if usage["month"] != current_month:
            usage = {"month": current_month, "tokens": 0}
        
        usage["tokens"] += tokens
        self._save_usage(usage)
        
        if usage["tokens"] > self.budget:
            raise RuntimeError(f"Token budget exceeded: {usage['tokens']}/{self.budget}")
    
    def estimate_tokens(self, text: str) -> int:
        """Estimate token count for text."""
        return len(self.tokenizer.encode(text))


class GraphAgentSimplified:
    """Simplified Graph Counselor agent for our use case."""
    
    def __init__(self, client: OpenAI, graph: nx.MultiDiGraph, token_tracker: TokenTracker):
        self.client = client
        self.graph = graph
        self.token_tracker = token_tracker
        
    def plan(self, question: str, context_nodes: List[str]) -> str:
        """Create a plan for answering the question."""
        context_info = []
        for node_id in context_nodes[:15]:  # Increased limit for more context
            node_data = self.graph.nodes.get(node_id, {})
            name = node_data.get('name', node_data.get('title', node_id))
            details = node_data.get('description', '')
            context_info.append(f"- {name}: {details}")
        
        context_str = "\\n".join(context_info) if context_info else "No relevant context found."
        
        prompt = f"""Create a step-by-step plan to answer this question about banking models:

Question: {question}

Available context:
{context_str}

Plan (2-3 specific steps):"""
        
        return self._call_llm(prompt, "PLAN")
    
    def thought(self, question: str, plan: str, context: List[str]) -> str:
        """Generate thoughts about the current state."""
        prompt = f"""Given this question and plan, what should we think about?

Question: {question}
Plan: {plan}

Current context: {', '.join(context[:5])}

Thought (1-2 sentences):"""
        
        return self._call_llm(prompt, "THOUGHT")
    
    def action(self, question: str, plan: str, thought: str) -> Tuple[str, List[str]]:
        """Decide what action to take and execute it."""
        prompt = f"""What specific action should we take to progress on this plan?

Question: {question}
Plan: {plan}
Current thought: {thought}

Choose an action:
1. Search for specific nodes/entities
2. Explore relationships between entities  
3. Summarize findings and conclude

Action:"""
        
        action_text = self._call_llm(prompt, "ACTION")
        
        # Simple action execution - return relevant nodes
        relevant_nodes = self._find_relevant_nodes(question)
        return action_text, relevant_nodes
    
    def observation(self, action: str, nodes: List[str]) -> str:
        """Make observations based on action results."""
        node_details = []
        edges_found = []
        
        for node_id in nodes[:10]:  # Increased limit for better observations
            node_data = self.graph.nodes.get(node_id, {})
            name = node_data.get('name', node_data.get('title', node_id))
            details = node_data.get('description', '')
            node_details.append(f"- {name}: {details}")
            
            # Check edges
            for edge in list(self.graph.edges(node_id, data=True))[:3]:
                source, target, edge_data = edge
                edges_found.append(f"  {edge_data.get('type', 'RELATED')} -> {self.graph.nodes.get(target, {}).get('label', target)}")
        
        observations = "\\n".join(node_details) if node_details else "No relevant nodes found."
        if edges_found:
            observations += "\\n\\nRelationships:\\n" + "\\n".join(edges_found)
        
        return observations
    
    def reflection(self, question: str, trace: List[AgentStep]) -> str:
        """Reflect on the reasoning process and suggest improvements."""
        trace_summary = "\\n".join([f"{step.step_type}: {step.content[:100]}..." for step in trace[-3:]])
        
        prompt = f"""Review this reasoning trace for the question: {question}

Recent steps:
{trace_summary}

Is the reasoning on track? If not, suggest a corrected approach. If yes, respond "CONTINUE".

Reflection:"""
        
        return self._call_llm(prompt, "REFLECTION")
    
    def _find_relevant_nodes(self, question: str) -> List[str]:
        """Find nodes relevant to the question."""
        question_lower = question.lower()
        relevant_nodes = []
        
        # If asking about authors, find ALL author nodes
        if 'author' in question_lower:
            for node_id, node_data in self.graph.nodes(data=True):
                if node_data.get('type') == 'Author':
                    relevant_nodes.append(node_id)
            return relevant_nodes  # Return all authors
        
        # Otherwise, general search
        for node_id, node_data in self.graph.nodes(data=True):
            name = node_data.get('name', node_data.get('title', '')).lower()
            description = node_data.get('description', '').lower()
            
            # Simple keyword matching
            if any(word in name or word in description for word in question_lower.split() if len(word) > 3):
                relevant_nodes.append(node_id)
        
        return relevant_nodes[:15]  # Increased limit
    
    def _call_llm(self, prompt: str, step_type: str) -> str:
        """Make LLM call with token tracking."""
        estimated_tokens = self.token_tracker.estimate_tokens(prompt) + 150  # Response estimate
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                temperature=0.0,
                max_tokens=150,
                messages=[{"role": "user", "content": prompt}]
            )
            
            result = response.choices[0].message.content.strip()
            
            # Track actual usage
            actual_tokens = response.usage.total_tokens
            self.token_tracker.add_usage(actual_tokens)
            
            return result
            
        except Exception as e:
            logger.error(f"LLM call failed for {step_type}: {e}")
            return f"[Error in {step_type}]"


class AgentRunner:
    """Main agent runner that loads KG and executes queries."""
    
    def __init__(self, kg_path: str, openai_api_key: str, token_budget: int = 200000):
        self.kg_path = Path(kg_path)
        self.client = OpenAI(api_key=openai_api_key)
        self.token_tracker = TokenTracker(budget=token_budget)
        
        # Load graph and index
        self.graph = None
        self.faiss_index = None
        self.node_mapping = None
        self._load_knowledge_graph()
        
    def _load_knowledge_graph(self):
        """Load the knowledge graph and FAISS index."""
        try:
            # Load graph
            graph_path = self.kg_path / "graph.pkl"
            with open(graph_path, 'rb') as f:
                self.graph = pickle.load(f)
            
            # Load FAISS index
            index_path = self.kg_path / "faiss.index"
            self.faiss_index = faiss.read_index(str(index_path))
            
            # Load node mapping
            mapping_path = self.kg_path / "node_mapping.pkl"
            with open(mapping_path, 'rb') as f:
                self.node_mapping = pickle.load(f)
                
            logger.info(f"Loaded KG: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")
            
        except Exception as e:
            logger.error(f"Failed to load knowledge graph: {e}")
            raise
    
    def _retrieve_seed_nodes(self, question: str, k: int = 10) -> List[str]:
        """Retrieve top-k seed nodes using FAISS similarity search."""
        try:
            logger.info(f"🔍 Starting retrieval for: {question}")
            logger.info(f"📊 FAISS index loaded: {self.faiss_index is not None}")
            logger.info(f"🗂️ Node mapping size: {len(self.node_mapping) if self.node_mapping else 0}")
            
            # Embed the question
            logger.info("🔢 Creating embeddings...")
            response = self.client.embeddings.create(
                model="text-embedding-3-small",
                input=[question]
            )
            
            query_embedding = np.array([response.data[0].embedding], dtype=np.float32)
            logger.info(f"✅ Embedding created with shape: {query_embedding.shape}")
            
            faiss.normalize_L2(query_embedding)
            logger.info("✅ Embedding normalized")
            
            # Search FAISS index
            logger.info(f"🔎 Searching FAISS index for top {k} nodes...")
            scores, indices = self.faiss_index.search(query_embedding, k)
            logger.info(f"✅ Search completed - scores: {scores[0][:3]}, indices: {indices[0][:3]}")
            
            # Map back to node IDs
            seed_nodes = []
            for i, idx in enumerate(indices[0]):
                if idx < len(self.node_mapping):
                    node_id = self.node_mapping[idx]
                    seed_nodes.append(node_id)
                    logger.info(f"📍 Mapped index {idx} to node: {node_id} (score: {scores[0][i]:.3f})")
            
            logger.info(f"✅ Retrieved {len(seed_nodes)} seed nodes: {seed_nodes}")
            return seed_nodes
            
        except Exception as e:
            logger.error(f"❌ Retrieval failed: {e}")
            logger.error(f"💥 Exception type: {type(e).__name__}")
            logger.error(f"💥 Exception details: {str(e)}")
            import traceback
            logger.error(f"💥 Full traceback: {traceback.format_exc()}")
            return []
    
    def query(self, question: str, max_steps: int = 6) -> QueryResult:
        """Execute the 4-agent loop to answer a question."""
        logger.info(f"🚀 Starting query execution for: {question}")
        
        if not self.graph:
            logger.error("❌ Knowledge graph not loaded")
            raise RuntimeError("Knowledge graph not loaded")
        
        # Check token budget
        current_usage, remaining = self.token_tracker.get_current_usage()
        logger.info(f"💰 Token budget: {remaining} remaining / {current_usage} used")
        if remaining <= 100:
            logger.error("❌ Token budget exceeded - refill required")
            raise RuntimeError("Token budget exceeded - refill required")
        
        start_time = time.time()
        
        # Retrieve seed nodes
        seed_nodes = self._retrieve_seed_nodes(question)
        logger.info(f"✅ Retrieved {len(seed_nodes)} seed nodes: {seed_nodes}")
        
        # Initialize agent
        agent = GraphAgentSimplified(self.client, self.graph, self.token_tracker)
        
        # Execute 4-agent loop
        trace = []
        citations = set()
        
        # 1. PLAN
        plan = agent.plan(question, seed_nodes)
        trace.append(AgentStep("PLAN", plan, time.time() - start_time))
        
        context_nodes = seed_nodes
        
        for step in range(max_steps):
            # 2. THOUGHT
            thought = agent.thought(question, plan, context_nodes)
            trace.append(AgentStep("THOUGHT", thought, time.time() - start_time))
            
            # 3. ACTION
            action, action_nodes = agent.action(question, plan, thought)
            trace.append(AgentStep("ACTION", action, time.time() - start_time))
            context_nodes.extend(action_nodes)
            
            # 4. OBSERVATION
            observation = agent.observation(action, action_nodes)
            trace.append(AgentStep("OBSERVATION", observation, time.time() - start_time))
            
            # Collect citations from observed nodes
            for node_id in action_nodes:
                node_data = self.graph.nodes.get(node_id, {})
                node_type = node_data.get('type', '')
                # Add citations based on node type
                if node_type == 'Author':
                    citations.add('knowledge_base/model_1/graphrag_model_doc.pdf')
                elif node_type == 'Model':
                    citations.add('knowledge_base/model_1/graphrag_model_doc.pdf')
                elif node_type == 'Organization':
                    citations.add('knowledge_base/model_1/graphrag_model_doc.pdf')
            
            # 5. SELF-REFLECTION
            if step < max_steps - 1:  # Don't reflect on last step
                reflection = agent.reflection(question, trace)
                if "CONTINUE" not in reflection.upper():
                    # Agent suggests correction
                    plan = reflection  # Use reflection as new plan
                    trace.append(AgentStep("REFLECTION", reflection, time.time() - start_time))
        
        # Generate final answer
        final_context = "\\n".join([
            step.content for step in trace 
            if step.step_type in ["OBSERVATION", "THOUGHT"]
        ])[-2000:]  # Limit context
        
        answer_prompt = f"""Based on this reasoning trace, provide a final answer to the question:

Question: {question}

Context from reasoning:
{final_context}

Final Answer:"""
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                temperature=0.0,
                max_tokens=300,
                messages=[{"role": "user", "content": answer_prompt}]
            )
            final_answer = response.choices[0].message.content.strip()
            self.token_tracker.add_usage(response.usage.total_tokens)
        except Exception as e:
            logger.error(f"Final answer generation failed: {e}")
            final_answer = "Unable to generate final answer due to error."
        
        # Create subgraph of traversed nodes
        subgraph_nodes = list(set(context_nodes))[:20]  # Limit size
        subgraph = self.graph.subgraph(subgraph_nodes).copy()
        
        # Flag as unverifiable if no citations
        if not citations:
            final_answer = f"[UNVERIFIABLE] {final_answer}"
        
        return QueryResult(
            answer=final_answer,
            trace=[{
                "type": step.step_type,
                "content": step.content,
                "timestamp": step.timestamp
            } for step in trace],
            citations=list(citations),
            graph_subview=subgraph,
            confidence=0.8 if citations else 0.3
        )
    
    def get_status(self) -> Dict[str, Any]:
        """Get system status for UI."""
        current_usage, remaining = self.token_tracker.get_current_usage()
        
        return {
            "graph_loaded": self.graph is not None,
            "faiss_loaded": self.faiss_index is not None,
            "llm_connected": True,  # Assume connected if we got here
            "node_count": self.graph.number_of_nodes() if self.graph else 0,
            "edge_count": self.graph.number_of_edges() if self.graph else 0,
            "token_usage": current_usage,
            "token_remaining": remaining,
            "budget_exceeded": remaining <= 0
        }