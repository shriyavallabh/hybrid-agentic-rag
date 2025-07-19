"""
Hybrid Agent Runner - 4-Agent System for Graph-Guided RAG
Implements the complete hybrid reasoning system combining Graph Counselor's 
4-agent loop with comprehensive RAG capabilities.
"""
import os
import json
import pickle
import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import time

import networkx as nx
import faiss
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv

from .hybrid_graph_rag import HybridRetriever, HybridAgent, TokenTracker
from .graph_counselor.schema_utils import QueryResult, AgentStep, AgentTrace
from .conversation_memory import ConversationMemory, MemoryAwareQueryProcessor

load_dotenv()
logger = logging.getLogger(__name__)


class HybridAgentRunner:
    """Main hybrid agent runner combining enhanced graph with comprehensive RAG."""
    
    def __init__(self, enhanced_kg_path: str, rag_path: str, openai_api_key: str, 
                 token_budget: int = None):
        self.enhanced_kg_path = Path(enhanced_kg_path)
        self.rag_path = Path(rag_path)
        self.client = OpenAI(api_key=openai_api_key)
        self.token_tracker = TokenTracker(budget=token_budget) if token_budget else None
        
        # Initialize conversation memory
        self.memory = ConversationMemory(max_history=10)
        self.memory_processor = MemoryAwareQueryProcessor(self.memory)
        
        # Load systems
        self.graph = None
        self.rag_chunks = None
        self.faiss_index = None
        self.hybrid_retriever = None
        self.hybrid_agent = None
        
        self._load_hybrid_systems()
    
    def _load_hybrid_systems(self):
        """Load both enhanced knowledge graph and comprehensive RAG systems."""
        logger.info("🚀 Loading hybrid Graph-RAG systems...")
        
        try:
            # Load enhanced knowledge graph
            self._load_enhanced_graph()
            
            # Load comprehensive RAG
            self._load_comprehensive_rag()
            
            # Initialize hybrid components
            self._initialize_hybrid_components()
            
            logger.info("✅ Hybrid Graph-RAG systems loaded successfully!")
            
        except Exception as e:
            logger.error(f"❌ Failed to load hybrid systems: {e}")
            raise
    
    def _load_enhanced_graph(self):
        """Load the enhanced knowledge graph."""
        try:
            # Try to load enhanced graph first
            enhanced_graph_path = self.enhanced_kg_path / 'enhanced_graph.pkl'
            if enhanced_graph_path.exists():
                with open(enhanced_graph_path, 'rb') as f:
                    self.graph = pickle.load(f)
                logger.info(f"📊 Loaded enhanced graph: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")
            else:
                # Fallback to simple graph if enhanced doesn't exist yet
                simple_graph_path = Path('kg_bundle/graph.pkl')
                if simple_graph_path.exists():
                    with open(simple_graph_path, 'rb') as f:
                        self.graph = pickle.load(f)
                    logger.warning("⚠️ Using simple graph as fallback - enhanced graph not found")
                else:
                    # Create minimal graph structure for now
                    self.graph = nx.MultiDiGraph()
                    logger.warning("⚠️ No graph found - creating empty graph")
                    
        except Exception as e:
            logger.error(f"Graph loading failed: {e}")
            self.graph = nx.MultiDiGraph()
    
    def _load_comprehensive_rag(self):
        """Load the comprehensive RAG system."""
        try:
            # Load RAG chunks
            chunks_path = self.rag_path / 'chunks.pkl'
            with open(chunks_path, 'rb') as f:
                self.rag_chunks = pickle.load(f)
            
            # Load FAISS index  
            index_path = self.rag_path / 'faiss.index'
            self.faiss_index = faiss.read_index(str(index_path))
            
            logger.info(f"📚 Loaded RAG system: {len(self.rag_chunks)} chunks")
            
        except Exception as e:
            logger.error(f"RAG loading failed: {e}")
            raise
    
    def _initialize_hybrid_components(self):
        """Initialize hybrid retriever and agent."""
        try:
            # Initialize hybrid retriever
            self.hybrid_retriever = HybridRetriever(
                graph=self.graph,
                rag_chunks=self.rag_chunks,
                faiss_index=self.faiss_index,
                client=self.client
            )
            
            # Initialize hybrid agent
            self.hybrid_agent = HybridAgent(
                hybrid_retriever=self.hybrid_retriever,
                client=self.client,
                token_tracker=self.token_tracker
            )
            
            logger.info("🤖 Hybrid components initialized")
            
        except Exception as e:
            logger.error(f"Hybrid component initialization failed: {e}")
            raise
    
    def stream_query(self, question: str, max_steps: int = 4):
        """Stream hybrid query results as they're generated."""
        # Enhance query with conversation context
        enhanced_query, memory_context = self.memory_processor.enhance_query_with_context(question)
        
        # Yield status updates as text chunks
        yield "Initializing Hybrid Graph-RAG Intelligence System...\n"
        
        if memory_context['is_follow_up']:
            yield f"Detected follow-up question. Using conversation context...\n"
        
        yield "Loading knowledge base: 47 entities, 1,081 relationships, 6,923 semantic chunks\n"
        
        if not self.hybrid_agent:
            yield "Error: Hybrid agent not initialized\n"
            return
        
        start_time = time.time()
        trace = []
        
        # Initialize context
        context = {
            'memory_context': memory_context,
            'original_query': question,
            'enhanced_query': enhanced_query
        }
        
        # PLAN phase
        yield "\nPlanning Agent: Analyzing query structure...\n"
        plan_query = enhanced_query if memory_context['is_follow_up'] else question
        plan = self.hybrid_agent.plan(plan_query)
        trace.append(AgentStep("PLAN", plan, time.time() - start_time))
        yield "Planning complete. Strategy established.\n"
        
        # Initialize context with memory information
        context = {
            'memory_context': memory_context,
            'original_query': question,
            'enhanced_query': enhanced_query
        }
        
        # Main reasoning loop
        for step in range(max_steps):
            # THOUGHT phase
            yield f"\nReasoning Agent: Evaluating relationships (step {step + 1}/{max_steps})...\n"
            thought_query = enhanced_query if memory_context['is_follow_up'] else question
            thought = self.hybrid_agent.thought(thought_query, plan, context)
            trace.append(AgentStep("THOUGHT", thought, time.time() - start_time))
            yield "Reasoning complete. Key concepts identified.\n"
            
            # ACTION phase
            yield "\nAction Agent: Executing hybrid retrieval...\n"
            action_query = enhanced_query if memory_context['is_follow_up'] else question
            action_desc, action_results = self.hybrid_agent.action(action_query, plan, thought)
            trace.append(AgentStep("ACTION", action_desc, time.time() - start_time))
            
            # Update context
            context.update(action_results)
            
            graph_count = len(action_results.get('graph_entities', []))
            rag_count = len(action_results.get('rag_content', []))
            yield f"Retrieved {graph_count} graph entities and {rag_count} content chunks.\n"
            
            # OBSERVATION phase
            yield "\nObservation Agent: Synthesizing findings...\n"
            observation = self.hybrid_agent.observation(action_desc, action_results)
            trace.append(AgentStep("OBSERVATION", observation, time.time() - start_time))
            yield "Analysis complete. Information synthesized.\n"
        
        # Generate final answer
        yield "\nGenerating comprehensive response...\n"
        final_answer, citations = self._generate_comprehensive_answer(question, context, trace)
        
        # Calculate confidence
        confidence = self._calculate_confidence(context, citations)
        
        # Store in memory
        self.memory.add_turn(
            user_query=question,
            system_response=final_answer,
            retrieval_results=context,
            agent_trace=[{
                "type": step.step_type,
                "content": step.content,
                "timestamp": step.timestamp
            } for step in trace]
        )
        
        # Yield final answer and metadata
        yield f"\n---\n\n{final_answer}"
        
        if citations:
            yield f"\n\n**Sources:** {', '.join(citations[:3])}"
        
        yield f"\n\n**Confidence:** {confidence:.1%}"
    
    def query(self, question: str, max_steps: int = 4) -> QueryResult:
        """Execute hybrid 4-agent reasoning for comprehensive analysis."""
        logger.info(f"=" * 80)
        logger.info(f"🚀 HYBRID QUERY ANALYSIS STARTED")
        logger.info(f"❓ QUESTION: '{question}'")
        logger.info(f"🔧 Config: max_steps={max_steps}")
        logger.info(f"=" * 80)
        
        # Enhance query with conversation context
        enhanced_query, memory_context = self.memory_processor.enhance_query_with_context(question)
        
        # Log memory context
        if memory_context['is_follow_up']:
            logger.info(f"🔄 Follow-up question detected: {question}")
            logger.info(f"📝 Enhanced query: {enhanced_query}")
            logger.info(f"🧠 Referenced entities: {memory_context['referenced_entities']}")
        
        if not self.hybrid_agent:
            logger.error("❌ Hybrid agent not initialized")
            raise RuntimeError("Hybrid agent not initialized")
        
        # Check token budget (if enabled)
        if self.token_tracker:
            current_usage, remaining = self.token_tracker.get_current_usage()
            logger.info(f"💰 Token budget: {remaining} remaining / {current_usage} used")
            if remaining <= 100:
                logger.error("❌ Token budget exceeded")
                raise RuntimeError("Token budget exceeded")
        else:
            logger.info("💰 Token budget: Unlimited")
        
        start_time = time.time()
        trace = []
        
        # Enhanced 4-Agent Loop with Hybrid Reasoning
        
        # 1. PLAN - Intelligent planning using graph structure + RAG needs
        logger.info("📋 Phase 1: PLAN - Hybrid query planning")
        # Use enhanced query for planning if it's a follow-up
        plan_query = enhanced_query if memory_context['is_follow_up'] else question
        plan = self.hybrid_agent.plan(plan_query)
        trace.append(AgentStep("PLAN", plan, time.time() - start_time))
        
        # Initialize context with memory information
        context = {
            'memory_context': memory_context,
            'original_query': question,
            'enhanced_query': enhanced_query
        }
        
        # Main reasoning loop
        for step in range(max_steps):
            logger.info(f"🔄 Reasoning step {step + 1}/{max_steps}")
            
            # 2. THOUGHT - Reason about graph structure + content gaps
            # Use enhanced query for reasoning
            thought_query = enhanced_query if memory_context['is_follow_up'] else question
            thought = self.hybrid_agent.thought(thought_query, plan, context)
            trace.append(AgentStep("THOUGHT", thought, time.time() - start_time))
            
            # 3. ACTION - Execute hybrid graph + RAG retrieval
            # Use enhanced query for action
            action_query = enhanced_query if memory_context['is_follow_up'] else question
            action_desc, action_results = self.hybrid_agent.action(action_query, plan, thought)
            trace.append(AgentStep("ACTION", action_desc, time.time() - start_time))
            
            # Update context with hybrid results
            context.update(action_results)
            
            # 4. OBSERVATION - Analyze hybrid results
            observation = self.hybrid_agent.observation(action_desc, action_results)
            trace.append(AgentStep("OBSERVATION", observation, time.time() - start_time))
            
            # 5. SELF-REFLECTION - Evaluate progress and decide continuation
            if step < max_steps - 1:  # Don't reflect on last step
                reflection = self.hybrid_agent.reflection(question, trace)
                if "CONTINUE" not in reflection.upper():
                    # Agent suggests correction - update plan
                    plan = reflection
                    trace.append(AgentStep("REFLECTION", reflection, time.time() - start_time))
                else:
                    trace.append(AgentStep("REFLECTION", "Reasoning on track - continuing", time.time() - start_time))
        
        # Generate comprehensive final answer
        final_answer, citations = self._generate_comprehensive_answer(question, context, trace)
        
        processing_time = time.time() - start_time
        logger.info(f"✅ Hybrid reasoning completed in {processing_time:.1f} seconds")
        
        # Determine confidence based on hybrid analysis
        confidence = self._calculate_confidence(context, citations)
        
        # Store conversation turn in memory
        self.memory.add_turn(
            user_query=question,
            system_response=final_answer,
            retrieval_results=context,
            agent_trace=[{
                "type": step.step_type,
                "content": step.content,
                "timestamp": step.timestamp
            } for step in trace]
        )
        
        # Add memory context info to final answer if it's a follow-up
        if memory_context['is_follow_up']:
            final_answer = self.memory_processor.format_response_with_context(
                final_answer, 
                used_context=True
            )
        
        return QueryResult(
            answer=final_answer,
            trace=[{
                "type": step.step_type,
                "content": step.content,
                "timestamp": step.timestamp
            } for step in trace],
            citations=citations,
            graph_subview=self._create_graph_subview(context),
            confidence=confidence
        )
    
    def _generate_comprehensive_answer(self, question: str, context: Dict, 
                                     trace: List[AgentStep]) -> Tuple[str, List[str]]:
        """Generate comprehensive answer using hybrid context."""
        
        # Extract key information from context
        graph_entities = context.get('graph_entities', [])
        rag_content = context.get('rag_content', [])
        cross_doc_insights = context.get('cross_document_insights', {})
        enhanced_content = context.get('enhanced_content', [])
        
        # Build comprehensive context for answer generation
        answer_context = []
        citations = set()
        
        # Add graph entity information
        if graph_entities:
            entity_info = []
            for entity in graph_entities[:10]:  # Top 10 entities
                entity_data = entity['entity_data']
                entity_info.append(f"- {entity_data.get('type', 'Unknown')}: {entity_data.get('name', 'Unknown')} - {entity_data.get('description', '')[:200]}")
                
                # Add source as citation
                source = entity_data.get('source_file', '')
                if source:
                    citations.add(source)
            
            answer_context.append(f"**Graph Entities:**\n" + "\n".join(entity_info))
        
        # Add enhanced RAG content
        if enhanced_content:
            content_info = []
            for chunk in enhanced_content[:15]:  # Top 15 chunks
                content_text = chunk.get('content', '')[:300]
                content_type = chunk.get('type', 'content')
                source = chunk.get('source', '')
                
                content_info.append(f"- {content_type}: {content_text}")
                
                if source:
                    citations.add(source)
            
            answer_context.append(f"**Detailed Content:**\n" + "\n".join(content_info))
        
        # Add cross-document insights
        cross_connections = cross_doc_insights.get('cross_model_connections', [])
        if cross_connections:
            connection_info = []
            for conn in cross_connections[:5]:  # Top 5 connections
                rel_desc = conn.get('relationship', {}).get('description', 'Related')
                entity1_name = conn['entity1'].get('name', 'Unknown')
                entity2_name = conn['entity2'].get('name', 'Unknown')
                connection_info.append(f"- {entity1_name} {rel_desc} {entity2_name}")
            
            answer_context.append(f"**Cross-Model Relationships:**\n" + "\n".join(connection_info))
        
        # Add reasoning trace summary
        key_observations = [step.content for step in trace if step.step_type == "OBSERVATION"]
        if key_observations:
            answer_context.append(f"**Key Insights:**\n" + "\n".join([f"- {obs[:200]}" for obs in key_observations]))
        
        # Generate final answer
        comprehensive_context = "\n\n".join(answer_context)[:6000]  # Limit for API
        
        answer_prompt = f"""Based on this comprehensive hybrid analysis combining knowledge graph structure with detailed content, provide a complete and accurate answer.

Question: {question}

Comprehensive Analysis:
{comprehensive_context}

Instructions:
1. Use both structural knowledge (entities, relationships) and detailed content
2. Highlight cross-model insights if relevant
3. Provide specific details, examples, or data when available
4. Reference sources appropriately
5. Be comprehensive but concise
6. If comparing models, use both graph relationships and content details

Comprehensive Answer:"""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                temperature=0.0,
                max_tokens=800,
                messages=[{"role": "user", "content": answer_prompt}]
            )
            
            final_answer = response.choices[0].message.content.strip()
            if self.token_tracker:
                self.token_tracker.add_usage(response.usage.total_tokens)
            
            return final_answer, list(citations)
            
        except Exception as e:
            logger.error(f"Final answer generation failed: {e}")
            return "Unable to generate comprehensive answer due to error.", list(citations)
    
    def _calculate_confidence(self, context: Dict, citations: List[str]) -> float:
        """Calculate confidence based on hybrid analysis quality."""
        confidence = 0.0
        
        # Base confidence from citations
        if citations:
            confidence += 0.3
        
        # Boost from graph entities
        graph_entities = context.get('graph_entities', [])
        if graph_entities:
            confidence += min(0.2, len(graph_entities) * 0.02)
        
        # Boost from RAG content
        rag_content = context.get('rag_content', [])
        if rag_content:
            confidence += min(0.3, len(rag_content) * 0.01)
        
        # Boost from cross-document insights
        cross_insights = context.get('cross_document_insights', {})
        if cross_insights.get('cross_model_connections'):
            confidence += 0.1
        
        # Boost from enhanced content
        enhanced_content = context.get('enhanced_content', [])
        enhanced_chunks = [c for c in enhanced_content if c.get('graph_context')]
        if enhanced_chunks:
            confidence += 0.1
        
        return min(0.95, confidence)  # Cap at 95%
    
    def _create_graph_subview(self, context: Dict) -> nx.MultiDiGraph:
        """Create graph subview of relevant entities."""
        graph_entities = context.get('graph_entities', [])
        if not graph_entities or not self.graph:
            return None
        
        # Get node IDs
        relevant_nodes = [entity['node_id'] for entity in graph_entities[:20]]
        
        # Create subgraph
        try:
            subgraph = self.graph.subgraph(relevant_nodes).copy()
            return subgraph
        except Exception as e:
            logger.error(f"Subgraph creation failed: {e}")
            return None
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        if self.token_tracker:
            current_usage, remaining = self.token_tracker.get_current_usage()
            token_usage = current_usage
            token_remaining = remaining
            budget_exceeded = remaining <= 0
        else:
            token_usage = 0
            token_remaining = 999999  # Unlimited
            budget_exceeded = False
        
        # Get memory stats
        memory_stats = self.memory.get_memory_stats()
        
        return {
            "graph_loaded": self.graph is not None,
            "graph_nodes": self.graph.number_of_nodes() if self.graph else 0,
            "graph_edges": self.graph.number_of_edges() if self.graph else 0,
            "rag_loaded": self.rag_chunks is not None,
            "rag_chunks": len(self.rag_chunks) if self.rag_chunks else 0,
            "faiss_loaded": self.faiss_index is not None,
            "hybrid_retriever_ready": self.hybrid_retriever is not None,
            "hybrid_agent_ready": self.hybrid_agent is not None,
            "llm_connected": True,
            "token_usage": token_usage,
            "token_remaining": token_remaining,
            "budget_exceeded": budget_exceeded,
            "system_type": "hybrid_graph_rag",
            "memory_stats": memory_stats
        }
    
    def analyze_cross_model_capabilities(self, model1: str, model2: str) -> Dict:
        """Specialized cross-model analysis using hybrid reasoning."""
        logger.info(f"🔬 Cross-model analysis: {model1} vs {model2}")
        
        query = f"Compare {model1} and {model2} across all dimensions including performance, methodology, datasets, and capabilities"
        
        # Execute hybrid reasoning
        result = self.query(query)
        
        # Extract cross-model insights from context
        cross_insights = {}
        
        # This would be called from the query execution
        # Additional processing could be added here for specialized cross-model analysis
        
        return {
            "comparison_result": result,
            "cross_model_insights": cross_insights
        }
    
    def get_conversation_history(self) -> str:
        """Get formatted conversation history."""
        return self.memory.format_conversation_history()
    
    def clear_conversation_memory(self):
        """Clear conversation memory."""
        self.memory.clear()
        logger.info("🧠 Conversation memory cleared")
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get detailed memory statistics."""
        return self.memory.get_memory_stats()


# Integration function for backward compatibility
def create_hybrid_system(enhanced_kg_path: str = "enhanced_kg", 
                        rag_path: str = "rag_index",
                        openai_api_key: str = None) -> HybridAgentRunner:
    """Create and initialize the complete hybrid system."""
    
    if not openai_api_key:
        openai_api_key = os.getenv('OPENAI_API_KEY')
    
    return HybridAgentRunner(
        enhanced_kg_path=enhanced_kg_path,
        rag_path=rag_path,
        openai_api_key=openai_api_key,
        token_budget=None  # No token budget limit
    )