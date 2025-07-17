"""
Hybrid Graph-RAG Integration Layer
Combines enhanced knowledge graph with comprehensive RAG for intelligent reasoning.
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
import tiktoken
from dotenv import load_dotenv

from .graph_counselor.schema_utils import QueryResult, AgentStep, AgentTrace

load_dotenv()
logger = logging.getLogger(__name__)


class HybridRetriever:
    """Intelligent retriever that combines graph structure with RAG content."""
    
    def __init__(self, graph: nx.MultiDiGraph, rag_chunks: List[Dict], 
                 faiss_index, client: OpenAI):
        self.graph = graph
        self.rag_chunks = rag_chunks
        self.faiss_index = faiss_index
        self.client = client
        
        # Create entity-to-chunk mapping for hybrid retrieval
        self._build_entity_chunk_mapping()
    
    def _build_entity_chunk_mapping(self):
        """Build mapping between graph entities and RAG chunks."""
        self.entity_chunk_map = {}
        
        # Map entities to their source content in RAG
        for node_id, node_data in self.graph.nodes(data=True):
            source_file = node_data.get('source_file', '')
            entity_name = node_data.get('name', '')
            
            # Find related RAG chunks
            related_chunks = []
            for i, chunk in enumerate(self.rag_chunks):
                chunk_source = chunk.get('source', '')
                chunk_content = chunk.get('content', '')
                
                # Match by source file or content similarity
                if (source_file in chunk_source or 
                    entity_name.lower() in chunk_content.lower()):
                    related_chunks.append(i)
            
            self.entity_chunk_map[node_id] = related_chunks
        
        logger.info(f"Built entity-chunk mapping for {len(self.entity_chunk_map)} entities")
    
    def hybrid_retrieve(self, query: str, k_graph: int = 10, k_rag: int = 20) -> Dict:
        """Perform hybrid retrieval using both graph and RAG."""
        logger.info(f"🔍 HYBRID RETRIEVAL STARTED")
        logger.info(f"📝 Query: '{query}'")
        logger.info(f"🎯 Targets: {k_graph} graph entities, {k_rag} RAG chunks")
        
        # Phase 1: Graph-based entity discovery
        logger.info(f"📊 Phase 1: Graph-based entity discovery")
        graph_entities = self._graph_entity_discovery(query, k_graph)
        
        # Phase 2: RAG content retrieval  
        logger.info(f"📚 Phase 2: RAG content retrieval")
        rag_content = self._rag_content_retrieval(query, k_rag)
        
        # Phase 3: Graph-guided RAG enhancement
        logger.info(f"🔗 Phase 3: Graph-guided RAG enhancement")
        enhanced_content = self._enhance_rag_with_graph(graph_entities, rag_content)
        
        # Phase 4: Cross-document relationship analysis
        logger.info(f"🌐 Phase 4: Cross-document relationship analysis")
        cross_doc_insights = self._analyze_cross_document_relationships(graph_entities)
        
        # Summary logging
        logger.info(f"✅ HYBRID RETRIEVAL COMPLETED")
        logger.info(f"📊 Results: {len(graph_entities)} graph entities, {len(rag_content)} RAG chunks, {len(enhanced_content)} enhanced chunks")
        logger.info(f"🔗 Cross-doc insights: {len(cross_doc_insights.get('cross_model_connections', []))} connections")
        
        return {
            'graph_entities': graph_entities,
            'rag_content': rag_content,
            'enhanced_content': enhanced_content,
            'cross_document_insights': cross_doc_insights,
            'retrieval_strategy': 'hybrid_graph_rag'
        }
    
    def _graph_entity_discovery(self, query: str, k: int) -> List[Dict]:
        """Discover relevant entities using graph structure."""
        logger.info(f"🔍 GRAPH ENTITY DISCOVERY")
        logger.info(f"📝 Query: '{query}' -> Query terms: {query.lower().split()}")
        
        relevant_entities = []
        
        # Find entities by name/description matching
        query_lower = query.lower()
        query_terms = query_lower.split()
        
        logger.info(f"🌐 Traversing graph with {self.graph.number_of_nodes()} nodes")
        
        scored_entities = []
        for node_id, node_data in self.graph.nodes(data=True):
            entity_name = node_data.get('name', '').lower()
            entity_desc = node_data.get('description', '').lower()
            entity_type = node_data.get('type', '')
            
            # Calculate relevance score
            score = 0
            matched_terms = []
            
            for term in query_terms:
                if term in entity_name:
                    score += 3
                    matched_terms.append(f"name:'{term}'")
                if term in entity_desc:
                    score += 1
                    matched_terms.append(f"desc:'{term}'")
            
            # Boost certain entity types based on query
            boosts = []
            if 'model' in query_lower and entity_type in ['MODELS', 'MODEL', 'ALGORITHM']:
                score += 2
                boosts.append(f"model_type_boost:+2")
            if 'dataset' in query_lower and entity_type in ['DATASETS', 'DATASET']:
                score += 2
                boosts.append(f"dataset_type_boost:+2")
            if 'performance' in query_lower and entity_type in ['METRICS', 'PERFORMANCE']:
                score += 2
                boosts.append(f"performance_type_boost:+2")
            if 'compare' in query_lower and entity_type in ['MODELS', 'MODEL']:
                score += 1
                boosts.append(f"compare_boost:+1")
            
            if score > 0:
                entity_info = {
                    'node_id': node_id,
                    'entity_data': node_data,
                    'relevance_score': score,
                    'related_chunks': self.entity_chunk_map.get(node_id, [])
                }
                relevant_entities.append(entity_info)
                scored_entities.append({
                    'name': entity_name,
                    'type': entity_type,
                    'score': score,
                    'matches': matched_terms,
                    'boosts': boosts
                })
        
        # Sort by relevance and return top k
        relevant_entities.sort(key=lambda x: x['relevance_score'], reverse=True)
        
        # Log detailed scoring results
        logger.info(f"📊 Graph scoring results (top 10):")
        for i, entity in enumerate(scored_entities[:10]):
            logger.info(f"  {i+1}. {entity['name']} ({entity['type']}) - Score: {entity['score']}")
            if entity['matches']:
                logger.info(f"     Matches: {', '.join(entity['matches'])}")
            if entity['boosts']:
                logger.info(f"     Boosts: {', '.join(entity['boosts'])}")
        
        final_entities = relevant_entities[:k]
        logger.info(f"📍 Selected {len(final_entities)} top graph entities")
        
        for i, entity in enumerate(final_entities):
            logger.info(f"  {i+1}. {entity['entity_data'].get('name', 'Unknown')} (Score: {entity['relevance_score']}, Chunks: {len(entity['related_chunks'])})")
        
        return final_entities
    
    def _rag_content_retrieval(self, query: str, k: int) -> List[Dict]:
        """Retrieve content using enhanced multi-modal RAG search."""
        logger.info(f"🔍 RAG CONTENT RETRIEVAL")
        logger.info(f"📝 Query: '{query}' -> Target: {k} chunks")
        
        try:
            # Phase 1: Semantic similarity search (primary)
            logger.info(f"🧠 Phase 1: Semantic similarity search")
            semantic_results = self._semantic_similarity_search(query, k)
            
            # Phase 2: Keyword-based search (complementary)
            logger.info(f"🔤 Phase 2: Keyword-based search")
            keyword_results = self._keyword_based_search(query, k // 2)
            
            # Phase 3: Hybrid ranking combining both approaches
            logger.info(f"🔀 Phase 3: Hybrid ranking")
            combined_results = self._hybrid_rank_results(semantic_results, keyword_results, query)
            
            # Return top k results
            final_results = combined_results[:k]
            
            logger.info(f"📖 RAG RETRIEVAL COMPLETED")
            logger.info(f"📊 Results: {len(final_results)} final chunks (semantic: {len(semantic_results)}, keyword: {len(keyword_results)}, combined: {len(combined_results)})")
            
            # Log final chunk details
            logger.info(f"📋 Final selected chunks:")
            for i, chunk in enumerate(final_results):
                source = chunk.get('source', 'unknown')
                method = chunk.get('retrieval_method', 'unknown')
                score = chunk.get('final_score', chunk.get('relevance_score', 0))
                logger.info(f"  {i+1}. {source} (Method: {method}, Score: {score:.3f})")
            
            return final_results
            
        except Exception as e:
            logger.error(f"❌ RAG retrieval failed: {e}")
            return []
    
    def _semantic_similarity_search(self, query: str, k: int) -> List[Dict]:
        """Traditional semantic similarity search."""
        logger.info(f"🧠 SEMANTIC SIMILARITY SEARCH")
        logger.info(f"📝 Query: '{query}' -> Target: {k} chunks (searching {k*2} for filtering)")
        
        try:
            # Create embedding for query
            logger.info(f"🔤 Creating embedding using text-embedding-ada-002")
            response = self.client.embeddings.create(
                model="text-embedding-ada-002",
                input=[query]
            )
            
            query_embedding = np.array([response.data[0].embedding], dtype=np.float32)
            faiss.normalize_L2(query_embedding)
            logger.info(f"🎯 Query embedding created (dimension: {len(query_embedding[0])})")
            
            # Search FAISS index
            logger.info(f"🔍 Searching FAISS index with {len(self.rag_chunks)} chunks")
            scores, indices = self.faiss_index.search(query_embedding, k * 2)  # Get more for filtering
            
            # Get relevant chunks with scores
            results = []
            logger.info(f"📊 Top semantic similarity results:")
            for i, idx in enumerate(indices[0]):
                if idx < len(self.rag_chunks):
                    chunk = self.rag_chunks[idx].copy()
                    chunk['relevance_score'] = float(scores[0][i])
                    chunk['retrieval_method'] = 'semantic'
                    results.append(chunk)
                    
                    # Log top 10 results
                    if i < 10:
                        source = chunk.get('source', 'unknown')
                        score = float(scores[0][i])
                        logger.info(f"  {i+1}. {source} (Score: {score:.3f})")
            
            logger.info(f"✅ Semantic search completed: {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"❌ Semantic search failed: {e}")
            return []
    
    def _keyword_based_search(self, query: str, k: int) -> List[Dict]:
        """Keyword-based search to catch exact matches."""
        logger.info(f"🔤 KEYWORD-BASED SEARCH")
        logger.info(f"📝 Query: '{query}' -> Target: {k} chunks")
        
        results = []
        query_words = query.lower().split()
        
        # Expand query with synonyms and related terms
        expanded_terms = self._expand_query_terms(query_words)
        logger.info(f"🔄 Query expansion: {query_words} -> {expanded_terms}")
        
        logger.info(f"🔍 Scanning {len(self.rag_chunks)} chunks for keyword matches")
        
        scored_chunks = []
        for idx, chunk in enumerate(self.rag_chunks):
            content = chunk.get('content', '').lower()
            score = 0
            matched_terms = []
            
            # Score based on keyword matches
            for term in expanded_terms:
                if term in content:
                    # Boost score for exact phrase matches
                    if term in query.lower():
                        score += 2
                        matched_terms.append(f"exact:'{term}'")
                    else:
                        score += 1
                        matched_terms.append(f"expanded:'{term}'")
            
            # Boost code-related chunks for implementation queries
            code_boost = 0
            if any(code_indicator in content for code_indicator in ['def ', 'class ', 'import ', '# ', '"""']):
                if any(impl_word in query.lower() for impl_word in ['handle', 'implement', 'callback', 'function']):
                    code_boost = 1
                    score += 1
                    matched_terms.append("code_boost:+1")
            
            if score > 0:
                chunk_copy = chunk.copy()
                normalized_score = score / len(expanded_terms)  # Normalize
                chunk_copy['relevance_score'] = normalized_score
                chunk_copy['retrieval_method'] = 'keyword'
                results.append(chunk_copy)
                
                scored_chunks.append({
                    'source': chunk.get('source', 'unknown'),
                    'raw_score': score,
                    'normalized_score': normalized_score,
                    'matches': matched_terms,
                    'has_code_boost': code_boost > 0
                })
        
        # Sort by score and return top k
        results.sort(key=lambda x: x['relevance_score'], reverse=True)
        
        # Log top keyword results
        logger.info(f"📊 Top keyword search results:")
        for i, chunk_info in enumerate(scored_chunks[:10]):
            logger.info(f"  {i+1}. {chunk_info['source']} (Score: {chunk_info['normalized_score']:.3f})")
            if chunk_info['matches']:
                logger.info(f"     Matches: {', '.join(chunk_info['matches'])}")
        
        final_results = results[:k]
        logger.info(f"✅ Keyword search completed: {len(final_results)} results from {len(results)} matches")
        
        return final_results
    
    def _expand_query_terms(self, query_words: List[str]) -> List[str]:
        """Expand query terms with synonyms and related terms."""
        expanded = query_words.copy()
        
        # Synonym mapping for common terms
        synonyms = {
            'context': ['context', 'ctx', 'contextual'],
            'data': ['data', 'information', 'content'],
            'construction': ['construction', 'construct', 'build', 'create', 'handle'],
            'handle': ['handle', 'process', 'manage', 'deal'],
            'callback': ['callback', 'handler', 'hook', 'event'],
            'query': ['query', 'search', 'request', 'question']
        }
        
        for word in query_words:
            if word in synonyms:
                expanded.extend(synonyms[word])
        
        return list(set(expanded))  # Remove duplicates
    
    def _hybrid_rank_results(self, semantic_results: List[Dict], keyword_results: List[Dict], query: str) -> List[Dict]:
        """Combine and re-rank results from both retrieval methods."""
        logger.info(f"🔀 HYBRID RANKING")
        logger.info(f"📝 Query: '{query}'")
        logger.info(f"📊 Input: {len(semantic_results)} semantic + {len(keyword_results)} keyword results")
        
        # Create a unified scoring system
        combined_results = {}
        
        # Add semantic results
        logger.info(f"🧠 Processing semantic results...")
        for result in semantic_results:
            content_hash = hash(result.get('content', ''))
            if content_hash not in combined_results:
                combined_results[content_hash] = result.copy()
                combined_results[content_hash]['final_score'] = result['relevance_score']
                combined_results[content_hash]['methods'] = ['semantic']
            else:
                combined_results[content_hash]['methods'].append('semantic')
        
        # Add keyword results with boosting
        logger.info(f"🔤 Processing keyword results...")
        for result in keyword_results:
            content_hash = hash(result.get('content', ''))
            if content_hash not in combined_results:
                combined_results[content_hash] = result.copy()
                combined_results[content_hash]['final_score'] = result['relevance_score'] * 0.7  # Slight penalty for keyword-only
                combined_results[content_hash]['methods'] = ['keyword']
            else:
                # Boost items found by both methods
                combined_results[content_hash]['final_score'] += result['relevance_score'] * 0.3
                combined_results[content_hash]['methods'].append('keyword')
        
        # Apply hybrid boosting
        logger.info(f"⚡ Applying hybrid boosting...")
        hybrid_boosted = 0
        for content_hash, result in combined_results.items():
            if len(result['methods']) > 1:
                # Boost results found by multiple methods
                old_score = result['final_score']
                result['final_score'] *= 1.2
                result['retrieval_method'] = 'hybrid'
                hybrid_boosted += 1
                logger.info(f"  Boosted: {result.get('source', 'unknown')} ({old_score:.3f} -> {result['final_score']:.3f})")
        
        # Sort by final score
        final_results = list(combined_results.values())
        final_results.sort(key=lambda x: x['final_score'], reverse=True)
        
        logger.info(f"📊 Hybrid ranking results:")
        logger.info(f"  Combined: {len(combined_results)} unique chunks")
        logger.info(f"  Hybrid boosted: {hybrid_boosted} chunks")
        
        # Log top 10 hybrid results
        logger.info(f"🏆 Top 10 hybrid-ranked results:")
        for i, result in enumerate(final_results[:10]):
            source = result.get('source', 'unknown')
            methods = '+'.join(result.get('methods', []))
            final_score = result.get('final_score', 0)
            logger.info(f"  {i+1}. {source} (Methods: {methods}, Score: {final_score:.3f})")
        
        logger.info(f"✅ Hybrid ranking completed: {len(final_results)} results")
        return final_results
    
    def _enhance_rag_with_graph(self, graph_entities: List[Dict], 
                               rag_content: List[Dict]) -> List[Dict]:
        """Enhance RAG content with graph entity context."""
        enhanced_content = []
        
        for chunk in rag_content:
            chunk_enhanced = chunk.copy()
            
            # Find related graph entities for this chunk
            related_entities = []
            chunk_source = chunk.get('source', '')
            chunk_content = chunk.get('content', '').lower()
            
            for entity_info in graph_entities:
                entity_data = entity_info['entity_data']
                entity_name = entity_data.get('name', '').lower()
                entity_source = entity_data.get('source_file', '')
                
                # Check if entity is related to this chunk
                if (entity_source in chunk_source or 
                    entity_name in chunk_content or 
                    chunk_source in entity_source):
                    related_entities.append(entity_info)
            
            chunk_enhanced['related_entities'] = related_entities
            chunk_enhanced['graph_context'] = len(related_entities) > 0
            enhanced_content.append(chunk_enhanced)
        
        logger.info(f"🔗 Enhanced {len(enhanced_content)} chunks with graph context")
        return enhanced_content
    
    def _analyze_cross_document_relationships(self, graph_entities: List[Dict]) -> Dict:
        """Analyze relationships across different documents/models."""
        logger.info(f"🌐 CROSS-DOCUMENT RELATIONSHIP ANALYSIS")
        logger.info(f"📊 Input: {len(graph_entities)} graph entities")
        
        cross_doc_insights = {
            'cross_model_connections': [],
            'shared_concepts': [],
            'comparison_opportunities': []
        }
        
        # Group entities by source
        entities_by_source = {}
        for entity_info in graph_entities:
            source = entity_info['entity_data'].get('source_file', 'unknown')
            if source not in entities_by_source:
                entities_by_source[source] = []
            entities_by_source[source].append(entity_info)
        
        # Find cross-document relationships in graph
        sources = list(entities_by_source.keys())
        for i, source1 in enumerate(sources):
            for source2 in sources[i+1:]:
                # Look for relationships between entities from different sources
                entities1 = entities_by_source[source1]
                entities2 = entities_by_source[source2]
                
                for entity1 in entities1:
                    for entity2 in entities2:
                        node1 = entity1['node_id']
                        node2 = entity2['node_id']
                        
                        # Check if there's a direct relationship in graph
                        if self.graph.has_edge(node1, node2) or self.graph.has_edge(node2, node1):
                            relationship_data = (self.graph.get_edge_data(node1, node2) or 
                                               self.graph.get_edge_data(node2, node1))
                            
                            cross_doc_insights['cross_model_connections'].append({
                                'entity1': entity1['entity_data'],
                                'entity2': entity2['entity_data'],
                                'relationship': relationship_data,
                                'source1': source1,
                                'source2': source2
                            })
        
        # Find shared concepts (entities with similar names across sources)
        for source1 in sources:
            for source2 in sources:
                if source1 != source2:
                    entities1 = entities_by_source[source1]
                    entities2 = entities_by_source[source2]
                    
                    for entity1 in entities1:
                        for entity2 in entities2:
                            name1 = entity1['entity_data'].get('name', '').lower()
                            name2 = entity2['entity_data'].get('name', '').lower()
                            
                            # Check for similar names (shared concepts)
                            if (name1 and name2 and 
                                (name1 in name2 or name2 in name1 or 
                                 self._calculate_similarity(name1, name2) > 0.7)):
                                cross_doc_insights['shared_concepts'].append({
                                    'concept': name1,
                                    'entity1': entity1['entity_data'],
                                    'entity2': entity2['entity_data'],
                                    'source1': source1,
                                    'source2': source2
                                })
        
        # Identify comparison opportunities (models from different sources)
        model_entities = {}
        for source, entities in entities_by_source.items():
            for entity in entities:
                if entity['entity_data'].get('type') in ['MODELS', 'MODEL', 'ALGORITHM']:
                    if source not in model_entities:
                        model_entities[source] = []
                    model_entities[source].append(entity)
        
        model_sources = list(model_entities.keys())
        for i, source1 in enumerate(model_sources):
            for source2 in model_sources[i+1:]:
                for model1 in model_entities[source1]:
                    for model2 in model_entities[source2]:
                        cross_doc_insights['comparison_opportunities'].append({
                            'model1': model1['entity_data'],
                            'model2': model2['entity_data'],
                            'source1': source1,
                            'source2': source2,
                            'comparison_type': 'cross_model'
                        })
        
        logger.info(f"✅ Cross-document analysis completed:")
        logger.info(f"  🔗 Cross-model connections: {len(cross_doc_insights['cross_model_connections'])}")
        logger.info(f"  📚 Shared concepts: {len(cross_doc_insights['shared_concepts'])}")
        logger.info(f"  🎯 Comparison opportunities: {len(cross_doc_insights['comparison_opportunities'])}")
        
        return cross_doc_insights
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate simple text similarity."""
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)


class HybridAgent:
    """Enhanced agent that combines graph reasoning with RAG content analysis."""
    
    def __init__(self, hybrid_retriever: HybridRetriever, client: OpenAI, token_tracker):
        self.retriever = hybrid_retriever
        self.client = client
        self.token_tracker = token_tracker
    
    def plan(self, question: str, context: Dict = None) -> str:
        """Create intelligent plan using graph structure + RAG content needs."""
        
        # Analyze query type and requirements
        query_analysis = self._analyze_query_type(question)
        
        plan_prompt = f"""Create a comprehensive plan to answer this question using both knowledge graph structure and detailed content.

Question: {question}

Query Analysis:
- Type: {query_analysis['type']}
- Requires cross-model analysis: {query_analysis['cross_model']}
- Needs detailed content: {query_analysis['detailed_content']}
- Comparison required: {query_analysis['comparison']}

Available Resources:
1. Knowledge Graph: Entities and relationships across documents
2. RAG Content: Detailed text, code, and data from all sources
3. Cross-document analysis: Relationships between different models/sources

Create a step-by-step plan that:
1. Identifies what entities and relationships to explore
2. Determines what detailed content is needed
3. Plans cross-document analysis if required
4. Outlines synthesis strategy

Plan (3-4 specific steps):"""

        return self._call_llm(plan_prompt, "PLAN")
    
    def thought(self, question: str, plan: str, context: Dict) -> str:
        """Enhanced reasoning about graph structure + content gaps."""
        
        thought_prompt = f"""Given this question, plan, and retrieved context, reason about what we have and what we need.

Question: {question}
Plan: {plan}

Retrieved Context:
- Graph entities found: {len(context.get('graph_entities', []))}
- RAG content chunks: {len(context.get('rag_content', []))}
- Cross-document insights: {len(context.get('cross_document_insights', {}).get('cross_model_connections', []))}

Analyze:
1. Do we have the right entities and relationships?
2. Is the detailed content sufficient and relevant?
3. Are there gaps in cross-document understanding?
4. What additional retrieval or analysis is needed?

Reasoning (2-3 sentences):"""

        return self._call_llm(thought_prompt, "THOUGHT")
    
    def action(self, question: str, plan: str, thought: str) -> Tuple[str, Dict]:
        """Execute hybrid graph + RAG actions."""
        
        # Determine action type based on thought
        if "cross-document" in thought.lower() or "compare" in question.lower():
            action_type = "cross_model_analysis"
        elif "relationship" in thought.lower():
            action_type = "relationship_exploration"
        else:
            action_type = "comprehensive_retrieval"
        
        # Execute hybrid retrieval
        hybrid_results = self.retriever.hybrid_retrieve(question, k_graph=15, k_rag=25)
        
        action_description = f"Executed {action_type} using hybrid graph-RAG retrieval"
        
        return action_description, hybrid_results
    
    def observation(self, action: str, results: Dict) -> str:
        """Analyze hybrid results comprehensively."""
        
        graph_entities = results.get('graph_entities', [])
        rag_content = results.get('rag_content', [])
        cross_doc_insights = results.get('cross_document_insights', {})
        
        # Analyze what we found
        entity_types = {}
        for entity in graph_entities:
            entity_type = entity['entity_data'].get('type', 'Unknown')
            entity_types[entity_type] = entity_types.get(entity_type, 0) + 1
        
        content_sources = set()
        for chunk in rag_content:
            content_sources.add(Path(chunk.get('source', '')).name)
        
        cross_connections = len(cross_doc_insights.get('cross_model_connections', []))
        shared_concepts = len(cross_doc_insights.get('shared_concepts', []))
        
        observation_prompt = f"""Analyze these hybrid retrieval results:

Action: {action}

Graph Analysis:
- Found {len(graph_entities)} relevant entities
- Entity types: {entity_types}
- Relationships identified: Yes/No based on connections

Content Analysis:
- Retrieved {len(rag_content)} content chunks
- Source files: {len(content_sources)} different files
- Content types: Mixed (PDF, code, docs)

Cross-Document Analysis:
- Cross-model connections: {cross_connections}
- Shared concepts: {shared_concepts}

Summarize what this tells us about the question and what insights we've gained.

Observation (2-3 sentences):"""

        return self._call_llm(observation_prompt, "OBSERVATION")
    
    def reflection(self, question: str, trace: List[AgentStep]) -> str:
        """Reflect on hybrid reasoning process."""
        
        trace_summary = "\n".join([f"{step.step_type}: {step.content[:100]}..." for step in trace[-3:]])
        
        reflection_prompt = f"""Review this hybrid graph-RAG reasoning process:

Question: {question}

Recent reasoning:
{trace_summary}

Evaluate:
1. Did we effectively use both graph structure and detailed content?
2. Are we making good use of cross-document relationships?
3. Is the reasoning leading to a comprehensive answer?

If the approach is working well, respond "CONTINUE".
If corrections are needed, suggest a better approach.

Reflection:"""

        return self._call_llm(reflection_prompt, "REFLECTION")
    
    def _analyze_query_type(self, question: str) -> Dict:
        """Analyze what type of query this is."""
        question_lower = question.lower()
        
        return {
            'type': self._determine_query_type(question_lower),
            'cross_model': any(word in question_lower for word in ['compare', 'vs', 'versus', 'difference', 'both']),
            'detailed_content': any(word in question_lower for word in ['how', 'what', 'explain', 'describe', 'details']),
            'comparison': any(word in question_lower for word in ['better', 'best', 'worse', 'compare', 'superior']),
            'specific_entity': any(word in question_lower for word in ['model', 'dataset', 'algorithm', 'function', 'class'])
        }
    
    def _determine_query_type(self, question_lower: str) -> str:
        """Determine the type of query."""
        if any(word in question_lower for word in ['compare', 'vs', 'versus', 'difference']):
            return 'comparison'
        elif any(word in question_lower for word in ['how', 'explain', 'describe']):
            return 'explanation'
        elif any(word in question_lower for word in ['what', 'which', 'who']):
            return 'factual'
        elif any(word in question_lower for word in ['performance', 'result', 'score']):
            return 'performance'
        else:
            return 'general'
    
    def _call_llm(self, prompt: str, step_type: str) -> str:
        """Make LLM call with token tracking."""
        if self.token_tracker:
            estimated_tokens = self.token_tracker.estimate_tokens(prompt) + 200
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                temperature=0.0,
                max_tokens=200,
                messages=[{"role": "user", "content": prompt}]
            )
            
            result = response.choices[0].message.content.strip()
            
            # Track actual usage
            actual_tokens = response.usage.total_tokens
            if self.token_tracker:
                self.token_tracker.add_usage(actual_tokens)
            
            return result
            
        except Exception as e:
            logger.error(f"LLM call failed for {step_type}: {e}")
            return f"[Error in {step_type}]"


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