"""
Robust Cross-Model Analysis System for Hundreds of Models
Provides comprehensive comparison capabilities across multiple knowledge bases.
"""
import os
import json
import pickle
import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional, Set
import time
from dataclasses import dataclass
import networkx as nx
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


@dataclass
class ModelMetadata:
    """Metadata for a single model."""
    model_id: str
    name: str
    file_count: int
    entity_count: int
    relationship_count: int
    last_updated: float
    size_mb: float
    description: str = ""


@dataclass
class CrossModelRelationship:
    """Represents a relationship between entities from different models."""
    source_model: str
    target_model: str
    source_entity: str
    target_entity: str
    relationship_type: str
    confidence: float
    description: str


@dataclass
class ComparisonResult:
    """Result of comparing two or more models."""
    models_compared: List[str]
    comparison_summary: str
    key_differences: List[Dict[str, Any]]
    similarities: List[Dict[str, Any]]
    recommendations: List[str]
    confidence: float


class CrossModelAnalyzer:
    """Advanced cross-model analysis system for hundreds of models."""
    
    def __init__(self, knowledge_base_path: str = "knowledge_base", 
                 enhanced_kg_path: str = "enhanced_kg",
                 rag_index_path: str = "rag_index"):
        self.kb_path = Path(knowledge_base_path)
        self.kg_path = Path(enhanced_kg_path)
        self.rag_path = Path(rag_index_path)
        self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        
        # Model registry and caches
        self.models: Dict[str, ModelMetadata] = {}
        self.model_graphs: Dict[str, nx.MultiDiGraph] = {}
        self.cross_model_relationships: List[CrossModelRelationship] = []
        self.similarity_cache: Dict[Tuple[str, str], float] = {}
        
        # Load system
        self._discover_models()
        self._load_cross_model_index()
    
    def _discover_models(self):
        """Discover all available models in the knowledge base."""
        logger.info("🔍 Discovering models in knowledge base...")
        
        if not self.kb_path.exists():
            logger.warning(f"Knowledge base path {self.kb_path} does not exist")
            return
        
        for model_dir in self.kb_path.iterdir():
            if model_dir.is_dir() and model_dir.name.startswith('model_'):
                model_id = model_dir.name
                metadata = self._analyze_model_metadata(model_id, model_dir)
                self.models[model_id] = metadata
                logger.info(f"📊 Discovered {model_id}: {metadata.file_count} files, {metadata.entity_count} entities")
        
        logger.info(f"✅ Total models discovered: {len(self.models)}")
    
    def _analyze_model_metadata(self, model_id: str, model_dir: Path) -> ModelMetadata:
        """Analyze metadata for a single model."""
        # Count files
        file_count = len(list(model_dir.rglob("*.*")))
        
        # Calculate size
        total_size = sum(f.stat().st_size for f in model_dir.rglob("*.*") if f.is_file())
        size_mb = total_size / (1024 * 1024)
        
        # Get graph metadata if available
        graph_meta_path = self.kg_path / model_id / "graph" / "metadata.json"
        entity_count = 0
        relationship_count = 0
        last_updated = time.time()
        
        if graph_meta_path.exists():
            try:
                with open(graph_meta_path, 'r') as f:
                    graph_meta = json.load(f)
                entity_count = graph_meta.get('nodes', 0)
                relationship_count = graph_meta.get('edges', 0)
                last_updated = graph_meta.get('created_at', time.time())
            except Exception as e:
                logger.warning(f"Failed to load graph metadata for {model_id}: {e}")
        
        return ModelMetadata(
            model_id=model_id,
            name=model_id.replace('_', ' ').title(),
            file_count=file_count,
            entity_count=entity_count,
            relationship_count=relationship_count,
            last_updated=last_updated,
            size_mb=size_mb
        )
    
    def _load_model_graph(self, model_id: str) -> Optional[nx.MultiDiGraph]:
        """Load knowledge graph for a specific model (with caching)."""
        if model_id in self.model_graphs:
            return self.model_graphs[model_id]
        
        graph_path = self.kg_path / model_id / "graph" / "enhanced_graph.pkl"
        if not graph_path.exists():
            logger.warning(f"No graph found for {model_id}")
            return None
        
        try:
            with open(graph_path, 'rb') as f:
                graph = pickle.load(f)
            self.model_graphs[model_id] = graph
            logger.info(f"📊 Loaded graph for {model_id}: {graph.number_of_nodes()} nodes")
            return graph
        except Exception as e:
            logger.error(f"Failed to load graph for {model_id}: {e}")
            return None
    
    def _load_cross_model_index(self):
        """Load or build cross-model relationship index."""
        index_path = self.kg_path / "cross_model_relationships.json"
        
        if index_path.exists():
            try:
                with open(index_path, 'r') as f:
                    data = json.load(f)
                self.cross_model_relationships = [
                    CrossModelRelationship(**rel) for rel in data
                ]
                logger.info(f"📋 Loaded {len(self.cross_model_relationships)} cross-model relationships")
            except Exception as e:
                logger.error(f"Failed to load cross-model index: {e}")
        else:
            logger.info("🔨 Cross-model index not found, will build on first comparison")
    
    def build_cross_model_relationships(self, model_ids: Optional[List[str]] = None):
        """Build cross-model relationships using AI analysis."""
        if model_ids is None:
            model_ids = list(self.models.keys())
        
        logger.info(f"🔨 Building cross-model relationships for {len(model_ids)} models...")
        
        relationships = []
        for i, model_a in enumerate(model_ids):
            for model_b in model_ids[i+1:]:
                model_relationships = self._analyze_model_pair(model_a, model_b)
                relationships.extend(model_relationships)
        
        self.cross_model_relationships = relationships
        self._save_cross_model_index()
        logger.info(f"✅ Built {len(relationships)} cross-model relationships")
    
    def _analyze_model_pair(self, model_a: str, model_b: str) -> List[CrossModelRelationship]:
        """Analyze relationships between two specific models using AI."""
        logger.info(f"🔍 Analyzing relationship: {model_a} ↔ {model_b}")
        
        graph_a = self._load_model_graph(model_a)
        graph_b = self._load_model_graph(model_b)
        
        if not graph_a or not graph_b:
            logger.warning(f"Missing graphs for {model_a} or {model_b}")
            return []
        
        # Extract top entities from each model
        entities_a = self._get_top_entities(graph_a, model_a, limit=50)
        entities_b = self._get_top_entities(graph_b, model_b, limit=50)
        
        # Use AI to find relationships
        relationships = self._ai_detect_cross_model_relationships(
            model_a, entities_a, model_b, entities_b
        )
        
        return relationships
    
    def _get_top_entities(self, graph: nx.MultiDiGraph, model_id: str, limit: int = 50) -> List[Dict]:
        """Get top entities from a model's knowledge graph."""
        entities = []
        
        # Sort nodes by degree (most connected first)
        node_degrees = dict(graph.degree())
        sorted_nodes = sorted(node_degrees.items(), key=lambda x: x[1], reverse=True)
        
        for node_id, degree in sorted_nodes[:limit]:
            node_data = graph.nodes[node_id]
            entities.append({
                'id': node_id,
                'name': node_data.get('name', str(node_id)),
                'type': node_data.get('type', 'Unknown'),
                'description': node_data.get('description', ''),
                'degree': degree
            })
        
        return entities
    
    def _ai_detect_cross_model_relationships(self, model_a: str, entities_a: List[Dict],
                                           model_b: str, entities_b: List[Dict]) -> List[CrossModelRelationship]:
        """Use AI to detect relationships between entities from different models."""
        
        # Prepare entity summaries
        summary_a = "\n".join([
            f"- {e['name']} ({e['type']}): {e['description'][:100]}"
            for e in entities_a[:20]
        ])
        
        summary_b = "\n".join([
            f"- {e['name']} ({e['type']}): {e['description'][:100]}"
            for e in entities_b[:20]
        ])
        
        prompt = f"""Analyze these two models and identify relationships between their entities:

MODEL {model_a.upper()} - Top Entities:
{summary_a}

MODEL {model_b.upper()} - Top Entities:
{summary_b}

Find semantic relationships between entities from different models. Return JSON:
{{
  "relationships": [
    {{
      "source_entity": "entity_name_from_model_a",
      "target_entity": "entity_name_from_model_b", 
      "relationship_type": "similar|equivalent|related|implements|extends|uses",
      "confidence": 0.0-1.0,
      "description": "brief explanation"
    }}
  ]
}}

Focus on meaningful relationships with confidence > 0.6."""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                temperature=0.1,
                max_tokens=2000,
                messages=[
                    {"role": "system", "content": "You are an expert at analyzing relationships between software systems and models."},
                    {"role": "user", "content": prompt}
                ]
            )
            
            result = response.choices[0].message.content.strip()
            
            # Parse JSON response
            json_start = result.find('{')
            json_end = result.rfind('}') + 1
            if json_start != -1 and json_end > json_start:
                json_str = result[json_start:json_end]
                parsed = json.loads(json_str)
                
                relationships = []
                for rel in parsed.get('relationships', []):
                    if rel['confidence'] > 0.6:  # Only high-confidence relationships
                        relationships.append(CrossModelRelationship(
                            source_model=model_a,
                            target_model=model_b,
                            source_entity=rel['source_entity'],
                            target_entity=rel['target_entity'],
                            relationship_type=rel['relationship_type'],
                            confidence=rel['confidence'],
                            description=rel['description']
                        ))
                
                logger.info(f"🔗 Found {len(relationships)} relationships between {model_a} and {model_b}")
                return relationships
                
        except Exception as e:
            logger.error(f"AI relationship detection failed: {e}")
        
        return []
    
    def _save_cross_model_index(self):
        """Save cross-model relationships to disk."""
        index_path = self.kg_path / "cross_model_relationships.json"
        
        # Ensure directory exists
        index_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = [
            {
                'source_model': rel.source_model,
                'target_model': rel.target_model,
                'source_entity': rel.source_entity,
                'target_entity': rel.target_entity,
                'relationship_type': rel.relationship_type,
                'confidence': rel.confidence,
                'description': rel.description
            }
            for rel in self.cross_model_relationships
        ]
        
        with open(index_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"💾 Saved cross-model index with {len(data)} relationships")
    
    def compare_models(self, model_ids: List[str], comparison_query: str) -> ComparisonResult:
        """Compare multiple models based on a specific query."""
        logger.info(f"🔍 Comparing models {model_ids} for: {comparison_query}")
        
        # Load relevant data for each model
        model_data = {}
        for model_id in model_ids:
            model_data[model_id] = {
                'metadata': self.models.get(model_id),
                'graph': self._load_model_graph(model_id),
                'relevant_entities': self._find_relevant_entities(model_id, comparison_query)
            }
        
        # Use AI to perform comparison
        comparison = self._ai_compare_models(model_data, comparison_query)
        
        return comparison
    
    def _find_relevant_entities(self, model_id: str, query: str) -> List[Dict]:
        """Find entities relevant to the comparison query."""
        graph = self._load_model_graph(model_id)
        if not graph:
            return []
        
        # Simple relevance scoring based on name/description matching
        relevant_entities = []
        query_words = set(query.lower().split())
        
        for node_id in graph.nodes():
            node_data = graph.nodes[node_id]
            entity_text = f"{node_data.get('name', '')} {node_data.get('description', '')}".lower()
            
            # Calculate relevance score
            matches = sum(1 for word in query_words if word in entity_text)
            if matches > 0:
                relevant_entities.append({
                    'id': node_id,
                    'name': node_data.get('name', str(node_id)),
                    'type': node_data.get('type', 'Unknown'),
                    'description': node_data.get('description', ''),
                    'relevance_score': matches / len(query_words)
                })
        
        # Sort by relevance and return top entities
        relevant_entities.sort(key=lambda x: x['relevance_score'], reverse=True)
        return relevant_entities[:20]
    
    def _ai_compare_models(self, model_data: Dict[str, Dict], query: str) -> ComparisonResult:
        """Use AI to perform detailed model comparison."""
        
        # Prepare comparison context
        model_summaries = []
        for model_id, data in model_data.items():
            metadata = data['metadata']
            entities = data['relevant_entities']
            
            entity_summary = "\n".join([
                f"  - {e['name']} ({e['type']}): {e['description'][:100]}"
                for e in entities[:10]
            ])
            
            model_summaries.append(f"""
{model_id.upper()}:
- Files: {metadata.file_count if metadata else 'Unknown'}
- Entities: {metadata.entity_count if metadata else 'Unknown'}
- Size: {metadata.size_mb:.1f}MB
- Relevant entities for '{query}':
{entity_summary}
""")
        
        prompt = f"""Compare these models based on the query: "{query}"

{chr(10).join(model_summaries)}

Cross-model relationships:
{self._get_relevant_cross_relationships(list(model_data.keys()))}

Provide a comprehensive comparison in JSON format:
{{
  "comparison_summary": "High-level comparison overview",
  "key_differences": [
    {{"aspect": "specific area", "model_a": "description", "model_b": "description", "significance": "high|medium|low"}}
  ],
  "similarities": [
    {{"aspect": "shared feature", "description": "how they're similar", "models": ["model1", "model2"]}}
  ],
  "recommendations": ["actionable insights"],
  "confidence": 0.0-1.0
}}"""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                temperature=0.2,
                max_tokens=3000,
                messages=[
                    {"role": "system", "content": "You are an expert at comparing and analyzing software systems, models, and architectures."},
                    {"role": "user", "content": prompt}
                ]
            )
            
            result = response.choices[0].message.content.strip()
            
            # Parse JSON response
            json_start = result.find('{')
            json_end = result.rfind('}') + 1
            if json_start != -1 and json_end > json_start:
                json_str = result[json_start:json_end]
                parsed = json.loads(json_str)
                
                return ComparisonResult(
                    models_compared=list(model_data.keys()),
                    comparison_summary=parsed.get('comparison_summary', ''),
                    key_differences=parsed.get('key_differences', []),
                    similarities=parsed.get('similarities', []),
                    recommendations=parsed.get('recommendations', []),
                    confidence=parsed.get('confidence', 0.7)
                )
                
        except Exception as e:
            logger.error(f"AI comparison failed: {e}")
        
        # Fallback comparison
        return ComparisonResult(
            models_compared=list(model_data.keys()),
            comparison_summary=f"Basic comparison of {len(model_data)} models",
            key_differences=[],
            similarities=[],
            recommendations=["Enable detailed AI comparison by checking system logs"],
            confidence=0.3
        )
    
    def _get_relevant_cross_relationships(self, model_ids: List[str]) -> str:
        """Get cross-model relationships relevant to the models being compared."""
        relevant_rels = [
            rel for rel in self.cross_model_relationships
            if rel.source_model in model_ids and rel.target_model in model_ids
        ]
        
        if not relevant_rels:
            return "No cross-model relationships found."
        
        rel_text = []
        for rel in relevant_rels[:10]:  # Limit to avoid token overflow
            rel_text.append(
                f"- {rel.source_entity} ({rel.source_model}) {rel.relationship_type} "
                f"{rel.target_entity} ({rel.target_model}): {rel.description}"
            )
        
        return "\n".join(rel_text)
    
    def get_model_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about all models."""
        if not self.models:
            return {"error": "No models found"}
        
        total_files = sum(m.file_count for m in self.models.values())
        total_entities = sum(m.entity_count for m in self.models.values())
        total_size = sum(m.size_mb for m in self.models.values())
        
        return {
            "total_models": len(self.models),
            "total_files": total_files,
            "total_entities": total_entities,
            "total_size_mb": total_size,
            "cross_relationships": len(self.cross_model_relationships),
            "models": {
                model_id: {
                    "files": meta.file_count,
                    "entities": meta.entity_count,
                    "size_mb": meta.size_mb,
                    "last_updated": meta.last_updated
                }
                for model_id, meta in self.models.items()
            }
        }
    
    def find_similar_models(self, target_model: str, similarity_threshold: float = 0.7) -> List[Tuple[str, float]]:
        """Find models similar to the target model."""
        if target_model not in self.models:
            return []
        
        similar_models = []
        for model_id in self.models:
            if model_id != target_model:
                similarity = self._calculate_model_similarity(target_model, model_id)
                if similarity >= similarity_threshold:
                    similar_models.append((model_id, similarity))
        
        similar_models.sort(key=lambda x: x[1], reverse=True)
        return similar_models
    
    def _calculate_model_similarity(self, model_a: str, model_b: str) -> float:
        """Calculate similarity between two models."""
        cache_key = tuple(sorted([model_a, model_b]))
        if cache_key in self.similarity_cache:
            return self.similarity_cache[cache_key]
        
        # Calculate similarity based on:
        # 1. Shared entities/concepts
        # 2. Similar file structure
        # 3. Cross-model relationships
        
        meta_a = self.models[model_a]
        meta_b = self.models[model_b]
        
        # Basic metadata similarity
        size_similarity = 1 - abs(meta_a.size_mb - meta_b.size_mb) / max(meta_a.size_mb, meta_b.size_mb, 1)
        entity_similarity = 1 - abs(meta_a.entity_count - meta_b.entity_count) / max(meta_a.entity_count, meta_b.entity_count, 1)
        
        # Cross-relationship bonus
        relationships = [
            rel for rel in self.cross_model_relationships
            if (rel.source_model == model_a and rel.target_model == model_b) or
               (rel.source_model == model_b and rel.target_model == model_a)
        ]
        relationship_bonus = min(len(relationships) * 0.1, 0.4)
        
        similarity = (size_similarity * 0.3 + entity_similarity * 0.3 + relationship_bonus * 0.4)
        
        self.similarity_cache[cache_key] = similarity
        return similarity


def create_cross_model_analyzer() -> CrossModelAnalyzer:
    """Factory function to create cross-model analyzer."""
    return CrossModelAnalyzer()