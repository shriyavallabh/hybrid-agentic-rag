"""
Simple, Efficient Graph Builder for Hybrid RAG System
Creates a working knowledge graph from existing RAG chunks without timeouts.
"""
import os
import json
import pickle
import logging
from pathlib import Path
from typing import Dict, List, Any, Set
import time
import re

import networkx as nx
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


class SimpleGraphBuilder:
    """Build a working knowledge graph from existing RAG chunks."""
    
    def __init__(self, output_path: str):
        self.output_path = Path(output_path)
        self.output_path.mkdir(exist_ok=True)
        
    def build_graph_from_rag(self, rag_chunks: List[Dict]) -> nx.MultiDiGraph:
        """Build knowledge graph from existing RAG chunks."""
        logger.info(f"🔨 Building knowledge graph from {len(rag_chunks)} RAG chunks")
        
        # Create graph
        graph = nx.MultiDiGraph()
        
        # Extract entities from chunks
        entities = self._extract_entities_from_chunks(rag_chunks)
        logger.info(f"📊 Extracted {len(entities)} entities")
        
        # Add entities as nodes
        for entity in entities:
            graph.add_node(entity['id'], **entity)
        
        # Add relationships
        relationships = self._create_relationships(entities)
        logger.info(f"🔗 Created {len(relationships)} relationships")
        
        for rel in relationships:
            graph.add_edge(rel['source'], rel['target'], **rel)
        
        # Save graph
        self._save_graph(graph)
        
        logger.info(f"✅ Knowledge graph built: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
        return graph
    
    def _extract_entities_from_chunks(self, chunks: List[Dict]) -> List[Dict]:
        """Extract entities from RAG chunks using pattern matching."""
        entities = []
        entity_id = 0
        
        # Focus on PDF content (most important)
        pdf_chunks = [chunk for chunk in chunks if 'pdf' in chunk.get('source', '').lower()]
        
        for chunk in pdf_chunks[:100]:  # Limit to first 100 PDF chunks for efficiency
            content = chunk.get('content', '')
            source = chunk.get('source', '')
            
            # Extract different types of entities
            chunk_entities = []
            
            # 1. Extract figures and tables
            chunk_entities.extend(self._extract_figures_tables(content, source, entity_id))
            
            # 2. Extract authors
            chunk_entities.extend(self._extract_authors(content, source, entity_id))
            
            # 3. Extract models and algorithms
            chunk_entities.extend(self._extract_models_algorithms(content, source, entity_id))
            
            # 4. Extract datasets
            chunk_entities.extend(self._extract_datasets(content, source, entity_id))
            
            # 5. Extract performance metrics
            chunk_entities.extend(self._extract_metrics(content, source, entity_id))
            
            entities.extend(chunk_entities)
            entity_id += len(chunk_entities)
        
        # Remove duplicates
        unique_entities = self._deduplicate_entities(entities)
        return unique_entities
    
    def _extract_figures_tables(self, content: str, source: str, start_id: int) -> List[Dict]:
        """Extract figure and table references."""
        entities = []
        
        # Find figure references
        figure_pattern = r'(Figure|Fig\.?)\s*(\d+)'
        for match in re.finditer(figure_pattern, content, re.IGNORECASE):
            entities.append({
                'id': f"figure_{match.group(2)}",
                'name': f"Figure {match.group(2)}",
                'type': 'FIGURE',
                'source_file': source,
                'description': f"Figure {match.group(2)} from {Path(source).name}",
                'context': content[max(0, match.start()-100):match.end()+100]
            })
        
        # Find table references
        table_pattern = r'(Table|Tbl\.?)\s*(\d+)'
        for match in re.finditer(table_pattern, content, re.IGNORECASE):
            entities.append({
                'id': f"table_{match.group(2)}",
                'name': f"Table {match.group(2)}",
                'type': 'TABLE',
                'source_file': source,
                'description': f"Table {match.group(2)} from {Path(source).name}",
                'context': content[max(0, match.start()-100):match.end()+100]
            })
        
        return entities
    
    def _extract_authors(self, content: str, source: str, start_id: int) -> List[Dict]:
        """Extract author information."""
        entities = []
        
        # Look for author patterns
        author_patterns = [
            r'([A-Z][a-z]+\s+[A-Z][a-z]+)\s*\d+',  # "FirstName LastName1"
            r'([A-Z][a-z]+\s+[A-Z]\.\s+[A-Z][a-z]+)',  # "FirstName M. LastName"
        ]
        
        for pattern in author_patterns:
            for match in re.finditer(pattern, content):
                author_name = match.group(1)
                if len(author_name) > 3:  # Filter out short matches
                    entities.append({
                        'id': f"author_{author_name.replace(' ', '_').lower()}",
                        'name': author_name,
                        'type': 'AUTHOR',
                        'source_file': source,
                        'description': f"Author: {author_name}",
                        'context': content[max(0, match.start()-50):match.end()+50]
                    })
        
        return entities
    
    def _extract_models_algorithms(self, content: str, source: str, start_id: int) -> List[Dict]:
        """Extract models and algorithms."""
        entities = []
        
        # Model/algorithm keywords
        model_keywords = [
            'GraphRAG', 'RAG', 'GPT', 'LLM', 'transformer', 'BERT', 'algorithm',
            'model', 'neural network', 'deep learning', 'machine learning'
        ]
        
        content_lower = content.lower()
        for keyword in model_keywords:
            if keyword.lower() in content_lower:
                entities.append({
                    'id': f"model_{keyword.replace(' ', '_').lower()}",
                    'name': keyword,
                    'type': 'MODEL',
                    'source_file': source,
                    'description': f"Model/Algorithm: {keyword}",
                    'context': self._extract_context(content, keyword)
                })
        
        return entities
    
    def _extract_datasets(self, content: str, source: str, start_id: int) -> List[Dict]:
        """Extract dataset references."""
        entities = []
        
        # Dataset keywords
        dataset_keywords = [
            'dataset', 'data', 'corpus', 'collection', 'benchmark',
            'podcast', 'news', 'articles', 'transcripts'
        ]
        
        content_lower = content.lower()
        for keyword in dataset_keywords:
            if keyword.lower() in content_lower:
                entities.append({
                    'id': f"dataset_{keyword.replace(' ', '_').lower()}",
                    'name': keyword,
                    'type': 'DATASET',
                    'source_file': source,
                    'description': f"Dataset: {keyword}",
                    'context': self._extract_context(content, keyword)
                })
        
        return entities
    
    def _extract_metrics(self, content: str, source: str, start_id: int) -> List[Dict]:
        """Extract performance metrics."""
        entities = []
        
        # Metric keywords
        metric_keywords = [
            'accuracy', 'precision', 'recall', 'F1', 'performance', 'score',
            'win rate', 'percentage', 'comprehensiveness', 'diversity'
        ]
        
        content_lower = content.lower()
        for keyword in metric_keywords:
            if keyword.lower() in content_lower:
                entities.append({
                    'id': f"metric_{keyword.replace(' ', '_').lower()}",
                    'name': keyword,
                    'type': 'METRIC',
                    'source_file': source,
                    'description': f"Metric: {keyword}",
                    'context': self._extract_context(content, keyword)
                })
        
        return entities
    
    def _extract_context(self, content: str, keyword: str) -> str:
        """Extract context around a keyword."""
        pos = content.lower().find(keyword.lower())
        if pos != -1:
            start = max(0, pos - 100)
            end = min(len(content), pos + len(keyword) + 100)
            return content[start:end]
        return ""
    
    def _deduplicate_entities(self, entities: List[Dict]) -> List[Dict]:
        """Remove duplicate entities."""
        seen = set()
        unique_entities = []
        
        for entity in entities:
            key = (entity['name'].lower(), entity['type'])
            if key not in seen:
                seen.add(key)
                unique_entities.append(entity)
        
        return unique_entities
    
    def _create_relationships(self, entities: List[Dict]) -> List[Dict]:
        """Create relationships between entities."""
        relationships = []
        
        # Group entities by type
        entities_by_type = {}
        for entity in entities:
            entity_type = entity['type']
            if entity_type not in entities_by_type:
                entities_by_type[entity_type] = []
            entities_by_type[entity_type].append(entity)
        
        # Create relationships based on co-occurrence in same source
        entities_by_source = {}
        for entity in entities:
            source = entity['source_file']
            if source not in entities_by_source:
                entities_by_source[source] = []
            entities_by_source[source].append(entity)
        
        # Add relationships for entities in same source
        for source, source_entities in entities_by_source.items():
            for i, entity1 in enumerate(source_entities):
                for entity2 in source_entities[i+1:]:
                    relationships.append({
                        'source': entity1['id'],
                        'target': entity2['id'],
                        'type': 'CO_OCCURS_IN',
                        'source_file': source,
                        'description': f"Co-occurs in {Path(source).name}"
                    })
        
        return relationships
    
    def _save_graph(self, graph: nx.MultiDiGraph):
        """Save the graph to disk."""
        # Save as pickle
        graph_path = self.output_path / 'enhanced_graph.pkl'
        with open(graph_path, 'wb') as f:
            pickle.dump(graph, f)
        
        # Save metadata
        metadata = {
            'nodes': graph.number_of_nodes(),
            'edges': graph.number_of_edges(),
            'created_at': time.time(),
            'node_types': list(set(data.get('type', 'UNKNOWN') for _, data in graph.nodes(data=True)))
        }
        
        metadata_path = self.output_path / 'metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"💾 Graph saved to {graph_path}")
        logger.info(f"📋 Metadata saved to {metadata_path}")


def build_simple_graph(rag_path: str, output_path: str) -> nx.MultiDiGraph:
    """Build a simple knowledge graph from RAG chunks."""
    
    # Load RAG chunks
    chunks_path = Path(rag_path) / 'chunks.pkl'
    with open(chunks_path, 'rb') as f:
        chunks = pickle.load(f)
    
    # Build graph
    builder = SimpleGraphBuilder(output_path)
    graph = builder.build_graph_from_rag(chunks)
    
    return graph


if __name__ == "__main__":
    # Build the graph
    graph = build_simple_graph('rag_index', 'enhanced_kg')
    print(f"✅ Built graph with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges")