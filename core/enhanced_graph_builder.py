"""
Enhanced Graph Builder for Hybrid Graph-Guided RAG System
Creates rich, detailed knowledge graphs with comprehensive entity extraction
and intelligent relationship detection for cross-model analysis.
"""
import os
import json
import pickle
import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple, Set
import hashlib
import time

import networkx as nx
import pdfplumber
import ast
from openai import OpenAI
import re
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


class EnhancedEntityExtractor:
    """Extract rich entities with full context and metadata."""
    
    def __init__(self, client: OpenAI):
        self.client = client
        
    def extract_from_pdf(self, pdf_path: Path, content: str) -> List[Dict]:
        """Extract comprehensive entities from PDF content."""
        entities = []
        
        # Extract with GPT-4o for high accuracy
        extraction_prompt = f"""Extract ALL important entities from this technical document with rich detail and context.

Document: {pdf_path.name}
Content: {content[:4000]}...

Extract entities in these categories:
1. MODELS: Any machine learning models, algorithms, or systems mentioned
2. DATASETS: Any datasets, data sources, or data collections
3. METRICS: Performance measures, evaluation criteria, scores
4. AUTHORS: All authors and their affiliations
5. ORGANIZATIONS: Companies, universities, research institutions
6. METHODS: Techniques, approaches, methodologies
7. CONCEPTS: Key technical concepts, theories, principles
8. PERFORMANCE: Specific performance results, benchmarks, comparisons

For each entity provide:
- type: Entity category
- name: Clear entity name
- description: Rich description with context
- properties: Key properties/attributes
- source_context: Specific context where found
- page_reference: Page number if available

Output as JSON array of entities."""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                temperature=0.0,
                max_tokens=2000,
                messages=[{"role": "user", "content": extraction_prompt}]
            )
            
            result_text = response.choices[0].message.content.strip()
            
            # Parse JSON response
            if result_text.startswith('```json'):
                result_text = result_text.split('```json')[1].split('```')[0]
            elif result_text.startswith('```'):
                result_text = result_text.split('```')[1].split('```')[0]
            
            extracted_entities = json.loads(result_text)
            
            # Enrich with metadata
            for entity in extracted_entities:
                entity['source_file'] = str(pdf_path)
                entity['extraction_timestamp'] = time.time()
                entity['content_hash'] = hashlib.sha256(content.encode()).hexdigest()[:16]
                entities.append(entity)
                
            logger.info(f"Extracted {len(entities)} entities from {pdf_path.name}")
            return entities
            
        except Exception as e:
            logger.error(f"Entity extraction failed for {pdf_path}: {e}")
            return []
    
    def extract_from_python(self, py_path: Path, content: str) -> List[Dict]:
        """Extract entities from Python code with rich context."""
        entities = []
        
        try:
            # Parse AST for structured extraction
            tree = ast.parse(content)
            
            # Extract classes with rich context
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    entity = {
                        'type': 'CLASS',
                        'name': node.name,
                        'description': self._get_class_description(node, content),
                        'properties': {
                            'methods': [n.name for n in node.body if isinstance(n, ast.FunctionDef)],
                            'line_number': node.lineno,
                            'docstring': ast.get_docstring(node) or "",
                            'inheritance': [base.id if isinstance(base, ast.Name) else str(base) for base in node.bases]
                        },
                        'source_context': self._get_source_context(content, node.lineno),
                        'source_file': str(py_path),
                        'extraction_timestamp': time.time()
                    }
                    entities.append(entity)
                
                elif isinstance(node, ast.FunctionDef):
                    entity = {
                        'type': 'FUNCTION',
                        'name': node.name,
                        'description': self._get_function_description(node, content),
                        'properties': {
                            'parameters': [arg.arg for arg in node.args.args],
                            'line_number': node.lineno,
                            'docstring': ast.get_docstring(node) or "",
                            'returns': self._analyze_return_type(node)
                        },
                        'source_context': self._get_source_context(content, node.lineno),
                        'source_file': str(py_path),
                        'extraction_timestamp': time.time()
                    }
                    entities.append(entity)
            
            # Extract algorithmic concepts using GPT-4o
            concepts = self._extract_algorithmic_concepts(py_path, content)
            entities.extend(concepts)
            
            logger.info(f"Extracted {len(entities)} entities from {py_path.name}")
            return entities
            
        except Exception as e:
            logger.error(f"Python entity extraction failed for {py_path}: {e}")
            return []
    
    def _get_class_description(self, node: ast.ClassDef, content: str) -> str:
        """Generate rich description for class."""
        docstring = ast.get_docstring(node) or ""
        methods = [n.name for n in node.body if isinstance(n, ast.FunctionDef)]
        
        return f"Class {node.name} with {len(methods)} methods. {docstring[:200]}..."
    
    def _get_function_description(self, node: ast.FunctionDef, content: str) -> str:
        """Generate rich description for function."""
        docstring = ast.get_docstring(node) or ""
        params = [arg.arg for arg in node.args.args]
        
        return f"Function {node.name}({', '.join(params)}). {docstring[:200]}..."
    
    def _get_source_context(self, content: str, line_number: int, context_lines: int = 3) -> str:
        """Get surrounding source code context."""
        lines = content.split('\n')
        start = max(0, line_number - context_lines - 1)
        end = min(len(lines), line_number + context_lines)
        
        return '\n'.join(lines[start:end])
    
    def _analyze_return_type(self, node: ast.FunctionDef) -> str:
        """Analyze function return type."""
        if node.returns:
            return ast.unparse(node.returns) if hasattr(ast, 'unparse') else "Unknown"
        return "None"
    
    def _extract_algorithmic_concepts(self, py_path: Path, content: str) -> List[Dict]:
        """Extract algorithmic concepts and methods using GPT-4o."""
        if len(content) > 8000:  # Limit content for API
            content = content[:8000] + "..."
        
        try:
            concept_prompt = f"""Analyze this Python code and extract key algorithmic concepts, techniques, and methods.

File: {py_path.name}
Code: {content}

Extract:
1. ALGORITHMS: Any algorithms implemented or referenced
2. TECHNIQUES: ML/AI techniques, data processing methods
3. PATTERNS: Design patterns, architectural approaches
4. CONCEPTS: Key computational concepts or principles

For each, provide:
- type: Category (ALGORITHM, TECHNIQUE, PATTERN, CONCEPT)  
- name: Clear name
- description: What it does and why it's important
- properties: Key characteristics

Output as JSON array."""

            response = self.client.chat.completions.create(
                model="gpt-4o",
                temperature=0.0,
                max_tokens=1500,
                messages=[{"role": "user", "content": concept_prompt}]
            )
            
            result_text = response.choices[0].message.content.strip()
            
            # Parse JSON response
            if result_text.startswith('```json'):
                result_text = result_text.split('```json')[1].split('```')[0]
            elif result_text.startswith('```'):
                result_text = result_text.split('```')[1].split('```')[0]
            
            concepts = json.loads(result_text)
            
            # Enrich with metadata
            for concept in concepts:
                concept['source_file'] = str(py_path)
                concept['extraction_timestamp'] = time.time()
            
            return concepts
            
        except Exception as e:
            logger.error(f"Concept extraction failed for {py_path}: {e}")
            return []


class EnhancedRelationshipDetector:
    """Detect intelligent relationships between entities."""
    
    def __init__(self, client: OpenAI):
        self.client = client
    
    def detect_relationships(self, entities: List[Dict]) -> List[Dict]:
        """Detect relationships between entities using AI reasoning."""
        relationships = []
        
        # Group entities by source for cross-document analysis
        entities_by_source = {}
        for entity in entities:
            source = entity.get('source_file', 'unknown')
            if source not in entities_by_source:
                entities_by_source[source] = []
            entities_by_source[source].append(entity)
        
        # Intra-document relationships
        for source, source_entities in entities_by_source.items():
            intra_relationships = self._detect_intra_document_relationships(source_entities)
            relationships.extend(intra_relationships)
        
        # Cross-document relationships (key for cross-model analysis)
        cross_relationships = self._detect_cross_document_relationships(entities_by_source)
        relationships.extend(cross_relationships)
        
        logger.info(f"Detected {len(relationships)} relationships")
        return relationships
    
    def _detect_intra_document_relationships(self, entities: List[Dict]) -> List[Dict]:
        """Detect relationships within a single document."""
        relationships = []
        
        try:
            # Create entity summary for GPT-4o
            entity_summary = []
            for i, entity in enumerate(entities):
                entity_summary.append(f"{i}: {entity['type']} - {entity['name']} - {entity.get('description', '')[:100]}")
            
            if len(entity_summary) > 20:  # Limit for API
                entity_summary = entity_summary[:20]
            
            summary_text = '\n'.join(entity_summary)
            
            relationship_prompt = f"""Analyze these entities from the same document and identify meaningful relationships.

Entities:
{summary_text}

Identify relationships like:
- USES: Entity A uses Entity B
- IMPLEMENTS: Entity A implements Entity B  
- COMPARES_TO: Entity A is compared to Entity B
- BUILDS_ON: Entity A builds on Entity B
- HAS_METRIC: Entity A has metric Entity B
- DERIVED_FROM: Entity A is derived from Entity B

Output as JSON array with format:
[{{"source_id": 0, "target_id": 1, "relationship": "USES", "description": "Why this relationship exists"}}]"""

            response = self.client.chat.completions.create(
                model="gpt-4o",
                temperature=0.0,
                max_tokens=1500,
                messages=[{"role": "user", "content": relationship_prompt}]
            )
            
            result_text = response.choices[0].message.content.strip()
            
            # Parse JSON response
            if result_text.startswith('```json'):
                result_text = result_text.split('```json')[1].split('```')[0]
            elif result_text.startswith('```'):
                result_text = result_text.split('```')[1].split('```')[0]
            
            detected_relationships = json.loads(result_text)
            
            # Convert to full relationship objects
            for rel in detected_relationships:
                if (rel['source_id'] < len(entities) and rel['target_id'] < len(entities)):
                    source_entity = entities[rel['source_id']]
                    target_entity = entities[rel['target_id']]
                    
                    relationship = {
                        'source': source_entity['name'],
                        'target': target_entity['name'],
                        'relationship': rel['relationship'],
                        'description': rel['description'],
                        'source_entity': source_entity,
                        'target_entity': target_entity,
                        'confidence': 0.8,  # High confidence for intra-document
                        'detection_timestamp': time.time()
                    }
                    relationships.append(relationship)
            
            return relationships
            
        except Exception as e:
            logger.error(f"Intra-document relationship detection failed: {e}")
            return []
    
    def _detect_cross_document_relationships(self, entities_by_source: Dict) -> List[Dict]:
        """Detect relationships across different documents (crucial for cross-model analysis)."""
        relationships = []
        
        # Focus on cross-model relationships
        model_entities = {}
        for source, entities in entities_by_source.items():
            for entity in entities:
                if entity['type'] in ['MODELS', 'MODEL', 'ALGORITHM']:
                    if source not in model_entities:
                        model_entities[source] = []
                    model_entities[source].append(entity)
        
        # Detect cross-model relationships
        sources = list(model_entities.keys())
        for i, source1 in enumerate(sources):
            for source2 in sources[i+1:]:
                cross_rels = self._analyze_cross_model_relationships(
                    model_entities[source1], 
                    model_entities[source2],
                    source1, source2
                )
                relationships.extend(cross_rels)
        
        return relationships
    
    def _analyze_cross_model_relationships(self, models1: List[Dict], models2: List[Dict], 
                                         source1: str, source2: str) -> List[Dict]:
        """Analyze relationships between models from different sources."""
        relationships = []
        
        try:
            # Prepare model summaries
            models1_summary = []
            for i, model in enumerate(models1):
                models1_summary.append(f"A{i}: {model['name']} - {model.get('description', '')[:150]}")
            
            models2_summary = []  
            for i, model in enumerate(models2):
                models2_summary.append(f"B{i}: {model['name']} - {model.get('description', '')[:150]}")
            
            cross_analysis_prompt = f"""Analyze these models from different sources for relationships.

Source 1 ({Path(source1).name}):
{chr(10).join(models1_summary)}

Source 2 ({Path(source2).name}):
{chr(10).join(models2_summary)}

Identify cross-model relationships:
- COMPARES_TO: Models that are compared or benchmarked against each other
- BUILDS_ON: One model builds on another's approach
- ALTERNATIVE_TO: Models that solve similar problems differently  
- OUTPERFORMS: One model outperforms another
- EXTENDS: One model extends another's capabilities

Output JSON: [{{"source_id": "A0", "target_id": "B1", "relationship": "COMPARES_TO", "description": "Evidence"}}]"""

            response = self.client.chat.completions.create(
                model="gpt-4o",
                temperature=0.0,
                max_tokens=1000,
                messages=[{"role": "user", "content": cross_analysis_prompt}]
            )
            
            result_text = response.choices[0].message.content.strip()
            
            # Parse JSON response
            if result_text.startswith('```json'):
                result_text = result_text.split('```json')[1].split('```')[0]
            elif result_text.startswith('```'):
                result_text = result_text.split('```')[1].split('```')[0]
            
            detected_relationships = json.loads(result_text)
            
            # Convert to relationship objects
            for rel in detected_relationships:
                source_id = rel['source_id']
                target_id = rel['target_id']
                
                # Find entities
                source_entity = None
                target_entity = None
                
                if source_id.startswith('A'):
                    idx = int(source_id[1:])
                    if idx < len(models1):
                        source_entity = models1[idx]
                
                if target_id.startswith('B'):
                    idx = int(target_id[1:])
                    if idx < len(models2):
                        target_entity = models2[idx]
                
                if source_entity and target_entity:
                    relationship = {
                        'source': source_entity['name'],
                        'target': target_entity['name'],
                        'relationship': rel['relationship'],
                        'description': rel['description'],
                        'source_entity': source_entity,
                        'target_entity': target_entity,
                        'confidence': 0.7,  # Lower confidence for cross-document
                        'cross_document': True,
                        'detection_timestamp': time.time()
                    }
                    relationships.append(relationship)
            
            return relationships
            
        except Exception as e:
            logger.error(f"Cross-model relationship detection failed: {e}")
            return []


class EnhancedGraphBuilder:
    """Build enhanced knowledge graph with rich nodes and intelligent relationships."""
    
    def __init__(self, openai_api_key: str, output_path: str):
        self.client = OpenAI(api_key=openai_api_key)
        self.output_path = Path(output_path)
        self.output_path.mkdir(exist_ok=True)
        
        self.entity_extractor = EnhancedEntityExtractor(self.client)
        self.relationship_detector = EnhancedRelationshipDetector(self.client)
        
        self.graph = nx.MultiDiGraph()
        self.entities = []
        self.relationships = []
    
    def build_enhanced_graph(self, knowledge_base_path: str):
        """Build comprehensive enhanced knowledge graph."""
        logger.info("Building enhanced knowledge graph...")
        
        kb_path = Path(knowledge_base_path)
        
        # Process all files comprehensively
        for root, dirs, files in os.walk(kb_path):
            for file in files:
                filepath = Path(root) / file
                try:
                    if file.endswith('.pdf'):
                        self._process_pdf_enhanced(filepath)
                    elif file.endswith('.py'):
                        self._process_python_enhanced(filepath)
                    elif file.endswith('.ipynb'):
                        self._process_notebook_enhanced(filepath)
                except Exception as e:
                    logger.warning(f"Failed to process {filepath}: {e}")
        
        logger.info(f"Extracted {len(self.entities)} entities")
        
        # Detect relationships
        self.relationships = self.relationship_detector.detect_relationships(self.entities)
        
        # Build NetworkX graph
        self._build_networkx_graph()
        
        # Save enhanced graph
        self._save_enhanced_graph()
        
        logger.info("Enhanced knowledge graph built successfully!")
    
    def _process_pdf_enhanced(self, pdf_path: Path):
        """Process PDF with enhanced entity extraction."""
        logger.info(f"Processing PDF: {pdf_path}")
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                full_content = ""
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        full_content += text + "\n"
                
                if full_content.strip():
                    entities = self.entity_extractor.extract_from_pdf(pdf_path, full_content)
                    self.entities.extend(entities)
                    
        except Exception as e:
            logger.error(f"PDF processing failed: {e}")
    
    def _process_python_enhanced(self, py_path: Path):
        """Process Python file with enhanced extraction."""
        logger.info(f"Processing Python: {py_path}")
        
        try:
            with open(py_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            entities = self.entity_extractor.extract_from_python(py_path, content)
            self.entities.extend(entities)
            
        except Exception as e:
            logger.error(f"Python processing failed: {e}")
    
    def _process_notebook_enhanced(self, nb_path: Path):
        """Process Jupyter notebook with enhanced extraction."""
        logger.info(f"Processing notebook: {nb_path}")
        
        try:
            with open(nb_path, 'r', encoding='utf-8') as f:
                notebook = json.load(f)
            
            # Extract from code cells
            code_content = ""
            for cell in notebook.get('cells', []):
                if cell.get('cell_type') == 'code':
                    source = ''.join(cell.get('source', []))
                    code_content += source + "\n"
            
            if code_content.strip():
                entities = self.entity_extractor.extract_from_python(nb_path, code_content)
                self.entities.extend(entities)
                
        except Exception as e:
            logger.error(f"Notebook processing failed: {e}")
    
    def _build_networkx_graph(self):
        """Build NetworkX graph from entities and relationships."""
        logger.info("Building NetworkX graph...")
        
        # Add nodes with rich attributes
        for entity in self.entities:
            node_id = f"{entity['type']}_{entity['name']}_{entity.get('source_file', 'unknown')}"
            
            self.graph.add_node(
                node_id,
                **entity  # Rich node attributes
            )
        
        # Add edges with relationship data
        for relationship in self.relationships:
            source_node = f"{relationship['source_entity']['type']}_{relationship['source']}_" + \
                         f"{relationship['source_entity'].get('source_file', 'unknown')}"
            target_node = f"{relationship['target_entity']['type']}_{relationship['target']}_" + \
                         f"{relationship['target_entity'].get('source_file', 'unknown')}"
            
            if source_node in self.graph and target_node in self.graph:
                self.graph.add_edge(
                    source_node,
                    target_node,
                    relationship=relationship['relationship'],
                    description=relationship['description'],
                    confidence=relationship['confidence'],
                    **relationship
                )
        
        logger.info(f"Graph built: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")
    
    def _save_enhanced_graph(self):
        """Save the enhanced graph and metadata."""
        logger.info("Saving enhanced knowledge graph...")
        
        # Save graph
        with open(self.output_path / 'enhanced_graph.pkl', 'wb') as f:
            pickle.dump(self.graph, f)
        
        # Save entities
        with open(self.output_path / 'entities.json', 'w') as f:
            json.dump(self.entities, f, indent=2, default=str)
        
        # Save relationships
        with open(self.output_path / 'relationships.json', 'w') as f:
            json.dump(self.relationships, f, indent=2, default=str)
        
        # Save metadata
        metadata = {
            'total_entities': len(self.entities),
            'total_relationships': len(self.relationships),
            'nodes': self.graph.number_of_nodes(),
            'edges': self.graph.number_of_edges(),
            'entity_types': {},
            'relationship_types': {},
            'build_timestamp': time.time()
        }
        
        # Count entity types
        for entity in self.entities:
            entity_type = entity['type']
            metadata['entity_types'][entity_type] = metadata['entity_types'].get(entity_type, 0) + 1
        
        # Count relationship types
        for rel in self.relationships:
            rel_type = rel['relationship']
            metadata['relationship_types'][rel_type] = metadata['relationship_types'].get(rel_type, 0) + 1
        
        with open(self.output_path / 'enhanced_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info("Enhanced knowledge graph saved successfully!")


if __name__ == "__main__":
    builder = EnhancedGraphBuilder(
        openai_api_key=os.getenv('OPENAI_API_KEY'),
        output_path='enhanced_kg'
    )
    
    builder.build_enhanced_graph('knowledge_base')