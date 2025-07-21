#!/usr/bin/env python3
"""
Claude-Powered Direct Analysis System
Analyzes codebases directly without external APIs using structured analysis
"""

import os
import json
import pickle
import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple, Set
import networkx as nx
import hashlib
import time
from collections import defaultdict, Counter
import re

logger = logging.getLogger(__name__)

class CodebaseAnalyzer:
    """Direct codebase analysis using structured parsing and pattern recognition."""
    
    def __init__(self, knowledge_base_path: str = "knowledge_base"):
        self.kb_path = Path(knowledge_base_path)
        self.graph = nx.MultiDiGraph()
        self.entities = {}
        self.relationships = []
        self.file_contents = {}
        self.chunks = []
        
    def analyze_model(self, model_id: str) -> Dict[str, Any]:
        """Analyze a complete model directory."""
        logger.info(f"🔍 Starting direct analysis of {model_id}")
        
        model_path = self.kb_path / model_id
        if not model_path.exists():
            logger.error(f"Model path {model_path} does not exist")
            return {}
        
        # Phase 1: File Discovery and Content Loading
        logger.info("📁 Phase 1: Discovering and loading files...")
        files = self._discover_files(model_path)
        logger.info(f"Found {len(files)} files to analyze")
        
        # Phase 2: Content Analysis and Entity Extraction
        logger.info("🔍 Phase 2: Analyzing file contents and extracting entities...")
        for file_path in files:
            self._analyze_file(file_path)
        
        # Phase 3: Relationship Detection
        logger.info("🔗 Phase 3: Detecting relationships between entities...")
        self._detect_relationships()
        
        # Phase 4: Graph Construction
        logger.info("📊 Phase 4: Building knowledge graph...")
        self._build_graph()
        
        # Phase 5: Chunking for RAG
        logger.info("📝 Phase 5: Creating semantic chunks for RAG...")
        self._create_semantic_chunks()
        
        # Phase 6: Save Results
        logger.info("💾 Phase 6: Saving analysis results...")
        results = self._save_results(model_id)
        
        logger.info(f"✅ Analysis complete: {len(self.entities)} entities, {len(self.relationships)} relationships")
        return results
    
    def _discover_files(self, model_path: Path) -> List[Path]:
        """Discover all analyzable files in the model directory."""
        analyzable_extensions = {
            '.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.cpp', '.c', '.h', 
            '.cs', '.go', '.rs', '.rb', '.php', '.scala', '.kt', '.swift',
            '.md', '.txt', '.rst', '.json', '.yaml', '.yml', '.toml', '.cfg',
            '.ini', '.conf', '.sh', '.bat', '.ps1', '.sql', '.html', '.css'
        }
        
        files = []
        for file_path in model_path.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in analyzable_extensions:
                # Skip large binary files and common build artifacts
                if file_path.stat().st_size < 10 * 1024 * 1024:  # 10MB limit
                    files.append(file_path)
        
        return sorted(files)
    
    def _analyze_file(self, file_path: Path):
        """Analyze a single file and extract entities."""
        try:
            # Read file content
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            self.file_contents[str(file_path)] = content
            
            # Extract entities based on file type
            if file_path.suffix == '.py':
                self._analyze_python_file(file_path, content)
            elif file_path.suffix in ['.js', '.ts', '.jsx', '.tsx']:
                self._analyze_javascript_file(file_path, content)
            elif file_path.suffix == '.md':
                self._analyze_markdown_file(file_path, content)
            elif file_path.suffix in ['.json', '.yaml', '.yml']:
                self._analyze_config_file(file_path, content)
            else:
                self._analyze_generic_file(file_path, content)
                
        except Exception as e:
            logger.warning(f"Failed to analyze {file_path}: {e}")
    
    def _analyze_python_file(self, file_path: Path, content: str):
        """Extract entities from Python files."""
        # Extract classes
        class_matches = re.finditer(r'class\s+(\w+)(?:\([^)]*\))?:', content)
        for match in class_matches:
            class_name = match.group(1)
            entity_id = f"class_{class_name}_{hash(str(file_path))}"
            self.entities[entity_id] = {
                'id': entity_id,
                'name': class_name,
                'type': 'Class',
                'file': str(file_path),
                'description': f"Python class {class_name} in {file_path.name}",
                'language': 'Python'
            }
        
        # Extract functions
        func_matches = re.finditer(r'def\s+(\w+)\s*\([^)]*\):', content)
        for match in func_matches:
            func_name = match.group(1)
            entity_id = f"function_{func_name}_{hash(str(file_path))}"
            self.entities[entity_id] = {
                'id': entity_id,
                'name': func_name,
                'type': 'Function',
                'file': str(file_path),
                'description': f"Python function {func_name} in {file_path.name}",
                'language': 'Python'
            }
        
        # Extract imports
        import_matches = re.finditer(r'(?:from\s+(\S+)\s+)?import\s+([^#\n]+)', content)
        for match in import_matches:
            module = match.group(1) if match.group(1) else match.group(2).split()[0]
            entity_id = f"module_{module}_{hash(str(file_path))}"
            self.entities[entity_id] = {
                'id': entity_id,
                'name': module,
                'type': 'Module',
                'file': str(file_path),
                'description': f"Python module {module} imported in {file_path.name}",
                'language': 'Python'
            }
    
    def _analyze_javascript_file(self, file_path: Path, content: str):
        """Extract entities from JavaScript/TypeScript files."""
        # Extract classes
        class_matches = re.finditer(r'class\s+(\w+)', content)
        for match in class_matches:
            class_name = match.group(1)
            entity_id = f"class_{class_name}_{hash(str(file_path))}"
            self.entities[entity_id] = {
                'id': entity_id,
                'name': class_name,
                'type': 'Class',
                'file': str(file_path),
                'description': f"JavaScript class {class_name} in {file_path.name}",
                'language': 'JavaScript'
            }
        
        # Extract functions
        func_matches = re.finditer(r'(?:function\s+(\w+)|(\w+)\s*:\s*function|const\s+(\w+)\s*=\s*(?:async\s+)?\([^)]*\)\s*=>)', content)
        for match in func_matches:
            func_name = match.group(1) or match.group(2) or match.group(3)
            if func_name:
                entity_id = f"function_{func_name}_{hash(str(file_path))}"
                self.entities[entity_id] = {
                    'id': entity_id,
                    'name': func_name,
                    'type': 'Function',
                    'file': str(file_path),
                    'description': f"JavaScript function {func_name} in {file_path.name}",
                    'language': 'JavaScript'
                }
    
    def _analyze_markdown_file(self, file_path: Path, content: str):
        """Extract entities from Markdown files."""
        # Extract headers as concepts
        header_matches = re.finditer(r'^(#{1,6})\s+(.+)$', content, re.MULTILINE)
        for match in header_matches:
            level = len(match.group(1))
            title = match.group(2).strip()
            entity_id = f"concept_{title}_{hash(str(file_path))}"
            self.entities[entity_id] = {
                'id': entity_id,
                'name': title,
                'type': 'Concept',
                'file': str(file_path),
                'description': f"Documentation concept: {title} (H{level}) in {file_path.name}",
                'language': 'Markdown',
                'level': level
            }
        
        # Extract code blocks as examples
        code_matches = re.finditer(r'```(\w+)?\n(.*?)```', content, re.DOTALL)
        for i, match in enumerate(code_matches):
            lang = match.group(1) or 'text'
            entity_id = f"code_example_{i}_{hash(str(file_path))}"
            self.entities[entity_id] = {
                'id': entity_id,
                'name': f"Code Example {i+1}",
                'type': 'CodeExample',
                'file': str(file_path),
                'description': f"{lang} code example in {file_path.name}",
                'language': lang
            }
    
    def _analyze_config_file(self, file_path: Path, content: str):
        """Extract entities from configuration files."""
        try:
            if file_path.suffix == '.json':
                data = json.loads(content)
            else:
                # Simple YAML parsing for basic key extraction
                data = {}
                for line in content.split('\n'):
                    if ':' in line and not line.strip().startswith('#'):
                        key = line.split(':')[0].strip()
                        if key and not key.startswith(' '):
                            data[key] = True
            
            for key in data:
                entity_id = f"config_{key}_{hash(str(file_path))}"
                self.entities[entity_id] = {
                    'id': entity_id,
                    'name': key,
                    'type': 'Configuration',
                    'file': str(file_path),
                    'description': f"Configuration setting {key} in {file_path.name}",
                    'language': 'Config'
                }
        except Exception as e:
            logger.debug(f"Failed to parse config {file_path}: {e}")
    
    def _analyze_generic_file(self, file_path: Path, content: str):
        """Extract basic entities from generic text files."""
        # Create a file entity
        entity_id = f"file_{file_path.stem}_{hash(str(file_path))}"
        self.entities[entity_id] = {
            'id': entity_id,
            'name': file_path.name,
            'type': 'File',
            'file': str(file_path),
            'description': f"File {file_path.name} ({file_path.suffix})",
            'language': 'Generic'
        }
    
    def _detect_relationships(self):
        """Detect relationships between entities."""
        logger.info("🔗 Detecting entity relationships...")
        
        # Group entities by file
        file_entities = defaultdict(list)
        for entity in self.entities.values():
            file_entities[entity['file']].append(entity)
        
        # Create relationships within files
        for file_path, entities in file_entities.items():
            for i, entity1 in enumerate(entities):
                for entity2 in entities[i+1:]:
                    # Same file relationship
                    rel_id = f"rel_{entity1['id']}_{entity2['id']}"
                    self.relationships.append({
                        'id': rel_id,
                        'source': entity1['id'],
                        'target': entity2['id'],
                        'type': 'CO_LOCATED',
                        'description': f"{entity1['name']} and {entity2['name']} are in the same file",
                        'confidence': 0.8
                    })
        
        # Detect import/usage relationships
        for file_path, content in self.file_contents.items():
            file_entities_list = file_entities[file_path]
            
            # Look for entity names mentioned in other files
            for entity in self.entities.values():
                entity_name = str(entity.get('name', ''))
                if entity['file'] != file_path and isinstance(content, str) and entity_name and entity_name in content:
                    # Create usage relationship
                    for file_entity in file_entities_list:
                        rel_id = f"rel_{file_entity['id']}_{entity['id']}"
                        self.relationships.append({
                            'id': rel_id,
                            'source': file_entity['id'],
                            'target': entity['id'],
                            'type': 'USES',
                            'description': f"{file_entity['name']} references {entity['name']}",
                            'confidence': 0.6
                        })
    
    def _build_graph(self):
        """Build NetworkX graph from entities and relationships."""
        # Add nodes
        for entity in self.entities.values():
            self.graph.add_node(entity['id'], **entity)
        
        # Add edges
        for rel in self.relationships:
            self.graph.add_edge(
                rel['source'],
                rel['target'],
                id=rel['id'],
                type=rel['type'],
                description=rel['description'],
                confidence=rel['confidence']
            )
        
        logger.info(f"📊 Built graph: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")
    
    def _create_semantic_chunks(self):
        """Create semantic chunks for RAG indexing."""
        logger.info("📝 Creating semantic chunks...")
        
        for file_path, content in self.file_contents.items():
            # Split content into logical chunks
            if content.strip():
                # For code files, split by functions/classes
                if file_path.endswith('.py'):
                    chunks = self._chunk_python_code(content, file_path)
                elif file_path.endswith('.md'):
                    chunks = self._chunk_markdown(content, file_path)
                else:
                    chunks = self._chunk_generic_text(content, file_path)
                
                self.chunks.extend(chunks)
    
    def _chunk_python_code(self, content: str, file_path: str) -> List[Dict]:
        """Chunk Python code by functions and classes."""
        chunks = []
        lines = content.split('\n')
        current_chunk = []
        current_type = None
        current_name = None
        
        for line in lines:
            # Detect function or class start
            if re.match(r'^(class|def)\s+\w+', line):
                # Save previous chunk
                if current_chunk:
                    chunks.append({
                        'id': f"chunk_{len(chunks)}_{hash(file_path)}",
                        'content': '\n'.join(current_chunk),
                        'file': file_path,
                        'type': current_type or 'code',
                        'name': current_name or 'unknown',
                        'size': len('\n'.join(current_chunk))
                    })
                
                # Start new chunk
                current_chunk = [line]
                if line.startswith('class'):
                    current_type = 'class'
                    current_name = re.search(r'class\s+(\w+)', line).group(1)
                else:
                    current_type = 'function'
                    current_name = re.search(r'def\s+(\w+)', line).group(1)
            else:
                current_chunk.append(line)
        
        # Save final chunk
        if current_chunk:
            chunks.append({
                'id': f"chunk_{len(chunks)}_{hash(file_path)}",
                'content': '\n'.join(current_chunk),
                'file': file_path,
                'type': current_type or 'code',
                'name': current_name or 'unknown',
                'size': len('\n'.join(current_chunk))
            })
        
        return chunks
    
    def _chunk_markdown(self, content: str, file_path: str) -> List[Dict]:
        """Chunk Markdown by sections."""
        chunks = []
        sections = re.split(r'^(#{1,6}\s+.+)$', content, flags=re.MULTILINE)
        
        current_chunk = ""
        current_header = None
        
        for section in sections:
            if re.match(r'^#{1,6}\s+', section):
                # Save previous chunk
                if current_chunk.strip():
                    chunks.append({
                        'id': f"chunk_{len(chunks)}_{hash(file_path)}",
                        'content': current_chunk.strip(),
                        'file': file_path,
                        'type': 'section',
                        'name': current_header or 'Introduction',
                        'size': len(current_chunk)
                    })
                
                # Start new chunk
                current_header = section.strip()
                current_chunk = section + '\n'
            else:
                current_chunk += section
        
        # Save final chunk
        if current_chunk.strip():
            chunks.append({
                'id': f"chunk_{len(chunks)}_{hash(file_path)}",
                'content': current_chunk.strip(),
                'file': file_path,
                'type': 'section',
                'name': current_header or 'Content',
                'size': len(current_chunk)
            })
        
        return chunks
    
    def _chunk_generic_text(self, content: str, file_path: str) -> List[Dict]:
        """Chunk generic text files."""
        # Simple sentence-based chunking
        sentences = re.split(r'[.!?]+', content)
        chunk_size = 5  # sentences per chunk
        
        chunks = []
        for i in range(0, len(sentences), chunk_size):
            chunk_sentences = sentences[i:i+chunk_size]
            chunk_content = '. '.join(s.strip() for s in chunk_sentences if s.strip())
            
            if chunk_content:
                chunks.append({
                    'id': f"chunk_{len(chunks)}_{hash(file_path)}",
                    'content': chunk_content,
                    'file': file_path,
                    'type': 'text',
                    'name': f"Section {i//chunk_size + 1}",
                    'size': len(chunk_content)
                })
        
        return chunks
    
    def _save_results(self, model_id: str) -> Dict[str, Any]:
        """Save analysis results to disk."""
        # Create output directories
        kg_dir = Path("enhanced_kg") / model_id / "graph"
        rag_dir = Path("rag_index") / model_id
        
        kg_dir.mkdir(parents=True, exist_ok=True)
        rag_dir.mkdir(parents=True, exist_ok=True)
        
        # Save knowledge graph
        graph_path = kg_dir / "enhanced_graph.pkl"
        with open(graph_path, 'wb') as f:
            pickle.dump(self.graph, f)
        
        # Save graph metadata
        metadata = {
            'nodes': self.graph.number_of_nodes(),
            'edges': self.graph.number_of_edges(),
            'created_at': time.time(),
            'model_id': model_id,
            'analyzer': 'claude_direct'
        }
        
        with open(kg_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Save RAG chunks
        chunks_path = rag_dir / "chunks.pkl"
        with open(chunks_path, 'wb') as f:
            pickle.dump(self.chunks, f)
        
        # Save chunk metadata
        chunk_metadata = {
            'total_chunks': len(self.chunks),
            'total_size': sum(chunk['size'] for chunk in self.chunks),
            'created_at': time.time(),
            'model_id': model_id,
            'analyzer': 'claude_direct'
        }
        
        with open(rag_dir / "metadata.json", 'w') as f:
            json.dump(chunk_metadata, f, indent=2)
        
        # Return summary
        return {
            'model_id': model_id,
            'entities': len(self.entities),
            'relationships': len(self.relationships),
            'chunks': len(self.chunks),
            'files_analyzed': len(self.file_contents),
            'graph_path': str(graph_path),
            'chunks_path': str(chunks_path)
        }


def analyze_model_direct(model_id: str) -> Dict[str, Any]:
    """Direct analysis entry point."""
    analyzer = CodebaseAnalyzer()
    return analyzer.analyze_model(model_id)


if __name__ == "__main__":
    # Test analysis
    result = analyze_model_direct("model_1")
    print(f"Analysis complete: {result}")