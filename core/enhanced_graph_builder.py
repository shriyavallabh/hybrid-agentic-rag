"""
AI-First Enhanced Graph Builder for Comprehensive Knowledge Graphs
Uses LLMs to analyze all code types: Python, R, Java, C++, JavaScript, etc.
"""
import os
import json
import pickle
import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple, Set
import time

import networkx as nx
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

# Optional nbformat import for notebook processing
try:
    import nbformat
    NBFORMAT_AVAILABLE = True
except ImportError:
    NBFORMAT_AVAILABLE = False
    logger.warning("nbformat not available - Jupyter notebook processing disabled")


class AIGraphBuilder:
    """Build comprehensive knowledge graphs using AI to analyze code and documents."""
    
    def __init__(self, output_path: str):
        self.output_path = Path(output_path)
        self.output_path.mkdir(exist_ok=True)
        
        # Initialize OpenAI client
        self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        
        # Load existing graph if available
        self.graph = self._load_existing_graph()
        
        # Entity tracking for cross-references
        self.entity_registry = {}
        self.next_entity_id = self._get_next_entity_id()
        
    def _load_existing_graph(self) -> nx.MultiDiGraph:
        """Load existing graph or create new one."""
        graph_path = self.output_path / 'enhanced_graph.pkl'
        if graph_path.exists():
            try:
                with open(graph_path, 'rb') as f:
                    graph = pickle.load(f)
                logger.info(f"📊 Loaded existing graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
                return graph
            except Exception as e:
                logger.warning(f"Failed to load existing graph: {e}")
        
        return nx.MultiDiGraph()
    
    def _get_next_entity_id(self) -> int:
        """Get next available entity ID."""
        if self.graph.number_of_nodes() == 0:
            return 1
        
        max_id = 0
        for node_id in self.graph.nodes():
            if isinstance(node_id, str) and node_id.startswith('entity_'):
                try:
                    entity_num = int(node_id.split('_')[1])
                    max_id = max(max_id, entity_num)
                except (IndexError, ValueError):
                    pass
        
        return max_id + 1
    
    def process_files(self, files: List[Path]) -> Tuple[List[Dict], List[Dict]]:
        """Process multiple files and extract entities/relationships using AI."""
        logger.info(f"🤖 AI-analyzing {len(files)} files for graph extraction...")
        
        all_entities = []
        all_relationships = []
        
        # Group files by type for efficient processing
        file_groups = self._group_files_by_type(files)
        
        # Process each file type using AI
        for file_type, file_list in file_groups.items():
            if not file_list:
                continue
                
            logger.info(f"📁 AI-processing {len(file_list)} {file_type} files...")
            
            try:
                # Process files in smaller batches to avoid token limits
                batch_size = 5 if file_type == 'code' else 10
                for i in range(0, len(file_list), batch_size):
                    batch = file_list[i:i+batch_size]
                    entities, rels = self._ai_analyze_files(batch, file_type)
                    all_entities.extend(entities)
                    all_relationships.extend(rels)
                    
                    logger.info(f"✅ Batch {i//batch_size + 1}: {len(entities)} entities, {len(rels)} relationships")
                
            except Exception as e:
                logger.error(f"❌ Failed to AI-process {file_type} files: {e}")
        
        # Use AI to find cross-file relationships
        cross_relationships = self._ai_find_cross_relationships(all_entities)
        all_relationships.extend(cross_relationships)
        
        logger.info(f"🔗 AI found {len(cross_relationships)} cross-file relationships")
        logger.info(f"📊 Total: {len(all_entities)} entities, {len(all_relationships)} relationships")
        
        return all_entities, all_relationships
    
    def _group_files_by_type(self, files: List[Path]) -> Dict[str, List[Path]]:
        """Group files by their type for targeted AI analysis."""
        groups = {
            'code': [],
            'documentation': [],
            'config': [],
            'data': []
        }
        
        code_extensions = {'.py', '.r', '.java', '.cpp', '.c', '.js', '.ts', '.go', '.rs', '.php', '.rb', '.swift', '.kt'}
        doc_extensions = {'.md', '.rst', '.txt'}
        config_extensions = {'.json', '.yaml', '.yml', '.toml', '.cfg', '.ini', '.xml'}
        data_extensions = {'.csv', '.tsv', '.pdf'}
        
        # Add notebook to appropriate category if available
        if NBFORMAT_AVAILABLE:
            doc_extensions.add('.ipynb')
        
        for file_path in files:
            if not file_path.is_file():
                continue
                
            suffix = file_path.suffix.lower()
            
            if suffix in code_extensions:
                groups['code'].append(file_path)
            elif suffix in doc_extensions:
                groups['documentation'].append(file_path)
            elif suffix in config_extensions:
                groups['config'].append(file_path)
            elif suffix in data_extensions:
                groups['data'].append(file_path)
        
        return groups
    
    def _ai_analyze_files(self, files: List[Path], file_type: str) -> Tuple[List[Dict], List[Dict]]:
        """Use AI to analyze files and extract entities/relationships."""
        entities = []
        relationships = []
        
        # Prepare file contents for AI analysis
        file_contents = []
        for file_path in files:
            try:
                if file_path.suffix.lower() == '.ipynb':
                    content = self._extract_notebook_content(file_path)
                elif file_path.suffix.lower() == '.pdf':
                    content = f"[PDF FILE: {file_path.name}]"  # Handle via existing RAG
                    continue
                else:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                
                # Limit content size to avoid token limits
                if len(content) > 8000:
                    content = content[:8000] + "\n... [TRUNCATED]"
                
                file_contents.append({
                    'path': str(file_path),
                    'name': file_path.name,
                    'type': file_path.suffix.lower(),
                    'content': content
                })
                
            except Exception as e:
                logger.warning(f"Failed to read file {file_path}: {e}")
        
        if not file_contents:
            return entities, relationships
        
        # Create AI prompt based on file type
        prompt = self._create_analysis_prompt(file_contents, file_type)
        
        try:
            # Call OpenAI to analyze files
            response = self.client.chat.completions.create(
                model="gpt-4o",
                temperature=0.1,
                max_tokens=4000,
                messages=[
                    {"role": "system", "content": "You are an expert code and document analyzer. Extract entities and relationships in valid JSON format."},
                    {"role": "user", "content": prompt}
                ]
            )
            
            # Parse AI response
            ai_result = response.choices[0].message.content.strip()
            
            # Try to extract JSON from AI response
            try:
                # Find JSON in response
                json_start = ai_result.find('{')
                json_end = ai_result.rfind('}') + 1
                if json_start != -1 and json_end > json_start:
                    json_str = ai_result[json_start:json_end]
                    parsed_result = json.loads(json_str)
                    
                    # Convert AI results to our format
                    entities, relationships = self._convert_ai_results(parsed_result, file_contents)
                    
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse AI JSON response: {e}")
                logger.debug(f"AI response: {ai_result[:500]}...")
                
        except Exception as e:
            # Handle specific OpenAI quota errors gracefully
            if "insufficient_quota" in str(e) or "429" in str(e):
                logger.warning(f"⚠️ OpenAI API quota exceeded - switching to fallback mode")
                logger.info("💡 Suggestion: Check your OpenAI billing at https://platform.openai.com/usage")
                # Return empty results to continue processing with existing data
                return [], []
            else:
                logger.error(f"AI analysis failed: {e}")
        
        return entities, relationships
    
    def _create_analysis_prompt(self, file_contents: List[Dict], file_type: str) -> str:
        """Create AI prompt for analyzing files."""
        
        if file_type == 'code':
            prompt = f"""Analyze the following {len(file_contents)} code files and extract entities and relationships.

For each file, identify:
1. ENTITIES: Classes, functions, modules, imports, variables, constants, data structures
2. RELATIONSHIPS: What imports what, what calls what, inheritance, composition, dependencies

Files to analyze:
"""
            for file_info in file_contents:
                prompt += f"\n=== {file_info['name']} ({file_info['type']}) ===\n"
                prompt += file_info['content'][:2000] + ("..." if len(file_info['content']) > 2000 else "")
                
        elif file_type == 'documentation':
            prompt = f"""Analyze the following {len(file_contents)} documentation files and extract entities and relationships.

For each file, identify:
1. ENTITIES: Concepts, topics, API references, code examples, configuration options
2. RELATIONSHIPS: What documents what, conceptual relationships, references

Files to analyze:
"""
            for file_info in file_contents:
                prompt += f"\n=== {file_info['name']} ===\n"
                prompt += file_info['content'][:3000] + ("..." if len(file_info['content']) > 3000 else "")
                
        elif file_type == 'config':
            prompt = f"""Analyze the following {len(file_contents)} configuration files and extract entities and relationships.

For each file, identify:
1. ENTITIES: Configuration keys, values, sections, environment variables
2. RELATIONSHIPS: Dependencies, includes, references

Files to analyze:
"""
            for file_info in file_contents:
                prompt += f"\n=== {file_info['name']} ===\n"
                prompt += file_info['content'][:1500] + ("..." if len(file_info['content']) > 1500 else "")
        else:
            prompt = f"""Analyze the following {len(file_contents)} files and extract entities and relationships.

Files to analyze:
"""
            for file_info in file_contents:
                prompt += f"\n=== {file_info['name']} ===\n"
                prompt += file_info['content'][:2000] + ("..." if len(file_info['content']) > 2000 else "")
        
        prompt += """

Return your analysis as JSON in this exact format:
{
  "entities": [
    {
      "name": "entity_name",
      "type": "CLASS|FUNCTION|MODULE|CONCEPT|CONFIG_KEY|etc",
      "description": "Brief description",
      "source_file": "filename",
      "properties": {
        "additional": "metadata"
      }
    }
  ],
  "relationships": [
    {
      "source_entity": "entity1_name",
      "target_entity": "entity2_name", 
      "relationship_type": "IMPORTS|CALLS|INHERITS|DOCUMENTS|REFERENCES|etc",
      "description": "Brief description"
    }
  ]
}

Be thorough but focus on the most important entities and relationships. Use consistent naming."""
        
        return prompt
    
    def _extract_notebook_content(self, notebook_path: Path) -> str:
        """Extract content from Jupyter notebook."""
        if not NBFORMAT_AVAILABLE:
            logger.warning(f"Skipping notebook {notebook_path.name} - nbformat not available")
            return f"[JUPYTER NOTEBOOK: {notebook_path.name} - nbformat not available]"
        
        try:
            with open(notebook_path, 'r', encoding='utf-8') as f:
                notebook = nbformat.read(f, as_version=4)
            
            content = f"JUPYTER NOTEBOOK: {notebook_path.name}\n\n"
            
            for i, cell in enumerate(notebook.cells):
                if cell.cell_type == 'markdown':
                    content += f"=== MARKDOWN CELL {i} ===\n{cell.source}\n\n"
                elif cell.cell_type == 'code':
                    content += f"=== CODE CELL {i} ===\n{cell.source}\n\n"
            
            return content
            
        except Exception as e:
            logger.warning(f"Failed to parse notebook {notebook_path}: {e}")
            return f"[Failed to parse notebook: {notebook_path.name}]"
    
    def _convert_ai_results(self, ai_result: Dict, file_contents: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """Convert AI analysis results to our entity/relationship format."""
        entities = []
        relationships = []
        
        # Create entity name to ID mapping
        entity_name_to_id = {}
        
        # Process entities
        for ai_entity in ai_result.get('entities', []):
            entity_id = f'entity_{self.next_entity_id}'
            self.next_entity_id += 1
            
            entity = {
                'id': entity_id,
                'name': ai_entity.get('name', 'Unknown'),
                'type': ai_entity.get('type', 'UNKNOWN'),
                'description': ai_entity.get('description', ''),
                'source_file': ai_entity.get('source_file', ''),
                'file_type': 'ai_analyzed',
                'ai_extracted': True
            }
            
            # Add additional properties
            if 'properties' in ai_entity:
                entity.update(ai_entity['properties'])
            
            entities.append(entity)
            entity_name_to_id[ai_entity.get('name')] = entity_id
        
        # Process relationships
        for ai_rel in ai_result.get('relationships', []):
            source_name = ai_rel.get('source_entity')
            target_name = ai_rel.get('target_entity')
            
            if source_name in entity_name_to_id and target_name in entity_name_to_id:
                relationship = {
                    'source': entity_name_to_id[source_name],
                    'target': entity_name_to_id[target_name],
                    'type': ai_rel.get('relationship_type', 'UNKNOWN'),
                    'description': ai_rel.get('description', ''),
                    'ai_extracted': True
                }
                relationships.append(relationship)
        
        return entities, relationships
    
    def _ai_find_cross_relationships(self, entities: List[Dict]) -> List[Dict]:
        """Use AI to find relationships between entities across different files."""
        if len(entities) < 2:
            return []
        
        logger.info("🤖 Using AI to find cross-file relationships...")
        
        # Prepare entity summary for AI
        entity_summary = []
        for entity in entities[:100]:  # Limit to avoid token overflow
            summary = {
                'name': entity['name'],
                'type': entity['type'],
                'description': entity['description'],
                'source_file': entity.get('source_file', ''),
                'id': entity['id']
            }
            entity_summary.append(summary)
        
        prompt = f"""Analyze these {len(entity_summary)} entities from a codebase and identify cross-file relationships.

Entities:
{json.dumps(entity_summary, indent=2)}

Find relationships like:
- Modules importing other modules
- Functions calling functions from other files  
- Classes inheriting from classes in other files
- Documentation referencing code entities
- Configuration files used by code
- Similar/duplicate entities across files

Return JSON format:
{{
  "cross_relationships": [
    {{
      "source_entity_id": "entity_X",
      "target_entity_id": "entity_Y", 
      "relationship_type": "IMPORTS|CALLS|INHERITS|DOCUMENTS|USES|SIMILAR|etc",
      "description": "Brief description",
      "confidence": 0.8
    }}
  ]
}}

Focus on high-confidence relationships."""
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                temperature=0.1,
                max_tokens=2000,
                messages=[
                    {"role": "system", "content": "You are an expert at finding relationships in codebases. Return valid JSON."},
                    {"role": "user", "content": prompt}
                ]
            )
            
            ai_result = response.choices[0].message.content.strip()
            
            # Parse AI response
            json_start = ai_result.find('{')
            json_end = ai_result.rfind('}') + 1
            if json_start != -1 and json_end > json_start:
                json_str = ai_result[json_start:json_end]
                parsed_result = json.loads(json_str)
                
                relationships = []
                for ai_rel in parsed_result.get('cross_relationships', []):
                    if ai_rel.get('confidence', 0) > 0.6:  # Only high-confidence relationships
                        relationship = {
                            'source': ai_rel['source_entity_id'],
                            'target': ai_rel['target_entity_id'],
                            'type': ai_rel['relationship_type'],
                            'description': ai_rel['description'],
                            'ai_extracted': True,
                            'cross_file': True,
                            'confidence': ai_rel.get('confidence', 0.8)
                        }
                        relationships.append(relationship)
                
                return relationships
                
        except Exception as e:
            logger.warning(f"AI cross-relationship analysis failed: {e}")
        
        return []
    
    def update_graph_with_entities(self, entities: List[Dict], relationships: List[Dict]):
        """Update the existing graph with new entities and relationships."""
        logger.info(f"🔄 Updating graph with {len(entities)} entities and {len(relationships)} relationships...")
        
        # Add entities as nodes
        for entity in entities:
            entity_id = entity['id']
            if not self.graph.has_node(entity_id):
                self.graph.add_node(entity_id, **entity)
                self.entity_registry[entity['name']] = entity_id
        
        # Add relationships as edges
        for rel in relationships:
            source_id = rel['source']
            target_id = rel['target']
            
            if (self.graph.has_node(source_id) and 
                self.graph.has_node(target_id) and
                not self.graph.has_edge(source_id, target_id)):
                self.graph.add_edge(source_id, target_id, **rel)
        
        # Save updated graph
        self._save_graph()
        
        logger.info(f"✅ Graph updated: {self.graph.number_of_nodes()} total nodes, {self.graph.number_of_edges()} total edges")
    
    def _save_graph(self):
        """Save the graph to disk."""
        try:
            # Save graph
            graph_path = self.output_path / 'enhanced_graph.pkl'
            with open(graph_path, 'wb') as f:
                pickle.dump(self.graph, f)
            
            # Save metadata
            metadata = {
                'nodes': self.graph.number_of_nodes(),
                'edges': self.graph.number_of_edges(),
                'created_at': time.time(),
                'node_types': list(set(data.get('type', 'UNKNOWN') for _, data in self.graph.nodes(data=True))),
                'file_types': list(set(data.get('file_type', 'unknown') for _, data in self.graph.nodes(data=True))),
                'ai_enhanced': True
            }
            
            metadata_path = self.output_path / 'metadata.json'
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"💾 Saved AI-enhanced graph: {metadata['nodes']} nodes, {metadata['edges']} edges")
            
        except Exception as e:
            logger.error(f"Failed to save graph: {e}")
    
    def rebuild_full_graph(self, knowledge_base_path: str):
        """Rebuild the entire graph from scratch using AI analysis."""
        logger.info("🤖 AI-rebuilding entire knowledge graph from scratch...")
        
        # Clear existing graph
        self.graph.clear()
        self.entity_registry.clear()
        self.next_entity_id = 1
        
        # Find all files in knowledge base
        kb_path = Path(knowledge_base_path)
        all_files = []
        
        supported_extensions = {
            '.py', '.r', '.java', '.cpp', '.c', '.js', '.ts', '.go', '.rs', '.php', '.rb', '.swift', '.kt',  # Code
            '.md', '.rst', '.txt',  # Documentation  
            '.json', '.yaml', '.yml', '.toml', '.cfg', '.ini', '.xml',  # Config
            '.csv', '.tsv', '.pdf'  # Data
        }
        
        # Add notebook support if available
        if NBFORMAT_AVAILABLE:
            supported_extensions.add('.ipynb')
        
        for file_path in kb_path.rglob("*"):
            if (file_path.is_file() and 
                not file_path.name.startswith('.') and
                file_path.suffix.lower() in supported_extensions):
                all_files.append(file_path)
        
        logger.info(f"🔍 Found {len(all_files)} files to AI-analyze")
        
        # Process all files using AI
        entities, relationships = self.process_files(all_files)
        
        # Update graph
        self.update_graph_with_entities(entities, relationships)
        
        logger.info(f"✅ AI-powered full graph rebuild complete: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")
        
        return self.graph


# Alias for backward compatibility
EnhancedGraphBuilder = AIGraphBuilder