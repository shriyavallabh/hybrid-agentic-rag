"""
Knowledge Graph Builder
Converts model documents (PDFs, notebooks, code) into a searchable knowledge graph.
"""
import os
import json
import hashlib
import pickle
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime
import subprocess
import sys

import networkx as nx
import faiss
import numpy as np
from filelock import FileLock
import tiktoken
from openai import OpenAI
import pdfplumber
# import camelot  # Removed to avoid dependency issues
# from pdf2image import convert_from_path  # Removed to avoid dependency issues
# import pytesseract  # Removed to avoid dependency issues
# from paddleocr import PaddleOCR  # Commented out to bypass dependency
# import orjson  # Bypassed - use standard json

logger = logging.getLogger(__name__)


class GraphBuilder:
    def __init__(self, knowledge_base_path: str, output_path: str, openai_api_key: str):
        self.knowledge_base = Path(knowledge_base_path)
        self.output_path = Path(output_path)
        self.output_path.mkdir(exist_ok=True)
        
        self.client = OpenAI(api_key=openai_api_key)
        self.tokenizer = tiktoken.encoding_for_model("gpt-4o")
        self.embedding_dim = 1536
        
        # Check system dependencies (bypassed for now)
        # self._check_system_dependencies()
        
        # Initialize OCR (bypassed for now)
        self.paddle_ocr = None  # PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
        
    def _check_system_dependencies(self):
        """Check for required system dependencies."""
        deps = {
            'ghostscript': ['gs', '--version'],
            'poppler': ['pdfinfo', '-v'],
            'tesseract': ['tesseract', '--version']
        }
        
        missing = []
        for name, cmd in deps.items():
            try:
                subprocess.run(cmd, capture_output=True, check=True)
            except (subprocess.CalledProcessError, FileNotFoundError):
                missing.append(name)
        
        if missing:
            msg = f"Missing system dependencies: {', '.join(missing)}. Please install them first."
            logger.error(msg)
            raise RuntimeError(msg)
    
    def _compute_file_hash(self, filepath: Path) -> str:
        """Compute hash of file path + mtime + size."""
        stat = filepath.stat()
        content = f"{filepath}:{stat.st_mtime}:{stat.st_size}"
        return hashlib.sha256(content.encode()).hexdigest()
    
    def _load_manifest(self) -> Dict[str, Any]:
        """Load or create processing manifest."""
        manifest_path = self.output_path / "meta.json"
        if manifest_path.exists():
            with open(manifest_path, 'rb') as f:
                return json.loads(f.read())
        return {
            "version": "1.0.0",
            "created_at": datetime.now().isoformat(),
            "processed_files": {},
            "node_count": 0,
            "edge_count": 0
        }
    
    def _save_manifest(self, manifest: Dict[str, Any]):
        """Save processing manifest."""
        manifest_path = self.output_path / "meta.json"
        with open(manifest_path, 'wb') as f:
            json.dump(manifest, f, indent=2)
    
    def _extract_pdf_text(self, filepath: Path) -> List[Dict[str, Any]]:
        """Extract comprehensive content from PDF including text, tables, and metadata."""
        chunks = []
        
        try:
            logger.info(f"Extracting PDF content from {filepath}")
            
            # Extract comprehensive content with pdfplumber
            with pdfplumber.open(filepath) as pdf:
                for i, page in enumerate(pdf.pages):
                    # Extract main text content
                    text = page.extract_text()
                    if text and text.strip():
                        chunks.append({
                            "type": "text",
                            "content": text,
                            "page": i + 1,
                            "source": str(filepath)
                        })
                        logger.debug(f"Extracted {len(text)} characters from page {i+1}")
                    
                    # Extract tables using pdfplumber's table detection
                    try:
                        tables = page.extract_tables()
                        if tables:
                            for j, table in enumerate(tables):
                                if table and len(table) > 1:  # Skip empty tables
                                    # Convert table to readable text
                                    table_text = "\n".join([
                                        "\t".join([str(cell) if cell else "" for cell in row]) 
                                        for row in table
                                    ])
                                    chunks.append({
                                        "type": "table",
                                        "content": f"Table {j+1} on page {i+1}:\n{table_text}",
                                        "page": i + 1,
                                        "source": str(filepath)
                                    })
                                    logger.debug(f"Extracted table {j+1} from page {i+1}")
                    except Exception as table_error:
                        logger.warning(f"Table extraction failed on page {i+1}: {table_error}")
        
        except Exception as e:
            logger.error(f"PDF extraction failed for {filepath}: {e}")
        
        logger.info(f"Extracted {len(chunks)} chunks from {filepath}")
        return chunks
    
    def _extract_notebook_content(self, filepath: Path) -> List[Dict[str, Any]]:
        """Extract content from Jupyter notebook."""
        chunks = []
        
        try:
            with open(filepath, 'r') as f:
                notebook = json.load(f)
            
            for i, cell in enumerate(notebook.get('cells', [])):
                content = ""
                if cell['cell_type'] == 'markdown':
                    content = ''.join(cell.get('source', []))
                elif cell['cell_type'] == 'code':
                    # Limit code cells to 200 chars
                    source = ''.join(cell.get('source', []))
                    content = source[:200] + "..." if len(source) > 200 else source
                
                if content.strip():
                    chunks.append({
                        "type": f"notebook_{cell['cell_type']}",
                        "content": content,
                        "page": i + 1,
                        "source": str(filepath)
                    })
        
        except Exception as e:
            logger.error(f"Notebook extraction failed for {filepath}: {e}")
        
        return chunks
    
    def _extract_text_file(self, filepath: Path) -> List[Dict[str, Any]]:
        """Extract content from text files (.py, .md, .txt, etc)."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            return [{
                "type": "code" if filepath.suffix == '.py' else "text",
                "content": content,
                "page": 1,
                "source": str(filepath)
            }]
        except Exception as e:
            logger.error(f"Text extraction failed for {filepath}: {e}")
            return []
    
    def _summarize_chunk(self, chunk: Dict[str, Any]) -> str:
        """Summarize chunk to fit within token limit."""
        content = chunk['content']
        tokens = self.tokenizer.encode(content)
        
        if len(tokens) <= 512:
            return content
        
        # Summarize long chunks
        prompt = f"""Summarize this {chunk['type']} content in 150 tokens or less, preserving key model/dataset/metric information:

{content[:3000]}...

Summary:"""
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                temperature=0.0,
                max_tokens=200,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Summarization failed: {e}")
            # Fallback to truncation
            return content[:1000] + "..."
    
    def _extract_graph_elements(self, chunk: Dict[str, Any], namespace: str) -> Tuple[List[Dict], List[Dict]]:
        """Extract nodes and edges from chunk using GPT-4o."""
        summary = self._summarize_chunk(chunk)
        
        prompt = f"""Extract knowledge graph nodes and edges from this {chunk['type']} content.

Content from {namespace} model:
{summary}

Return ONLY valid JSON with this structure:
{{
  "nodes": [
    {{
      "id": "unique-uuid",
      "type": "Model|Dataset|Metric|CodeEntity",
      "label": "name",
      "details": "brief description",
      "page": {chunk.get('page', 1)}
    }}
  ],
  "edges": [
    {{
      "source": "source-node-id",
      "target": "target-node-id",
      "type": "USES_DATASET|HAS_METRIC|CALLS|DERIVED_FROM|DEPENDS_ON|COMPARES_TO",
      "confidence": 0.9,
      "source_page": {chunk.get('page', 1)}
    }}
  ]
}}

Focus on banking/financial model entities. Extract only high-confidence relationships."""
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                temperature=0.0,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            
            # Add namespace to node IDs
            nodes = result.get('nodes', [])
            edges = result.get('edges', [])
            
            for node in nodes:
                node['namespace'] = namespace
                node['source_file'] = chunk['source']
            
            return nodes, edges
            
        except Exception as e:
            logger.error(f"Graph extraction failed: {e}")
            return [], []
    
    def _embed_nodes(self, nodes: List[Dict]) -> np.ndarray:
        """Generate embeddings for nodes."""
        if not nodes:
            return np.array([])
        
        texts = []
        for node in nodes:
            text = f"{node['type']}: {node['label']} - {node['details']}"
            texts.append(text)
        
        embeddings = []
        
        # Batch embed for efficiency
        batch_size = 100
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            try:
                response = self.client.embeddings.create(
                    model="text-embedding-3-small",
                    input=batch
                )
                batch_embeddings = [e.embedding for e in response.data]
                embeddings.extend(batch_embeddings)
            except Exception as e:
                logger.error(f"Embedding generation failed: {e}")
                # Add zero embeddings as fallback
                embeddings.extend([np.zeros(self.embedding_dim).tolist()] * len(batch))
        
        return np.array(embeddings, dtype=np.float32)
    
    def build(self, force_rebuild: bool = False) -> Dict[str, Any]:
        """Build or update the knowledge graph."""
        manifest = self._load_manifest()
        
        # Check which files need processing
        files_to_process = []
        
        for model_dir in self.knowledge_base.iterdir():
            if not model_dir.is_dir():
                continue
            
            namespace = model_dir.name
            
            for filepath in model_dir.rglob("*"):
                if not filepath.is_file():
                    continue
                
                # Check supported extensions
                if filepath.suffix.lower() not in ['.pdf', '.ipynb', '.py', '.md', '.txt', '.r']:
                    continue
                
                file_hash = self._compute_file_hash(filepath)
                rel_path = str(filepath.relative_to(self.knowledge_base))
                
                if force_rebuild or rel_path not in manifest['processed_files'] or \
                   manifest['processed_files'][rel_path] != file_hash:
                    files_to_process.append((filepath, namespace, file_hash, rel_path))
        
        if not files_to_process and not force_rebuild:
            logger.info("No new or changed files to process")
            return manifest
        
        logger.info(f"Processing {len(files_to_process)} files...")
        
        # Load existing graph or create new
        graph_path = self.output_path / "graph.pkl"
        if graph_path.exists() and not force_rebuild:
            with open(graph_path, 'rb') as f:
                graph = pickle.load(f)
        else:
            graph = nx.MultiDiGraph()
        
        all_nodes = []
        
        # Process files
        for filepath, namespace, file_hash, rel_path in files_to_process:
            logger.info(f"Processing {rel_path}...")
            
            # Extract content based on file type
            chunks = []
            if filepath.suffix.lower() == '.pdf':
                chunks = self._extract_pdf_text(filepath)
            elif filepath.suffix.lower() == '.ipynb':
                chunks = self._extract_notebook_content(filepath)
            else:
                chunks = self._extract_text_file(filepath)
            
            # Extract graph elements from chunks
            for chunk in chunks:
                nodes, edges = self._extract_graph_elements(chunk, namespace)
                
                # Add nodes to graph
                for node in nodes:
                    node_id = f"{namespace}_{node['id']}"
                    graph.add_node(node_id, **node)
                    all_nodes.append(node)
                
                # Add edges to graph
                for edge in edges:
                    source_id = f"{namespace}_{edge['source']}"
                    target_id = f"{namespace}_{edge['target']}"
                    
                    if source_id in graph and target_id in graph:
                        graph.add_edge(
                            source_id,
                            target_id,
                            type=edge['type'],
                            confidence=edge['confidence'],
                            source_page=edge.get('source_page')
                        )
            
            # Update manifest
            manifest['processed_files'][rel_path] = file_hash
        
        # Update counts
        manifest['node_count'] = graph.number_of_nodes()
        manifest['edge_count'] = graph.number_of_edges()
        manifest['updated_at'] = datetime.now().isoformat()
        
        # Generate embeddings for all nodes
        logger.info("Generating embeddings...")
        node_list = list(graph.nodes(data=True))
        node_data = [data for _, data in node_list]
        embeddings = self._embed_nodes(node_data)
        
        # Build FAISS index
        if embeddings.shape[0] > 0:
            index = faiss.IndexFlatIP(self.embedding_dim)
            faiss.normalize_L2(embeddings)
            index.add(embeddings)
        else:
            index = faiss.IndexFlatIP(self.embedding_dim)
        
        # Save everything atomically with file lock
        lock_path = self.output_path / ".lock"
        with FileLock(lock_path):
            # Save graph
            with open(graph_path, 'wb') as f:
                pickle.dump(graph, f)
            
            # Save FAISS index
            faiss.write_index(index, str(self.output_path / "faiss.index"))
            
            # Save node mapping
            node_mapping = {i: node_id for i, (node_id, _) in enumerate(node_list)}
            with open(self.output_path / "node_mapping.pkl", 'wb') as f:
                pickle.dump(node_mapping, f)
            
            # Save manifest
            self._save_manifest(manifest)
        
        logger.info(f"Knowledge graph built: {manifest['node_count']} nodes, {manifest['edge_count']} edges")
        return manifest


if __name__ == "__main__":
    # Test build
    import dotenv
    dotenv.load_dotenv()
    
    builder = GraphBuilder(
        knowledge_base_path="knowledge_base",
        output_path="kg_bundle",
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )
    
    builder.build()