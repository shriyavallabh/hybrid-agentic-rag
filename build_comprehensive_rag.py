"""
Comprehensive RAG System Builder
Creates a complete understanding system that mirrors Claude's capabilities
"""
import os
import pickle
import json
from pathlib import Path
from typing import Dict, List, Any, Tuple
from dotenv import load_dotenv
from openai import OpenAI
import faiss
import numpy as np
import hashlib
import logging

# Text processing
import pdfplumber
import ast
import tokenize
import io

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ComprehensiveRAGBuilder:
    def __init__(self, knowledge_base_path: str, output_path: str, openai_api_key: str):
        self.kb_path = Path(knowledge_base_path)
        self.output_path = Path(output_path)
        self.output_path.mkdir(exist_ok=True)
        
        self.client = OpenAI(api_key=openai_api_key)
        self.chunks = []
        self.embeddings = []
        
    def build_comprehensive_index(self):
        """Build comprehensive RAG index from all knowledge base content"""
        logger.info("Building comprehensive RAG index...")
        
        # Process all files
        for root, dirs, files in os.walk(self.kb_path):
            for file in files:
                filepath = Path(root) / file
                try:
                    if file.endswith('.pdf'):
                        self._process_pdf(filepath)
                    elif file.endswith('.py'):
                        self._process_python_file(filepath)
                    elif file.endswith('.ipynb'):
                        self._process_notebook(filepath)
                    elif file.endswith(('.md', '.txt', '.rst')):
                        self._process_text_file(filepath)
                    elif file.endswith(('.yaml', '.yml', '.json')):
                        self._process_config_file(filepath)
                except Exception as e:
                    logger.warning(f"Failed to process {filepath}: {e}")
        
        logger.info(f"Created {len(self.chunks)} semantic chunks")
        
        # Create embeddings
        self._create_embeddings()
        
        # Build FAISS index
        self._build_faiss_index()
        
        # Save everything
        self._save_index()
        
        logger.info("Comprehensive RAG index built successfully!")
        
    def _process_pdf(self, filepath: Path):
        """Extract comprehensive content from PDF"""
        logger.info(f"Processing PDF: {filepath}")
        
        try:
            with pdfplumber.open(filepath) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    # Extract main text
                    text = page.extract_text()
                    if text and text.strip():
                        # Break into semantic paragraphs
                        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
                        for para_num, paragraph in enumerate(paragraphs, 1):
                            if len(paragraph) > 50:  # Skip very short paragraphs
                                self.chunks.append({
                                    'content': paragraph,
                                    'source': str(filepath),
                                    'type': 'pdf_text',
                                    'page': page_num,
                                    'paragraph': para_num,
                                    'metadata': f"PDF page {page_num}, paragraph {para_num}"
                                })
                    
                    # Extract tables
                    try:
                        tables = page.extract_tables()
                        if tables:
                            for table_num, table in enumerate(tables, 1):
                                if table and len(table) > 1:
                                    table_text = "\n".join([
                                        "\t".join([str(cell) if cell else "" for cell in row])
                                        for row in table
                                    ])
                                    self.chunks.append({
                                        'content': f"Table {table_num}:\n{table_text}",
                                        'source': str(filepath),
                                        'type': 'pdf_table',
                                        'page': page_num,
                                        'table': table_num,
                                        'metadata': f"PDF page {page_num}, table {table_num}"
                                    })
                    except Exception as e:
                        logger.debug(f"Table extraction failed on page {page_num}: {e}")
                        
        except Exception as e:
            logger.error(f"PDF processing failed: {e}")
    
    def _process_python_file(self, filepath: Path):
        """Extract comprehensive content from Python files"""
        logger.info(f"Processing Python: {filepath}")
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse AST for structured extraction
            try:
                tree = ast.parse(content)
                
                # Extract module docstring
                docstring = ast.get_docstring(tree)
                if docstring:
                    self.chunks.append({
                        'content': f"Module docstring:\n{docstring}",
                        'source': str(filepath),
                        'type': 'python_module_doc',
                        'metadata': f"Python module documentation"
                    })
                
                # Extract classes and functions
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        func_info = self._extract_function_info(node, filepath)
                        if func_info:
                            self.chunks.append(func_info)
                    
                    elif isinstance(node, ast.ClassDef):
                        class_info = self._extract_class_info(node, filepath)
                        if class_info:
                            self.chunks.append(class_info)
                            
            except SyntaxError:
                logger.warning(f"Syntax error in {filepath}, processing as raw text")
                
            # Also chunk the raw file for completeness
            lines = content.split('\n')
            current_chunk = []
            chunk_num = 1
            
            for line_num, line in enumerate(lines, 1):
                current_chunk.append(line)
                
                # Create chunks every ~20 lines or at major breaks
                if len(current_chunk) >= 20 or line.strip().startswith(('class ', 'def ', 'import ', 'from ')):
                    chunk_content = '\n'.join(current_chunk).strip()
                    if chunk_content:
                        self.chunks.append({
                            'content': chunk_content,
                            'source': str(filepath),
                            'type': 'python_code',
                            'lines': f"{line_num - len(current_chunk) + 1}-{line_num}",
                            'chunk': chunk_num,
                            'metadata': f"Python code lines {line_num - len(current_chunk) + 1}-{line_num}"
                        })
                    current_chunk = []
                    chunk_num += 1
            
            # Add remaining lines
            if current_chunk:
                chunk_content = '\n'.join(current_chunk).strip()
                if chunk_content:
                    self.chunks.append({
                        'content': chunk_content,
                        'source': str(filepath),
                        'type': 'python_code',
                        'lines': f"{len(lines) - len(current_chunk) + 1}-{len(lines)}",
                        'chunk': chunk_num,
                        'metadata': f"Python code lines {len(lines) - len(current_chunk) + 1}-{len(lines)}"
                    })
                    
        except Exception as e:
            logger.error(f"Python file processing failed: {e}")
    
    def _extract_function_info(self, node: ast.FunctionDef, filepath: Path) -> Dict:
        """Extract detailed function information"""
        try:
            # Get function signature
            args = [arg.arg for arg in node.args.args]
            signature = f"def {node.name}({', '.join(args)})"
            
            # Get docstring
            docstring = ast.get_docstring(node) or "No docstring"
            
            # Create comprehensive function description
            func_content = f"""Function: {node.name}
Signature: {signature}
Location: {filepath.name}:{node.lineno}
Docstring: {docstring}
"""
            
            return {
                'content': func_content,
                'source': str(filepath),
                'type': 'python_function',
                'function_name': node.name,
                'line_number': node.lineno,
                'metadata': f"Function {node.name} at line {node.lineno}"
            }
        except Exception as e:
            logger.debug(f"Function extraction failed: {e}")
            return None
    
    def _extract_class_info(self, node: ast.ClassDef, filepath: Path) -> Dict:
        """Extract detailed class information"""
        try:
            # Get base classes
            bases = [base.id if isinstance(base, ast.Name) else str(base) for base in node.bases]
            base_str = f"({', '.join(bases)})" if bases else ""
            
            # Get docstring
            docstring = ast.get_docstring(node) or "No docstring"
            
            # Get methods
            methods = [n.name for n in node.body if isinstance(n, ast.FunctionDef)]
            
            # Create comprehensive class description
            class_content = f"""Class: {node.name}{base_str}
Location: {filepath.name}:{node.lineno}
Docstring: {docstring}
Methods: {', '.join(methods) if methods else 'None'}
"""
            
            return {
                'content': class_content,
                'source': str(filepath),
                'type': 'python_class',
                'class_name': node.name,
                'line_number': node.lineno,
                'methods': methods,
                'metadata': f"Class {node.name} at line {node.lineno}"
            }
        except Exception as e:
            logger.debug(f"Class extraction failed: {e}")
            return None
    
    def _process_notebook(self, filepath: Path):
        """Extract content from Jupyter notebooks"""
        logger.info(f"Processing notebook: {filepath}")
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                notebook = json.load(f)
            
            for cell_num, cell in enumerate(notebook.get('cells', []), 1):
                cell_type = cell.get('cell_type', 'unknown')
                source = ''.join(cell.get('source', []))
                
                if source.strip():
                    self.chunks.append({
                        'content': source,
                        'source': str(filepath),
                        'type': f'notebook_{cell_type}',
                        'cell_number': cell_num,
                        'metadata': f"Notebook cell {cell_num} ({cell_type})"
                    })
                
                # Extract outputs if present
                outputs = cell.get('outputs', [])
                for output_num, output in enumerate(outputs, 1):
                    output_text = ""
                    if 'text' in output:
                        output_text = ''.join(output['text'])
                    elif 'data' in output and 'text/plain' in output['data']:
                        output_text = ''.join(output['data']['text/plain'])
                    
                    if output_text.strip():
                        self.chunks.append({
                            'content': f"Output:\n{output_text}",
                            'source': str(filepath),
                            'type': 'notebook_output',
                            'cell_number': cell_num,
                            'output_number': output_num,
                            'metadata': f"Notebook cell {cell_num} output {output_num}"
                        })
                        
        except Exception as e:
            logger.error(f"Notebook processing failed: {e}")
    
    def _process_text_file(self, filepath: Path):
        """Extract content from text/markdown files"""
        logger.info(f"Processing text file: {filepath}")
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Split into sections (by headers for markdown)
            if filepath.suffix.lower() == '.md':
                sections = self._split_markdown_sections(content)
                for section_num, section in enumerate(sections, 1):
                    if section.strip():
                        self.chunks.append({
                            'content': section,
                            'source': str(filepath),
                            'type': 'markdown_section',
                            'section': section_num,
                            'metadata': f"Markdown section {section_num}"
                        })
            else:
                # Split into paragraphs
                paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
                for para_num, paragraph in enumerate(paragraphs, 1):
                    if len(paragraph) > 50:
                        self.chunks.append({
                            'content': paragraph,
                            'source': str(filepath),
                            'type': 'text_paragraph',
                            'paragraph': para_num,
                            'metadata': f"Text paragraph {para_num}"
                        })
                        
        except Exception as e:
            logger.error(f"Text file processing failed: {e}")
    
    def _split_markdown_sections(self, content: str) -> List[str]:
        """Split markdown into logical sections"""
        lines = content.split('\n')
        sections = []
        current_section = []
        
        for line in lines:
            if line.strip().startswith('#'):  # Header
                if current_section:
                    sections.append('\n'.join(current_section))
                current_section = [line]
            else:
                current_section.append(line)
        
        if current_section:
            sections.append('\n'.join(current_section))
        
        return sections
    
    def _process_config_file(self, filepath: Path):
        """Extract content from configuration files"""
        logger.info(f"Processing config: {filepath}")
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            self.chunks.append({
                'content': content,
                'source': str(filepath),
                'type': 'config_file',
                'metadata': f"Configuration file {filepath.name}"
            })
            
        except Exception as e:
            logger.error(f"Config file processing failed: {e}")
    
    def _create_embeddings(self):
        """Create embeddings for all chunks"""
        logger.info("Creating embeddings for all chunks...")
        
        # Create embeddings in batches
        batch_size = 100
        all_embeddings = []
        
        for i in range(0, len(self.chunks), batch_size):
            batch = self.chunks[i:i + batch_size]
            texts = [chunk['content'] for chunk in batch]
            
            response = self.client.embeddings.create(
                model="text-embedding-ada-002",
                input=texts
            )
            
            batch_embeddings = [emb.embedding for emb in response.data]
            all_embeddings.extend(batch_embeddings)
            
            logger.info(f"Created embeddings for batch {i//batch_size + 1}/{(len(self.chunks) + batch_size - 1)//batch_size}")
        
        self.embeddings = np.array(all_embeddings, dtype=np.float32)
        logger.info(f"Created {len(self.embeddings)} embeddings")
    
    def _build_faiss_index(self):
        """Build FAISS index for similarity search"""
        logger.info("Building FAISS index...")
        
        # Normalize embeddings
        faiss.normalize_L2(self.embeddings)
        
        # Create index
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(self.embeddings)
        
        logger.info(f"FAISS index built with {self.index.ntotal} vectors")
    
    def _save_index(self):
        """Save the complete RAG index"""
        logger.info("Saving comprehensive RAG index...")
        
        # Save FAISS index
        faiss.write_index(self.index, str(self.output_path / 'faiss.index'))
        
        # Save chunks
        with open(self.output_path / 'chunks.pkl', 'wb') as f:
            pickle.dump(self.chunks, f)
        
        # Save embeddings
        with open(self.output_path / 'embeddings.pkl', 'wb') as f:
            pickle.dump(self.embeddings, f)
        
        # Save metadata
        metadata = {
            'total_chunks': len(self.chunks),
            'embedding_dimension': self.embeddings.shape[1],
            'chunk_types': {},
            'source_files': set()
        }
        
        for chunk in self.chunks:
            chunk_type = chunk.get('type', 'unknown')
            metadata['chunk_types'][chunk_type] = metadata['chunk_types'].get(chunk_type, 0) + 1
            metadata['source_files'].add(chunk['source'])
        
        metadata['source_files'] = list(metadata['source_files'])
        
        with open(self.output_path / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info("RAG index saved successfully!")
        return metadata

if __name__ == "__main__":
    builder = ComprehensiveRAGBuilder(
        knowledge_base_path='knowledge_base',
        output_path='rag_index',
        openai_api_key=os.getenv('OPENAI_API_KEY')
    )
    
    builder.build_comprehensive_index()