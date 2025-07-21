"""
AI-Powered Comprehensive RAG Builder
Processes all file types using AI for semantic chunking and indexing
"""
import os
import json
import pickle
import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple
import time
import hashlib

import faiss
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv

# Optional nbformat import for notebook processing
try:
    import nbformat
    NBFORMAT_AVAILABLE = True
except ImportError:
    NBFORMAT_AVAILABLE = False

load_dotenv()
logger = logging.getLogger(__name__)


class ComprehensiveRAGBuilder:
    """AI-powered RAG builder for all file types with semantic chunking."""
    
    def __init__(self, output_path: str):
        self.output_path = Path(output_path)
        self.output_path.mkdir(exist_ok=True)
        
        # Initialize OpenAI client
        self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        
        # Load existing components
        self.chunks = self._load_existing_chunks()
        self.faiss_index = self._load_existing_faiss_index()
        self.chunk_registry = self._build_chunk_registry()
        
        # Track processed files to avoid duplicates
        self.processed_files = self._load_processed_files()
    
    def _load_existing_chunks(self) -> List[Dict]:
        """Load existing RAG chunks."""
        chunks_path = self.output_path / 'chunks.pkl'
        if chunks_path.exists():
            try:
                with open(chunks_path, 'rb') as f:
                    chunks = pickle.load(f)
                logger.info(f"📚 Loaded {len(chunks)} existing RAG chunks")
                return chunks
            except Exception as e:
                logger.warning(f"Failed to load existing chunks: {e}")
        return []
    
    def _load_existing_faiss_index(self):
        """Load existing FAISS index."""
        index_path = self.output_path / 'faiss.index'
        if index_path.exists():
            try:
                index = faiss.read_index(str(index_path))
                logger.info(f"📊 Loaded existing FAISS index with {index.ntotal} vectors")
                return index
            except Exception as e:
                logger.warning(f"Failed to load existing FAISS index: {e}")
        
        # Create new index
        return faiss.IndexFlatIP(1536)  # OpenAI embedding dimension
    
    def _build_chunk_registry(self) -> Dict[str, str]:
        """Build registry of chunk content hashes to avoid duplicates."""
        registry = {}
        for chunk in self.chunks:
            content_hash = hashlib.sha256(chunk.get('content', '').encode()).hexdigest()
            registry[content_hash] = chunk.get('chunk_id', '')
        return registry
    
    def _load_processed_files(self) -> Dict[str, str]:
        """Load registry of processed files with their hashes."""
        processed_path = self.output_path / 'processed_files.json'
        if processed_path.exists():
            try:
                with open(processed_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load processed files registry: {e}")
        return {}
    
    def _save_processed_files(self):
        """Save registry of processed files."""
        processed_path = self.output_path / 'processed_files.json'
        try:
            with open(processed_path, 'w') as f:
                json.dump(self.processed_files, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save processed files registry: {e}")
    
    def process_file(self, file_path: Path) -> List[Dict]:
        """Process a single file and return chunks."""
        if not file_path.is_file():
            return []
        
        # Check if file already processed
        file_hash = self._compute_file_hash(file_path)
        file_key = str(file_path)
        
        if file_key in self.processed_files and self.processed_files[file_key] == file_hash:
            logger.debug(f"⏭️ Skipping already processed file: {file_path.name}")
            return []
        
        logger.info(f"🔄 Processing file: {file_path.name}")
        
        try:
            # Determine file type and process accordingly
            chunks = self._process_file_by_type(file_path)
            
            # Update processed files registry
            self.processed_files[file_key] = file_hash
            
            logger.info(f"✅ Processed {file_path.name}: {len(chunks)} chunks")
            return chunks
            
        except Exception as e:
            logger.error(f"❌ Failed to process {file_path}: {e}")
            return []
    
    def _compute_file_hash(self, file_path: Path) -> str:
        """Compute hash for file change detection."""
        stat = file_path.stat()
        hash_input = f"{file_path}:{stat.st_mtime}:{stat.st_size}"
        return hashlib.sha256(hash_input.encode()).hexdigest()
    
    def _process_file_by_type(self, file_path: Path) -> List[Dict]:
        """Process file based on its type using AI."""
        suffix = file_path.suffix.lower()
        
        if suffix in ['.py', '.r', '.java', '.cpp', '.c', '.js', '.ts', '.go', '.rs', '.php', '.rb', '.swift', '.kt']:
            return self._process_code_file(file_path)
        elif suffix in ['.md', '.rst', '.txt']:
            return self._process_text_file(file_path)
        elif suffix == '.ipynb':
            return self._process_notebook_file(file_path)
        elif suffix in ['.json', '.yaml', '.yml', '.toml', '.cfg', '.ini', '.xml']:
            return self._process_config_file(file_path)
        elif suffix in ['.csv', '.tsv']:
            return self._process_data_file(file_path)
        elif suffix == '.pdf':
            return self._process_pdf_file(file_path)
        else:
            return self._process_generic_file(file_path)
    
    def _process_code_file(self, file_path: Path) -> List[Dict]:
        """Process code files using AI for semantic chunking."""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            if not content.strip():
                return []
            
            # Use AI to create semantic chunks
            chunks = self._ai_chunk_code(content, file_path)
            return chunks
            
        except Exception as e:
            logger.warning(f"Failed to process code file {file_path}: {e}")
            return []
    
    def _ai_chunk_code(self, content: str, file_path: Path) -> List[Dict]:
        """Use AI to create semantic chunks from code."""
        # For very large files, split into manageable pieces first
        if len(content) > 12000:
            # Split by logical boundaries (classes, functions)
            logical_chunks = self._split_code_logically(content, file_path.suffix)
        else:
            logical_chunks = [content]
        
        ai_chunks = []
        
        for i, code_chunk in enumerate(logical_chunks):
            try:
                prompt = f"""Analyze this {file_path.suffix} code and create semantic chunks.

Split the code into logical, meaningful chunks such as:
- Class definitions with their methods
- Individual functions with their documentation
- Import/configuration sections
- Main execution blocks

For each chunk, provide:
1. The code content
2. A descriptive title
3. A brief summary of what it does
4. The type (class, function, config, etc.)

Code to analyze:
```{file_path.suffix}
{code_chunk[:8000]}{"..." if len(code_chunk) > 8000 else ""}
```

Return JSON format:
{{
  "chunks": [
    {{
      "content": "actual code content",
      "title": "descriptive title", 
      "summary": "what this code does",
      "type": "class|function|import|config|main",
      "start_line": 1,
      "end_line": 10
    }}
  ]
}}"""

                response = self.client.chat.completions.create(
                    model="gpt-4o",
                    temperature=0.1,
                    max_tokens=3000,
                    messages=[
                        {"role": "system", "content": "You are an expert code analyzer. Create semantic chunks and return valid JSON."},
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
                    
                    # Convert to our chunk format
                    for j, ai_chunk in enumerate(parsed_result.get('chunks', [])):
                        chunk = {
                            'chunk_id': f"{file_path.stem}_{i}_{j}",
                            'content': ai_chunk.get('content', ''),
                            'source': str(file_path),
                            'chunk_type': f"code_{ai_chunk.get('type', 'unknown')}",
                            'title': ai_chunk.get('title', f"{file_path.name} chunk {j}"),
                            'summary': ai_chunk.get('summary', ''),
                            'file_type': file_path.suffix,
                            'language': self._detect_language(file_path.suffix),
                            'ai_processed': True,
                            'start_line': ai_chunk.get('start_line'),
                            'end_line': ai_chunk.get('end_line')
                        }
                        ai_chunks.append(chunk)
                
            except Exception as e:
                # Handle specific OpenAI quota errors gracefully
                if "insufficient_quota" in str(e) or "429" in str(e):
                    logger.warning(f"⚠️ OpenAI API quota exceeded - using simple chunking for {file_path}")
                    logger.info("💡 Suggestion: Check your OpenAI billing at https://platform.openai.com/usage")
                else:
                    logger.warning(f"AI chunking failed for {file_path}, falling back to simple chunking: {e}")
                # Fallback to simple chunking
                ai_chunks.extend(self._simple_chunk_text(code_chunk, file_path, 'code'))
        
        return ai_chunks
    
    def _split_code_logically(self, content: str, file_ext: str) -> List[str]:
        """Split code into logical chunks before AI processing."""
        lines = content.split('\n')
        chunks = []
        current_chunk = []
        
        # Define logical boundaries based on language
        if file_ext == '.py':
            boundaries = ['class ', 'def ', 'import ', 'from ']
        elif file_ext in ['.java', '.cpp', '.c']:
            boundaries = ['class ', 'public class', 'private class', 'public void', 'private void', '#include']
        elif file_ext in ['.js', '.ts']:
            boundaries = ['class ', 'function ', 'const ', 'let ', 'var ', 'import ', 'export ']
        else:
            boundaries = ['function', 'class', 'import', 'include']
        
        for line in lines:
            line_stripped = line.strip()
            
            # Check if this line starts a new logical block
            is_boundary = any(line_stripped.startswith(boundary) for boundary in boundaries)
            
            if is_boundary and current_chunk and len('\n'.join(current_chunk)) > 100:
                # Save current chunk and start new one
                chunks.append('\n'.join(current_chunk))
                current_chunk = [line]
            else:
                current_chunk.append(line)
        
        # Add final chunk
        if current_chunk:
            chunks.append('\n'.join(current_chunk))
        
        return chunks if chunks else [content]
    
    def _detect_language(self, file_ext: str) -> str:
        """Detect programming language from file extension."""
        lang_map = {
            '.py': 'python',
            '.r': 'r',
            '.java': 'java',
            '.cpp': 'cpp',
            '.c': 'c',
            '.js': 'javascript',
            '.ts': 'typescript',
            '.go': 'go',
            '.rs': 'rust',
            '.php': 'php',
            '.rb': 'ruby',
            '.swift': 'swift',
            '.kt': 'kotlin'
        }
        return lang_map.get(file_ext.lower(), 'unknown')
    
    def _process_text_file(self, file_path: Path) -> List[Dict]:
        """Process text/markdown files using AI for semantic chunking."""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            if not content.strip():
                return []
            
            # Use AI for semantic chunking of documentation
            return self._ai_chunk_documentation(content, file_path)
            
        except Exception as e:
            logger.warning(f"Failed to process text file {file_path}: {e}")
            return []
    
    def _ai_chunk_documentation(self, content: str, file_path: Path) -> List[Dict]:
        """Use AI to create semantic chunks from documentation."""
        try:
            prompt = f"""Analyze this documentation and create semantic chunks.

Split the document into logical sections such as:
- Introduction/overview sections
- Feature descriptions
- Code examples with explanations
- Configuration instructions
- API documentation sections

For each chunk, provide:
1. The content
2. A descriptive title
3. A brief summary
4. The section type

Document to analyze:
{content[:10000]}{"..." if len(content) > 10000 else ""}

Return JSON format:
{{
  "chunks": [
    {{
      "content": "actual content",
      "title": "descriptive title",
      "summary": "what this section covers", 
      "type": "overview|feature|example|config|api|reference"
    }}
  ]
}}"""

            response = self.client.chat.completions.create(
                model="gpt-4o",
                temperature=0.1,
                max_tokens=3000,
                messages=[
                    {"role": "system", "content": "You are an expert documentation analyzer. Create semantic chunks and return valid JSON."},
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
                
                chunks = []
                for i, ai_chunk in enumerate(parsed_result.get('chunks', [])):
                    chunk = {
                        'chunk_id': f"{file_path.stem}_doc_{i}",
                        'content': ai_chunk.get('content', ''),
                        'source': str(file_path),
                        'chunk_type': f"doc_{ai_chunk.get('type', 'section')}",
                        'title': ai_chunk.get('title', f"{file_path.name} section {i}"),
                        'summary': ai_chunk.get('summary', ''),
                        'file_type': file_path.suffix,
                        'ai_processed': True
                    }
                    chunks.append(chunk)
                
                return chunks
        
        except Exception as e:
            # Handle specific OpenAI quota errors gracefully
            if "insufficient_quota" in str(e) or "429" in str(e):
                logger.warning(f"⚠️ OpenAI API quota exceeded - using simple chunking for documentation")
                logger.info("💡 Suggestion: Check your OpenAI billing at https://platform.openai.com/usage")
            else:
                logger.warning(f"AI documentation chunking failed: {e}")
            return self._simple_chunk_text(content, file_path, 'documentation')
    
    def _process_notebook_file(self, file_path: Path) -> List[Dict]:
        """Process Jupyter notebook files."""
        if not NBFORMAT_AVAILABLE:
            logger.warning(f"Skipping notebook {file_path.name} - nbformat not available")
            # Return a basic chunk for the notebook file
            chunk = {
                'chunk_id': f"{file_path.stem}_notebook",
                'content': f"[JUPYTER NOTEBOOK: {file_path.name} - nbformat not available for processing]",
                'source': str(file_path),
                'chunk_type': 'notebook_file',
                'title': f"Notebook: {file_path.name}",
                'summary': f"Jupyter notebook {file_path.name} (processing unavailable)",
                'file_type': '.ipynb'
            }
            return [chunk]
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                notebook = nbformat.read(f, as_version=4)
            
            chunks = []
            
            for i, cell in enumerate(notebook.cells):
                if cell.source.strip():
                    chunk = {
                        'chunk_id': f"{file_path.stem}_cell_{i}",
                        'content': cell.source,
                        'source': str(file_path),
                        'chunk_type': f"notebook_{cell.cell_type}",
                        'title': f"{file_path.name} - Cell {i} ({cell.cell_type})",
                        'summary': f"{cell.cell_type.title()} cell from {file_path.name}",
                        'file_type': '.ipynb',
                        'cell_index': i,
                        'cell_type': cell.cell_type
                    }
                    chunks.append(chunk)
            
            return chunks
            
        except Exception as e:
            logger.warning(f"Failed to process notebook {file_path}: {e}")
            return []
    
    def _process_config_file(self, file_path: Path) -> List[Dict]:
        """Process configuration files."""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            chunk = {
                'chunk_id': f"{file_path.stem}_config",
                'content': content,
                'source': str(file_path),
                'chunk_type': 'config_file',
                'title': f"Configuration: {file_path.name}",
                'summary': f"Configuration file {file_path.name}",
                'file_type': file_path.suffix,
                'config_type': file_path.suffix.lower()
            }
            
            return [chunk]
            
        except Exception as e:
            logger.warning(f"Failed to process config file {file_path}: {e}")
            return []
    
    def _process_data_file(self, file_path: Path) -> List[Dict]:
        """Process data files like CSV."""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # For CSV files, just store metadata and sample
            if file_path.suffix.lower() == '.csv':
                lines = content.split('\n')
                preview = '\n'.join(lines[:10]) + ('...' if len(lines) > 10 else '')
                
                chunk = {
                    'chunk_id': f"{file_path.stem}_data",
                    'content': preview,
                    'source': str(file_path),
                    'chunk_type': 'data_file',
                    'title': f"Data: {file_path.name}",
                    'summary': f"Data file with {len(lines)} rows",
                    'file_type': file_path.suffix,
                    'data_type': 'tabular',
                    'row_count': len(lines)
                }
                
                return [chunk]
            
        except Exception as e:
            logger.warning(f"Failed to process data file {file_path}: {e}")
            return []
    
    def _process_pdf_file(self, file_path: Path) -> List[Dict]:
        """Process PDF files (placeholder - use existing PDF processing)."""
        # PDF processing should use existing infrastructure
        # For now, create a placeholder chunk
        chunk = {
            'chunk_id': f"{file_path.stem}_pdf",
            'content': f"[PDF FILE: {file_path.name}]",
            'source': str(file_path),
            'chunk_type': 'pdf_document',
            'title': f"PDF Document: {file_path.name}",
            'summary': f"PDF document {file_path.name}",
            'file_type': '.pdf'
        }
        
        return [chunk]
    
    def _process_generic_file(self, file_path: Path) -> List[Dict]:
        """Process generic files."""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            return self._simple_chunk_text(content, file_path, 'generic')
            
        except Exception as e:
            logger.warning(f"Failed to process generic file {file_path}: {e}")
            return []
    
    def _simple_chunk_text(self, content: str, file_path: Path, chunk_type: str) -> List[Dict]:
        """Simple text chunking fallback."""
        chunks = []
        chunk_size = 1000
        overlap = 100
        
        for i in range(0, len(content), chunk_size - overlap):
            chunk_content = content[i:i + chunk_size]
            if chunk_content.strip():
                chunk = {
                    'chunk_id': f"{file_path.stem}_{chunk_type}_{i//chunk_size}",
                    'content': chunk_content,
                    'source': str(file_path),
                    'chunk_type': chunk_type,
                    'title': f"{file_path.name} - Part {i//chunk_size + 1}",
                    'summary': f"Text chunk from {file_path.name}",
                    'file_type': file_path.suffix,
                    'simple_chunked': True
                }
                chunks.append(chunk)
        
        return chunks
    
    def update_index_with_chunks(self, new_chunks: List[Dict]):
        """Update RAG index with new chunks."""
        if not new_chunks:
            return
        
        logger.info(f"📊 Updating RAG index with {len(new_chunks)} new chunks...")
        
        # Filter out duplicate chunks
        unique_chunks = []
        for chunk in new_chunks:
            content_hash = hashlib.sha256(chunk.get('content', '').encode()).hexdigest()
            if content_hash not in self.chunk_registry:
                unique_chunks.append(chunk)
                self.chunk_registry[content_hash] = chunk.get('chunk_id', '')
        
        if not unique_chunks:
            logger.info("⏭️ No new unique chunks to add")
            return
        
        logger.info(f"🔍 Adding {len(unique_chunks)} unique chunks to index...")
        
        # Generate embeddings for new chunks
        embeddings = self._generate_embeddings([chunk['content'] for chunk in unique_chunks])
        
        if embeddings is not None and len(embeddings) > 0:
            # Add to FAISS index
            embeddings_array = np.array(embeddings, dtype=np.float32)
            faiss.normalize_L2(embeddings_array)
            self.faiss_index.add(embeddings_array)
            
            # Add chunks to chunk list
            self.chunks.extend(unique_chunks)
            
            # Save everything
            self._save_chunks()
            self._save_faiss_index()
            self._save_processed_files()
            
            logger.info(f"✅ Updated RAG index: {len(self.chunks)} total chunks, {self.faiss_index.ntotal} vectors")
    
    def _generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for texts using OpenAI."""
        if not texts:
            return []
        
        try:
            # Process in batches to avoid rate limits
            batch_size = 100
            all_embeddings = []
            
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                
                response = self.client.embeddings.create(
                    model="text-embedding-ada-002",
                    input=batch
                )
                
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)
                
                logger.debug(f"Generated embeddings for batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}")
            
            return all_embeddings
            
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            return []
    
    def _save_chunks(self):
        """Save chunks to disk."""
        chunks_path = self.output_path / 'chunks.pkl'
        try:
            with open(chunks_path, 'wb') as f:
                pickle.dump(self.chunks, f)
        except Exception as e:
            logger.error(f"Failed to save chunks: {e}")
    
    def _save_faiss_index(self):
        """Save FAISS index to disk."""
        index_path = self.output_path / 'faiss.index'
        try:
            faiss.write_index(self.faiss_index, str(index_path))
        except Exception as e:
            logger.error(f"Failed to save FAISS index: {e}")
    
    def rebuild_full_index(self, knowledge_base_path: str):
        """Rebuild the entire RAG index from scratch."""
        logger.info("🔨 AI-rebuilding entire RAG index from scratch...")
        
        # Clear existing data
        self.chunks.clear()
        self.chunk_registry.clear()
        self.processed_files.clear()
        self.faiss_index = faiss.IndexFlatIP(1536)
        
        # Find all files
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
        
        logger.info(f"🔍 Found {len(all_files)} files to process")
        
        # Process all files
        all_chunks = []
        for file_path in all_files:
            chunks = self.process_file(file_path)
            all_chunks.extend(chunks)
        
        # Update index with all chunks
        self.update_index_with_chunks(all_chunks)
        
        logger.info(f"✅ Full RAG rebuild complete: {len(self.chunks)} chunks, {self.faiss_index.ntotal} vectors")
        
        return self.chunks, self.faiss_index