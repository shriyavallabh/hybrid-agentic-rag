#!/usr/bin/env python3
"""
Auto-Refresh Subsystem for Hybrid RAG System
Monitors knowledge_base directory for changes and incrementally updates embeddings and graph
"""
import os
import json
import pickle
import hashlib
import zipfile
import shutil
import threading
import time
import logging
from pathlib import Path
from typing import Dict, List, Set, Optional, Any
from datetime import datetime

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import networkx as nx
import faiss
import numpy as np

logger = logging.getLogger(__name__)

class FileChangeTracker:
    """Track file changes using SHA256 hashing for incremental processing."""
    
    def __init__(self, index_path: str):
        self.index_path = Path(index_path)
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self.file_index = self._load_index()
    
    def _load_index(self) -> Dict[str, Dict]:
        """Load existing file index or create new one."""
        if self.index_path.exists():
            try:
                with open(self.index_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load file index: {e}")
        return {}
    
    def _save_index(self):
        """Save file index to disk."""
        try:
            with open(self.index_path, 'w') as f:
                json.dump(self.file_index, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save file index: {e}")
    
    def _compute_file_hash(self, file_path: Path) -> str:
        """Compute hash from path, mtime, and size."""
        stat = file_path.stat()
        hash_input = f"{file_path}:{stat.st_mtime}:{stat.st_size}"
        return hashlib.sha256(hash_input.encode()).hexdigest()
    
    def get_changed_files(self, directory: Path) -> Dict[str, List[Path]]:
        """Get files that have been added, modified, or deleted."""
        changes = {
            'added': [],
            'modified': [],
            'deleted': []
        }
        
        # Current files
        current_files = {}
        for file_path in directory.rglob("*"):
            if file_path.is_file() and not file_path.name.startswith('.'):
                rel_path = str(file_path.relative_to(directory))
                current_files[rel_path] = {
                    'path': file_path,
                    'hash': self._compute_file_hash(file_path)
                }
        
        # Compare with previous state
        for rel_path, file_info in current_files.items():
            if rel_path not in self.file_index:
                changes['added'].append(file_info['path'])
            elif self.file_index[rel_path]['hash'] != file_info['hash']:
                changes['modified'].append(file_info['path'])
        
        # Find deleted files
        for rel_path in self.file_index:
            if rel_path not in current_files:
                changes['deleted'].append(rel_path)
        
        # Update index
        self.file_index = {k: {'hash': v['hash']} for k, v in current_files.items()}
        self._save_index()
        
        return changes

class DebouncedHandler(FileSystemEventHandler):
    """File system event handler with debouncing to batch changes."""
    
    def __init__(self, refresh_flag: threading.Event, debounce_seconds: int = 5):
        self.refresh_flag = refresh_flag
        self.debounce_seconds = debounce_seconds
        self._timer: Optional[threading.Timer] = None
        self._lock = threading.Lock()
        
    def on_any_event(self, event):
        """Handle any file system event."""
        if event.is_directory:
            return
            
        # Ignore hidden files and temp files
        if any(part.startswith('.') for part in Path(event.src_path).parts):
            return
            
        with self._lock:
            if self._timer:
                self._timer.cancel()
            
            self._timer = threading.Timer(
                self.debounce_seconds, 
                self._trigger_refresh
            )
            self._timer.start()
            
        logger.info(f"📁 File change detected: {event.event_type} - {event.src_path}")
    
    def _trigger_refresh(self):
        """Trigger refresh after debounce period."""
        logger.info("🔄 Triggering auto-refresh after debounce period")
        self.refresh_flag.set()

class ZipExtractor:
    """Handle zip file extraction with safety checks."""
    
    MAX_ZIP_SIZE = 500 * 1024 * 1024  # 500MB
    MAX_FILES = 10000
    
    @staticmethod
    def extract_zip(zip_path: Path, extract_to: Path) -> bool:
        """Safely extract zip file with size and file count limits."""
        try:
            # Check zip size
            if zip_path.stat().st_size > ZipExtractor.MAX_ZIP_SIZE:
                logger.error(f"❌ Zip file too large: {zip_path} ({zip_path.stat().st_size} bytes)")
                return False
            
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                # Check file count
                if len(zip_ref.filelist) > ZipExtractor.MAX_FILES:
                    logger.error(f"❌ Zip contains too many files: {len(zip_ref.filelist)}")
                    return False
                
                # Check for zip bombs (compressed vs uncompressed size ratio)
                total_compressed = sum(info.compress_size for info in zip_ref.filelist)
                total_uncompressed = sum(info.file_size for info in zip_ref.filelist)
                
                if total_uncompressed > 0 and (total_uncompressed / total_compressed) > 100:
                    logger.error(f"❌ Potential zip bomb detected (ratio: {total_uncompressed / total_compressed:.1f})")
                    return False
                
                # Create extraction directory
                extract_to.mkdir(parents=True, exist_ok=True)
                
                # Extract safely
                zip_ref.extractall(extract_to)
                logger.info(f"✅ Successfully extracted {len(zip_ref.filelist)} files from {zip_path.name}")
                return True
                
        except Exception as e:
            logger.error(f"❌ Failed to extract {zip_path}: {e}")
            return False

class IncrementalBuilder:
    """Incremental knowledge graph and RAG builder."""
    
    def __init__(self, knowledge_base_path: str, output_path: str):
        self.kb_path = Path(knowledge_base_path)
        self.output_path = Path(output_path)
        self.tracker = FileChangeTracker(self.output_path / "file_index.json")
        
        # Import heavy dependencies only when needed
        self._hybrid_runner = None
        self._graph_builder = None
        
    def _get_hybrid_runner(self):
        """Lazy import of hybrid runner to avoid circular imports."""
        if self._hybrid_runner is None:
            from .hybrid_agent_runner import HybridAgentRunner
            openai_api_key = os.getenv('OPENAI_API_KEY')
            self._hybrid_runner = HybridAgentRunner(
                enhanced_kg_path='enhanced_kg',
                rag_path='rag_index', 
                openai_api_key=openai_api_key
            )
        return self._hybrid_runner
    
    def _get_graph_builder(self):
        """Lazy import of graph builder."""
        if self._graph_builder is None:
            from .simple_graph_builder import SimpleGraphBuilder
            self._graph_builder = SimpleGraphBuilder('enhanced_kg')
        return self._graph_builder
    
    def process_changes(self) -> bool:
        """Process all changes and return True if updates were made."""
        logger.info("🔍 Checking for file changes...")
        
        changes = self.tracker.get_changed_files(self.kb_path)
        
        total_changes = len(changes['added']) + len(changes['modified']) + len(changes['deleted'])
        if total_changes == 0:
            logger.info("✅ No changes detected")
            return False
        
        logger.info(f"📊 Changes detected: {len(changes['added'])} added, {len(changes['modified'])} modified, {len(changes['deleted'])} deleted")
        
        try:
            # Handle zip files first
            zip_files = [f for f in changes['added'] + changes['modified'] if f.suffix.lower() == '.zip']
            for zip_file in zip_files:
                self._process_zip_file(zip_file)
            
            # Process regular files
            files_to_process = [f for f in changes['added'] + changes['modified'] if f.suffix.lower() != '.zip']
            if files_to_process:
                self._process_files(files_to_process)
            
            # Handle deletions
            if changes['deleted']:
                self._handle_deletions(changes['deleted'])
            
            logger.info("✅ Incremental build completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Incremental build failed: {e}")
            return False
    
    def _process_zip_file(self, zip_path: Path):
        """Process a zip file by extracting and processing its contents."""
        logger.info(f"📦 Processing zip file: {zip_path.name}")
        
        # Create extraction directory
        extract_dir = zip_path.parent / f"_{zip_path.stem}_extracted"
        
        # Remove existing extraction if present
        if extract_dir.exists():
            shutil.rmtree(extract_dir)
        
        # Extract zip file
        if ZipExtractor.extract_zip(zip_path, extract_dir):
            # Process extracted files
            extracted_files = list(extract_dir.rglob("*"))
            file_count = len([f for f in extracted_files if f.is_file()])
            logger.info(f"🔄 Processing {file_count} extracted files...")
            
            # TODO: Integrate with actual RAG and graph building pipeline
            self._build_rag_for_files(extracted_files)
            self._build_graph_for_files(extracted_files)
            
    def _process_files(self, files: List[Path]):
        """Process regular files (PDFs, text, etc.)."""
        logger.info(f"📄 Processing {len(files)} files...")
        
        # TODO: Integrate with actual RAG and graph building pipeline
        self._build_rag_for_files(files)
        self._build_graph_for_files(files)
    
    def _build_rag_for_files(self, files: List[Path]):
        """Build RAG embeddings for new/modified files."""
        # Placeholder for RAG building logic
        logger.info(f"🔍 Building RAG embeddings for {len(files)} files...")
        
    def _build_graph_for_files(self, files: List[Path]):
        """Build knowledge graph nodes/edges for new/modified files.""" 
        # Placeholder for graph building logic
        logger.info(f"🕸️ Building knowledge graph for {len(files)} files...")
    
    def _handle_deletions(self, deleted_paths: List[str]):
        """Handle deleted files by removing associated data."""
        logger.info(f"🗑️ Handling {len(deleted_paths)} deleted files...")
        # TODO: Remove nodes/edges and embeddings for deleted files

class AutoRefreshSubsystem:
    """Main auto-refresh subsystem coordinator."""
    
    def __init__(self, knowledge_base_path: str = "knowledge_base", 
                 output_path: str = "enhanced_kg",
                 debounce_seconds: int = 5):
        self.kb_path = Path(knowledge_base_path)
        self.output_path = Path(output_path)
        self.debounce_seconds = debounce_seconds
        
        # Threading components
        self.refresh_flag = threading.Event()
        self.observer: Optional[Observer] = None
        self.builder_thread: Optional[threading.Thread] = None
        self.running = False
        
        # Builder
        self.builder = IncrementalBuilder(knowledge_base_path, output_path)
        
        # Status tracking
        self.last_refresh = None
        self.refresh_count = 0
        self.is_refreshing = False
        
    def start(self):
        """Start the auto-refresh subsystem."""
        if self.running:
            logger.warning("Auto-refresh already running")
            return
            
        logger.info("🚀 Starting auto-refresh subsystem...")
        
        # Ensure knowledge base directory exists
        self.kb_path.mkdir(exist_ok=True)
        
        # Start file watcher
        self.observer = Observer()
        handler = DebouncedHandler(self.refresh_flag, self.debounce_seconds)
        self.observer.schedule(handler, str(self.kb_path), recursive=True)
        self.observer.start()
        
        # Start builder thread
        self.running = True
        self.builder_thread = threading.Thread(target=self._builder_loop, daemon=True)
        self.builder_thread.start()
        
        logger.info("✅ Auto-refresh subsystem started")
        
        # Do initial scan for changes
        self.refresh_flag.set()
    
    def stop(self):
        """Stop the auto-refresh subsystem."""
        if not self.running:
            return
            
        logger.info("🛑 Stopping auto-refresh subsystem...")
        
        self.running = False
        
        if self.observer:
            self.observer.stop()
            self.observer.join()
        
        if self.builder_thread and self.builder_thread.is_alive():
            self.refresh_flag.set()  # Wake up thread to exit
            self.builder_thread.join(timeout=5)
        
        logger.info("✅ Auto-refresh subsystem stopped")
    
    def _builder_loop(self):
        """Main builder loop running in background thread."""
        logger.info("🔄 Builder thread started")
        
        while self.running:
            try:
                # Wait for refresh signal
                self.refresh_flag.wait()
                self.refresh_flag.clear()
                
                if not self.running:
                    break
                
                # Process changes
                self.is_refreshing = True
                logger.info("🔄 Starting incremental build...")
                
                start_time = time.time()
                changed = self.builder.process_changes()
                elapsed = time.time() - start_time
                
                if changed:
                    self.refresh_count += 1
                    self.last_refresh = datetime.now()
                    logger.info(f"✅ Incremental build completed in {elapsed:.1f}s (refresh #{self.refresh_count})")
                    
                    # TODO: Notify UI components of update
                    # This could update session state or trigger graph reload
                else:
                    logger.info(f"ℹ️ No changes to process ({elapsed:.1f}s)")
                
                self.is_refreshing = False
                
            except Exception as e:
                logger.error(f"❌ Builder loop error: {e}")
                self.is_refreshing = False
                time.sleep(5)  # Wait before retrying
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status of auto-refresh subsystem."""
        return {
            'running': self.running,
            'is_refreshing': self.is_refreshing,
            'last_refresh': self.last_refresh.isoformat() if self.last_refresh else None,
            'refresh_count': self.refresh_count,
            'debounce_seconds': self.debounce_seconds
        }
    
    def trigger_refresh(self):
        """Manually trigger a refresh."""
        logger.info("🔄 Manual refresh triggered")
        self.refresh_flag.set()

# Global instance
_auto_refresh_instance: Optional[AutoRefreshSubsystem] = None

def get_auto_refresh() -> AutoRefreshSubsystem:
    """Get or create global auto-refresh instance."""
    global _auto_refresh_instance
    if _auto_refresh_instance is None:
        _auto_refresh_instance = AutoRefreshSubsystem()
    return _auto_refresh_instance

def start_auto_refresh():
    """Start the auto-refresh subsystem."""
    auto_refresh = get_auto_refresh()
    auto_refresh.start()

def stop_auto_refresh():
    """Stop the auto-refresh subsystem."""
    auto_refresh = get_auto_refresh()
    auto_refresh.stop()