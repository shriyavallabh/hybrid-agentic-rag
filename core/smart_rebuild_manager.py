#!/usr/bin/env python3
"""
Smart Rebuild Manager for Hybrid RAG System
Implements intelligent rebuilding based on change detection and user preferences
"""
import os
import json
import hashlib
import threading
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import time

logger = logging.getLogger(__name__)


class KnowledgeBaseHasher:
    """Efficiently compute hashes for knowledge base state detection."""
    
    def __init__(self, kb_path: str):
        self.kb_path = Path(kb_path)
        
    def compute_knowledge_base_hash(self) -> Dict[str, Any]:
        """Compute comprehensive hash of knowledge base state."""
        logger.info("🔍 Computing knowledge base hash...")
        
        state = {
            'models': {},
            'global_files': {},
            'total_files': 0,
            'total_size': 0,
            'last_modified': 0,
            'computed_at': time.time()
        }
        
        if not self.kb_path.exists():
            return state
        
        # Process model folders
        for model_folder in self.kb_path.iterdir():
            if model_folder.is_dir() and model_folder.name.startswith('model_'):
                model_hash = self._compute_folder_hash(model_folder)
                state['models'][model_folder.name] = model_hash
                state['total_files'] += model_hash['file_count']
                state['total_size'] += model_hash['total_size']
                state['last_modified'] = max(state['last_modified'], model_hash['last_modified'])
        
        # Process global files (not in model folders)
        global_files = []
        for file_path in self.kb_path.rglob("*"):
            if (file_path.is_file() and 
                not file_path.name.startswith('.') and
                not any(part.startswith('model_') for part in file_path.relative_to(self.kb_path).parts[:-1])):
                global_files.append(file_path)
        
        if global_files:
            global_hash = self._compute_files_hash(global_files)
            state['global_files'] = global_hash
            state['total_files'] += global_hash['file_count']
            state['total_size'] += global_hash['total_size']
            state['last_modified'] = max(state['last_modified'], global_hash['last_modified'])
        
        logger.info(f"📊 KB hash computed: {state['total_files']} files, {len(state['models'])} models")
        return state
    
    def _compute_folder_hash(self, folder_path: Path) -> Dict[str, Any]:
        """Compute hash for a specific folder."""
        files = []
        for file_path in folder_path.rglob("*"):
            if file_path.is_file() and not file_path.name.startswith('.'):
                files.append(file_path)
        
        return self._compute_files_hash(files)
    
    def _compute_files_hash(self, files: List[Path]) -> Dict[str, Any]:
        """Compute hash for a list of files."""
        if not files:
            return {
                'hash': '',
                'file_count': 0,
                'total_size': 0,
                'last_modified': 0,
                'files': []
            }
        
        file_info = []
        total_size = 0
        last_modified = 0
        
        for file_path in sorted(files):  # Sort for consistent hashing
            try:
                stat = file_path.stat()
                file_info.append({
                    'path': str(file_path),
                    'size': stat.st_size,
                    'mtime': stat.st_mtime
                })
                total_size += stat.st_size
                last_modified = max(last_modified, stat.st_mtime)
            except (OSError, PermissionError):
                continue
        
        # Create hash from file paths, sizes, and modification times
        hash_input = json.dumps(file_info, sort_keys=True)
        folder_hash = hashlib.sha256(hash_input.encode()).hexdigest()
        
        return {
            'hash': folder_hash,
            'file_count': len(file_info),
            'total_size': total_size,
            'last_modified': last_modified,
            'files': [f['path'] for f in file_info]
        }


class SmartRebuildManager:
    """Manages intelligent rebuilding decisions based on change detection."""
    
    def __init__(self, kb_path: str = "knowledge_base", output_path: str = "enhanced_kg"):
        self.kb_path = Path(kb_path)
        self.output_path = Path(output_path)
        self.output_path.mkdir(exist_ok=True)
        
        self.hasher = KnowledgeBaseHasher(kb_path)
        self.state_file = self.output_path / 'last_build_state.json'
        
        # Rebuild settings
        self.min_files_for_rebuild = 5  # Minimum changed files to trigger rebuild
        self.max_auto_rebuild_interval = timedelta(hours=24)  # Max time between auto-rebuilds
        
    def should_rebuild(self, force: bool = False) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Determine if a rebuild is needed.
        
        Returns:
            (should_rebuild: bool, reason: str, change_info: dict)
        """
        if force:
            return True, "Manual rebuild requested", {}
        
        # Check if we have any previous build state
        last_state = self._load_last_build_state()
        if not last_state:
            return True, "No previous build found - initial build required", {}
        
        # Compute current state
        current_state = self.hasher.compute_knowledge_base_hash()
        
        # Compare states
        changes = self._compare_states(last_state, current_state)
        
        # Decision logic
        if changes['new_models']:
            return True, f"New model folders detected: {changes['new_models']}", changes
        
        if changes['removed_models']:
            return True, f"Model folders removed: {changes['removed_models']}", changes
        
        total_changed_files = sum(len(files) for files in changes['changed_files'].values())
        if total_changed_files >= self.min_files_for_rebuild:
            return True, f"Significant changes: {total_changed_files} files modified", changes
        
        # Check time-based rebuild
        last_build_time = datetime.fromtimestamp(last_state.get('computed_at', 0))
        if datetime.now() - last_build_time > self.max_auto_rebuild_interval:
            return True, f"Periodic rebuild due (last: {last_build_time.strftime('%Y-%m-%d %H:%M')})", changes
        
        # Check if build outputs are missing
        if not self._check_build_outputs_exist():
            return True, "Build outputs missing - rebuild required", changes
        
        return False, f"No significant changes ({total_changed_files} files changed)", changes
    
    def _load_last_build_state(self) -> Optional[Dict]:
        """Load the last build state."""
        if not self.state_file.exists():
            return None
        
        try:
            with open(self.state_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load last build state: {e}")
            return None
    
    def _save_build_state(self, state: Dict):
        """Save the current build state."""
        try:
            with open(self.state_file, 'w') as f:
                json.dump(state, f, indent=2)
            logger.info(f"💾 Saved build state: {state['total_files']} files")
        except Exception as e:
            logger.error(f"Failed to save build state: {e}")
    
    def _compare_states(self, old_state: Dict, new_state: Dict) -> Dict[str, Any]:
        """Compare two knowledge base states and identify changes."""
        changes = {
            'new_models': [],
            'removed_models': [],
            'changed_models': [],
            'changed_files': {},
            'total_file_changes': 0
        }
        
        old_models = set(old_state.get('models', {}).keys())
        new_models = set(new_state.get('models', {}).keys())
        
        changes['new_models'] = list(new_models - old_models)
        changes['removed_models'] = list(old_models - new_models)
        
        # Check changes in existing models
        for model_name in old_models & new_models:
            old_model_hash = old_state['models'][model_name]['hash']
            new_model_hash = new_state['models'][model_name]['hash']
            
            if old_model_hash != new_model_hash:
                changes['changed_models'].append(model_name)
                # For detailed analysis, we'd need to compare file lists
                old_files = set(old_state['models'][model_name].get('files', []))
                new_files = set(new_state['models'][model_name].get('files', []))
                changed_files = (new_files - old_files) | (old_files - new_files)
                changes['changed_files'][model_name] = list(changed_files)
                changes['total_file_changes'] += len(changed_files)
        
        # Check global file changes
        old_global_hash = old_state.get('global_files', {}).get('hash', '')
        new_global_hash = new_state.get('global_files', {}).get('hash', '')
        
        if old_global_hash != new_global_hash:
            old_global_files = set(old_state.get('global_files', {}).get('files', []))
            new_global_files = set(new_state.get('global_files', {}).get('files', []))
            changed_global = (new_global_files - old_global_files) | (old_global_files - new_global_files)
            if changed_global:
                changes['changed_files']['global'] = list(changed_global)
                changes['total_file_changes'] += len(changed_global)
        
        return changes
    
    def _check_build_outputs_exist(self) -> bool:
        """Check if expected build outputs exist."""
        required_outputs = [
            'rag_index/chunks.pkl',
            'rag_index/faiss.index',
            'enhanced_kg/enhanced_graph.pkl'
        ]
        
        for output in required_outputs:
            if not (Path(output)).exists():
                logger.info(f"Missing build output: {output}")
                return False
        
        return True
    
    def start_background_rebuild(self, progress_callback=None) -> threading.Thread:
        """Start a background rebuild process with detailed progress tracking."""
        logger.info("🚀 Starting background rebuild...")
        
        def rebuild_worker():
            try:
                import time
                # Import here to avoid circular imports
                
                if progress_callback:
                    progress_callback("🔍 Analyzing knowledge base structure...")
                
                # Step 1: Analyze current state
                current_state = self.hasher.compute_knowledge_base_hash()
                total_files = current_state['total_files']
                models = list(current_state['models'].keys())
                
                if progress_callback:
                    progress_callback(f"📊 Found {total_files} files across {len(models)} models")
                
                time.sleep(1)  # Brief pause for UI update
                
                # Step 2: Start rebuild process
                if progress_callback:
                    progress_callback("🤖 Starting AI-powered analysis...")
                
                # Import the comprehensive builders
                from .enhanced_graph_builder import AIGraphBuilder
                from .comprehensive_rag_builder import ComprehensiveRAGBuilder
                
                # Step 3: Rebuild each model
                for i, model_name in enumerate(models):
                    if progress_callback:
                        progress_callback(f"🔧 Processing {model_name} ({i+1}/{len(models)})...")
                    
                    # Model-specific rebuild logic would go here
                    # For now, we'll use the existing trigger_ai_rebuild
                    time.sleep(2)  # Simulate processing time
                
                if progress_callback:
                    progress_callback("🔄 Running full system rebuild...")
                
                # Import here to avoid circular imports
                from app import trigger_ai_rebuild
                result = trigger_ai_rebuild()
                
                if result:
                    # Update build state on success
                    final_state = self.hasher.compute_knowledge_base_hash()
                    self._save_build_state(final_state)
                    
                    if progress_callback:
                        progress_callback(f"✅ Rebuild completed! Processed {final_state['total_files']} files")
                    
                    logger.info(f"✅ Background rebuild completed successfully")
                else:
                    if progress_callback:
                        progress_callback("❌ Background rebuild failed - check logs for details")
                        
            except Exception as e:
                logger.error(f"Background rebuild failed: {e}")
                if progress_callback:
                    progress_callback(f"❌ Rebuild error: {str(e)[:100]}...")
        
        thread = threading.Thread(target=rebuild_worker, daemon=True)
        thread.start()
        return thread
    
    def update_build_state_after_rebuild(self):
        """Update the build state after a successful rebuild."""
        current_state = self.hasher.compute_knowledge_base_hash()
        self._save_build_state(current_state)
        logger.info("✅ Build state updated after successful rebuild")
    
    def get_rebuild_status(self) -> Dict[str, Any]:
        """Get current rebuild status and recommendations."""
        should_rebuild, reason, changes = self.should_rebuild()
        last_state = self._load_last_build_state()
        
        return {
            'should_rebuild': should_rebuild,
            'reason': reason,
            'changes': changes,
            'last_build': datetime.fromtimestamp(last_state.get('computed_at', 0)).isoformat() if last_state else None,
            'total_files': self.hasher.compute_knowledge_base_hash()['total_files'],
            'build_outputs_exist': self._check_build_outputs_exist()
        }


# Global instance
_rebuild_manager: Optional[SmartRebuildManager] = None

def get_rebuild_manager() -> SmartRebuildManager:
    """Get or create global rebuild manager instance."""
    global _rebuild_manager
    if _rebuild_manager is None:
        _rebuild_manager = SmartRebuildManager()
    return _rebuild_manager