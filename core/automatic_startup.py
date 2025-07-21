#!/usr/bin/env python3
"""
Automatic Startup Manager for Hybrid RAG System
Handles intelligent system initialization with automatic rebuilds when needed
"""

import os
import time
import logging
import threading
from pathlib import Path
from typing import Dict, Any, Callable, Optional
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


class StartupState:
    """Track startup state and progress."""
    INITIALIZING = "initializing"
    CHECKING = "checking"
    REBUILDING = "rebuilding"
    READY = "ready"
    ERROR = "error"


class AutomaticStartupManager:
    """Manages automatic system startup with intelligent rebuilds."""
    
    def __init__(self, knowledge_base_path: str = "knowledge_base"):
        self.kb_path = Path(knowledge_base_path)
        self.state = StartupState.INITIALIZING
        self.progress_message = "System initializing..."
        self.rebuild_thread: Optional[threading.Thread] = None
        self.is_ready = False
        self.startup_time = None
        self.error_message = None
        
        # Progress tracking
        self.progress_callback: Optional[Callable[[str, str], None]] = None
        self.detailed_logs = []
        
    def set_progress_callback(self, callback: Callable[[str, str], None]):
        """Set callback for progress updates (state, message)."""
        self.progress_callback = callback
    
    def _update_progress(self, state: str, message: str):
        """Update progress and notify callback."""
        self.state = state
        self.progress_message = message
        self.detailed_logs.append(f"{datetime.now().strftime('%H:%M:%S')} - {message}")
        
        # Enhanced terminal logging
        logger.info(f"🚀 STARTUP: {message}")
        
        if self.progress_callback:
            self.progress_callback(state, message)
    
    def start_automatic_initialization(self) -> None:
        """Start the automatic initialization process."""
        logger.info("=" * 80)
        logger.info("🚀 AUTOMATIC STARTUP MANAGER INITIATED")
        logger.info(f"📁 Knowledge Base: {self.kb_path}")
        logger.info(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("=" * 80)
        
        self.startup_time = time.time()
        
        # Start initialization in background thread
        self.rebuild_thread = threading.Thread(
            target=self._initialization_worker,
            daemon=True
        )
        self.rebuild_thread.start()
    
    def _initialization_worker(self):
        """Background worker for system initialization."""
        try:
            # Phase 1: System Check
            self._update_progress(StartupState.CHECKING, "🔍 Analyzing knowledge base for changes...")
            
            # Import rebuild manager
            from .smart_rebuild_manager import get_rebuild_manager
            rebuild_manager = get_rebuild_manager()
            
            # Check if auto AI rebuild is enabled
            enable_auto_rebuild = os.getenv('ENABLE_AUTO_AI_REBUILD', '1') == '1'
            
            if not enable_auto_rebuild:
                logger.info("🔒 AUTO AI REBUILD DISABLED - Using existing data")
                self._handle_quick_startup("Auto rebuild disabled in configuration")
                return
            
            # Check if rebuild is needed
            should_rebuild, reason, changes = rebuild_manager.should_rebuild()
            
            if should_rebuild:
                logger.info(f"🔄 REBUILD REQUIRED: {reason}")
                self._handle_automatic_rebuild(rebuild_manager, reason, changes)
            else:
                logger.info(f"✅ SYSTEM UP TO DATE: {reason}")
                self._handle_quick_startup(reason)
            
        except Exception as e:
            logger.error(f"❌ STARTUP FAILED: {e}")
            import traceback
            traceback.print_exc()
            self._update_progress(StartupState.ERROR, f"Startup failed: {str(e)[:100]}...")
            self.error_message = str(e)
    
    def _handle_quick_startup(self, reason: str):
        """Handle quick startup when no rebuild is needed."""
        self._update_progress(StartupState.CHECKING, "⚡ No changes detected - preparing system...")
        
        # Quick validation of existing data
        time.sleep(1)  # Brief pause for any file system checks
        
        # Verify core components exist
        required_files = [
            'rag_index/chunks.pkl',
            'enhanced_kg/enhanced_graph.pkl'
        ]
        
        missing_files = []
        for file_path in required_files:
            if not Path(file_path).exists():
                missing_files.append(file_path)
        
        if missing_files:
            logger.warning(f"⚠️ Missing core files: {missing_files}")
            self._update_progress(StartupState.REBUILDING, "🔧 Missing core files - rebuilding...")
            
            # Force rebuild if core files missing
            from .smart_rebuild_manager import get_rebuild_manager
            rebuild_manager = get_rebuild_manager()
            self._execute_rebuild(rebuild_manager, "Missing core system files")
        else:
            # System ready immediately
            elapsed = time.time() - self.startup_time
            self._update_progress(StartupState.READY, f"✅ System ready! ({elapsed:.1f}s startup)")
            self.is_ready = True
            
            logger.info("🎉 FAST STARTUP COMPLETED")
            logger.info(f"⚡ Total time: {elapsed:.1f} seconds")
            logger.info(f"📊 Reason: {reason}")
    
    def _handle_automatic_rebuild(self, rebuild_manager, reason: str, changes: Dict):
        """Handle automatic rebuild when changes are detected."""
        total_changes = changes.get('total_file_changes', 0)
        new_models = changes.get('new_models', [])
        
        logger.info(f"📊 CHANGE ANALYSIS:")
        logger.info(f"   📝 Files changed: {total_changes}")
        logger.info(f"   📁 New models: {new_models}")
        logger.info(f"   🔄 Reason: {reason}")
        
        self._update_progress(
            StartupState.REBUILDING, 
            f"🤖 Auto-rebuilding system ({total_changes} files changed)..."
        )
        
        self._execute_rebuild(rebuild_manager, reason)
    
    def _execute_rebuild(self, rebuild_manager, reason: str):
        """Execute the actual rebuild process with detailed logging."""
        logger.info("=" * 80)
        logger.info("🔨 AUTOMATIC REBUILD INITIATED")
        logger.info("=" * 80)
        logger.info(f"📋 Trigger Reason: {reason}")
        logger.info(f"⏰ Started at: {datetime.now().strftime('%H:%M:%S')}")
        
        # Phase 1: Environment Analysis
        self._update_progress(StartupState.REBUILDING, "📋 Analyzing knowledge base structure...")
        logger.info("\n🔍 PHASE 1: ENVIRONMENT ANALYSIS")
        
        current_state = rebuild_manager.hasher.compute_knowledge_base_hash()
        total_files = current_state['total_files']
        models = list(current_state['models'].keys())
        total_size_mb = current_state['total_size'] / (1024 * 1024)
        
        logger.info(f"📊 KNOWLEDGE BASE INVENTORY:")
        logger.info(f"   📄 Total files: {total_files:,}")
        logger.info(f"   📁 Model folders: {len(models)} ({', '.join(models)})")
        logger.info(f"   💾 Total size: {total_size_mb:.1f} MB ({current_state['total_size']:,} bytes)")
        logger.info(f"   🕐 Last modified: {datetime.fromtimestamp(current_state['last_modified']).strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Model-by-model breakdown
        for model_name, model_info in current_state['models'].items():
            model_size_mb = model_info['total_size'] / (1024 * 1024)
            logger.info(f"   └─ {model_name}: {model_info['file_count']:,} files, {model_size_mb:.1f} MB")
        
        # Phase 2: AI Processing Pipeline
        self._update_progress(
            StartupState.REBUILDING, 
            f"🤖 Starting AI analysis pipeline for {total_files:,} files..."
        )
        logger.info(f"\n🤖 PHASE 2: AI PROCESSING PIPELINE")
        logger.info(f"🎯 Target: {total_files:,} files across {len(models)} models")
        logger.info("🔄 Pipeline: RAG Chunking → Graph Extraction → Vector Indexing")
        
        try:
            # Execute the rebuild
            from app import trigger_ai_rebuild
            
            logger.info("\n🚀 EXECUTING AI REBUILD PIPELINE")
            logger.info("   ├─ Loading AI models and dependencies...")
            logger.info("   ├─ Processing files through AI analyzers...")
            logger.info("   ├─ Extracting entities and relationships...")
            logger.info("   ├─ Building vector embeddings...")
            logger.info("   └─ Constructing knowledge graph...")
            
            rebuild_start = time.time()
            
            success = trigger_ai_rebuild()
            
            rebuild_time = time.time() - rebuild_start
            total_startup_time = time.time() - self.startup_time
            
            if success:
                # Update build state
                rebuild_manager.update_build_state_after_rebuild()
                
                # Get final statistics
                final_state = rebuild_manager.hasher.compute_knowledge_base_hash()
                
                logger.info("=" * 80)
                logger.info("🎉 AUTOMATIC REBUILD COMPLETED SUCCESSFULLY")
                logger.info("=" * 80)
                logger.info(f"⏱️  Pure rebuild time: {rebuild_time:.1f} seconds")
                logger.info(f"🕐 Total startup time: {total_startup_time:.1f} seconds")
                logger.info(f"📊 Files processed: {total_files:,}")
                logger.info(f"📈 Processing rate: {total_files/rebuild_time:.1f} files/second")
                logger.info(f"💾 Data processed: {total_size_mb:.1f} MB")
                logger.info(f"🎯 System status: READY FOR USE")
                logger.info("=" * 80)
                
                self._update_progress(
                    StartupState.READY, 
                    f"✅ System ready! (Processed {total_files:,} files in {total_startup_time:.1f}s)"
                )
                self.is_ready = True
                
            else:
                logger.error("=" * 80)
                logger.error("❌ AUTOMATIC REBUILD FAILED")
                logger.error("=" * 80)
                logger.error(f"⏱️  Failed after: {rebuild_time:.1f} seconds")
                logger.error(f"📊 Target files: {total_files:,}")
                logger.error("🔍 Check logs above for specific error details")
                logger.error("💡 Try restarting the application or check knowledge base files")
                logger.error("=" * 80)
                
                self._update_progress(
                    StartupState.ERROR, 
                    "❌ Automatic rebuild failed - check terminal logs for details"
                )
                self.error_message = "Rebuild process failed"
                
        except Exception as e:
            # Handle OpenAI quota errors gracefully
            if "insufficient_quota" in str(e) or "429" in str(e):
                logger.warning("=" * 80)
                logger.warning("⚠️ OPENAI API QUOTA EXCEEDED - CONTINUING WITH EXISTING DATA")
                logger.warning("=" * 80)
                logger.warning("💡 The system will continue with existing knowledge base data")
                logger.warning("💡 To restore AI processing, check your OpenAI billing:")
                logger.warning("💡 https://platform.openai.com/usage")
                logger.warning("=" * 80)
                
                # Mark as ready with existing data
                total_startup_time = time.time() - self.startup_time
                self._update_progress(
                    StartupState.READY,
                    f"⚡ System ready with existing data! (OpenAI quota exceeded - {total_startup_time:.1f}s)"
                )
                self.is_ready = True
            else:
                logger.error(f"❌ REBUILD EXCEPTION: {e}")
                import traceback
                traceback.print_exc()
                
                self._update_progress(
                    StartupState.ERROR,
                    f"❌ Rebuild error: {str(e)[:100]}..."
                )
                self.error_message = str(e)
    
    def wait_for_ready(self, timeout: float = 300.0) -> bool:
        """
        Wait for system to be ready.
        
        Args:
            timeout: Maximum time to wait in seconds
            
        Returns:
            True if system is ready, False if timeout or error
        """
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if self.is_ready:
                return True
            elif self.state == StartupState.ERROR:
                return False
            
            time.sleep(0.5)
        
        logger.error(f"⏰ STARTUP TIMEOUT after {timeout}s")
        return False
    
    def get_status(self) -> Dict[str, Any]:
        """Get current startup status."""
        elapsed = time.time() - self.startup_time if self.startup_time else 0
        
        return {
            'state': self.state,
            'message': self.progress_message,
            'is_ready': self.is_ready,
            'elapsed_time': elapsed,
            'error_message': self.error_message,
            'detailed_logs': self.detailed_logs[-10:]  # Last 10 log entries
        }
    
    def force_rebuild(self) -> None:
        """Force a rebuild regardless of change detection."""
        if self.rebuild_thread and self.rebuild_thread.is_alive():
            logger.warning("⚠️ Rebuild already in progress")
            return
        
        logger.info("🔄 FORCING MANUAL REBUILD")
        self.is_ready = False
        self.state = StartupState.REBUILDING
        
        # Start rebuild thread
        self.rebuild_thread = threading.Thread(
            target=self._force_rebuild_worker,
            daemon=True
        )
        self.rebuild_thread.start()
    
    def _force_rebuild_worker(self):
        """Worker for forced rebuild."""
        try:
            from .smart_rebuild_manager import get_rebuild_manager
            rebuild_manager = get_rebuild_manager()
            self._execute_rebuild(rebuild_manager, "Manual rebuild requested")
        except Exception as e:
            logger.error(f"❌ FORCED REBUILD FAILED: {e}")
            self._update_progress(StartupState.ERROR, f"Manual rebuild failed: {e}")


# Global instance
_startup_manager: Optional[AutomaticStartupManager] = None

def get_startup_manager() -> AutomaticStartupManager:
    """Get or create global startup manager instance."""
    global _startup_manager
    if _startup_manager is None:
        _startup_manager = AutomaticStartupManager()
    return _startup_manager


def initialize_system_automatically() -> AutomaticStartupManager:
    """Initialize the system automatically and return manager for status tracking."""
    manager = get_startup_manager()
    
    if not manager.rebuild_thread or not manager.rebuild_thread.is_alive():
        manager.start_automatic_initialization()
    
    return manager