#!/usr/bin/env python3
"""
Validation unit test for data ingest and graph completeness
Tests the fixes for document counts and graph visualization
"""
from pathlib import Path
import pickle
import sys

# Add project root to path
sys.path.append('.')

def test_knowledge_base_exists():
    """Test that knowledge base directory exists and has files."""
    kb_path = Path('knowledge_base')
    assert kb_path.exists(), "Knowledge base directory should exist"
    
    # Check model directories
    model_dirs = [d for d in kb_path.iterdir() if d.is_dir()]
    assert len(model_dirs) > 0, "Should have at least one model directory"

def test_document_counter():
    """Test that document counting works correctly."""
    from app import file_count, get_model_information
    
    # Test file counting function
    kb_path = Path('knowledge_base')
    for model_dir in kb_path.iterdir():
        if model_dir.is_dir():
            count = file_count(model_dir)
            assert count >= 1, f"Model {model_dir.name} should have at least 1 file"
    
    # Test model information retrieval
    model_info = get_model_information()
    assert not model_info.empty, "Model information should not be empty"
    assert "Documents" in model_info.columns, "Should have Documents column"
    
    # Verify document counts are reasonable
    for _, row in model_info.iterrows():
        assert row["Documents"] >= 1, "Each model should have at least 1 document"

def test_enhanced_graph_structure():
    """Test that enhanced graph has proper structure."""
    enhanced_path = Path('enhanced_kg/enhanced_graph.pkl')
    assert enhanced_path.exists(), "Enhanced graph should exist"
    
    with open(enhanced_path, 'rb') as f:
        graph = pickle.load(f)
    
    # Basic graph statistics
    assert graph.number_of_nodes() >= 10, "Should have at least 10 nodes"
    assert graph.number_of_edges() >= 10, "Should have at least 10 edges"
    
    # Check node types diversity
    node_types = set()
    for _, data in graph.nodes(data=True):
        if 'type' in data:
            node_types.add(data['type'])
    
    expected_types = {'MODEL', 'DATASET', 'METRIC', 'AUTHOR', 'FIGURE', 'TABLE'}
    found_types = node_types.intersection(expected_types)
    assert len(found_types) >= 3, f"Should have at least 3 node types, found: {found_types}"

def test_rag_chunks_exist():
    """Test that RAG chunks are properly loaded."""
    rag_path = Path('rag_index/chunks.pkl')
    assert rag_path.exists(), "RAG chunks should exist"
    
    with open(rag_path, 'rb') as f:
        chunks = pickle.load(f)
    
    assert len(chunks) >= 100, "Should have at least 100 RAG chunks"
    
    # Check chunk structure
    if chunks:
        sample_chunk = chunks[0]
        assert 'content' in sample_chunk, "Chunks should have content"

def test_faiss_index_exists():
    """Test that FAISS index exists and is loadable."""
    import faiss
    
    index_path = Path('rag_index/faiss.index')
    assert index_path.exists(), "FAISS index should exist"
    
    # Try to load the index
    index = faiss.read_index(str(index_path))
    assert index.ntotal >= 100, "Index should have at least 100 vectors"

def test_graph_visualization_components():
    """Test that graph visualization has proper components."""
    from app import create_graph_modal
    
    # This is a structural test - check that the function exists and can be imported
    assert callable(create_graph_modal), "Graph modal function should be callable"

def test_system_integration():
    """Test that the hybrid system can be initialized."""
    try:
        from core.hybrid_agent_runner import HybridAgentRunner
        from core.cross_model_analyzer import CrossModelAnalyzer
        
        # Test imports work
        assert HybridAgentRunner is not None
        assert CrossModelAnalyzer is not None
        
    except ImportError as e:
        raise AssertionError(f"System integration test failed: {e}")

if __name__ == "__main__":
    # Run tests individually with detailed output
    test_functions = [
        test_knowledge_base_exists,
        test_document_counter, 
        test_enhanced_graph_structure,
        test_rag_chunks_exist,
        test_faiss_index_exists,
        test_graph_visualization_components,
        test_system_integration
    ]
    
    print("🧪 Running Data Ingest & Graph Completeness Tests")
    print("=" * 50)
    
    passed = 0
    failed = 0
    
    for test_func in test_functions:
        try:
            test_func()
            print(f"✅ {test_func.__name__}")
            passed += 1
        except Exception as e:
            print(f"❌ {test_func.__name__}: {e}")
            failed += 1
    
    print(f"\n📊 Test Results:")
    print(f"   Passed: {passed}")
    print(f"   Failed: {failed}")
    print(f"   Total:  {passed + failed}")
    
    if failed == 0:
        print("\n🎉 All tests PASSED! Data ingest and graph completeness verified.")
    else:
        print(f"\n⚠️  {failed} tests FAILED. Please review issues.")
    
    # Exit with appropriate code
    sys.exit(0 if failed == 0 else 1)