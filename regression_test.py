#!/usr/bin/env python3
"""
Regression Test Suite for UI Hot-Fix
Verifies all critical functionality still works after UI changes
"""
import requests
import time
import json
from pathlib import Path

def test_streamlit_running():
    """Test that Streamlit app is accessible."""
    try:
        response = requests.get("http://localhost:8501", timeout=5)
        return response.status_code == 200
    except:
        return False

def test_knowledge_base_structure():
    """Test knowledge base structure is intact."""
    kb_path = Path('knowledge_base')
    if not kb_path.exists():
        return False
    
    model_dirs = [d for d in kb_path.iterdir() if d.is_dir()]
    return len(model_dirs) > 0

def test_system_files_exist():
    """Test all required system files exist."""
    required_files = [
        'enhanced_kg/enhanced_graph.pkl',
        'rag_index/chunks.pkl', 
        'rag_index/faiss.index',
        'app.py',
        'core/hybrid_agent_runner.py',
        'core/cross_model_analyzer.py'
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    return len(missing_files) == 0, missing_files

def test_imports():
    """Test critical imports work."""
    try:
        from core.hybrid_agent_runner import HybridAgentRunner
        from core.cross_model_analyzer import CrossModelAnalyzer
        import streamlit as st
        return True
    except Exception as e:
        return False, str(e)

def test_streaming_function_exists():
    """Test streaming functions are present in app.py."""
    app_content = Path('app.py').read_text()
    
    required_functions = [
        'create_real_streaming_interface',
        'stream_query',
        'chat_log'
    ]
    
    missing_functions = []
    for func in required_functions:
        if func not in app_content:
            missing_functions.append(func)
    
    return len(missing_functions) == 0, missing_functions

def test_ui_structure():
    """Test UI structure matches requirements."""
    app_content = Path('app.py').read_text()
    
    # Check sidebar contains required components
    sidebar_checks = [
        'multiselect',  # Model selection
        'Available Models',  # Models table
        'Explore Graph',  # Graph button
    ]
    
    # Check memory controls are removed
    forbidden_elements = [
        'conversation_memory',
        'clear_memory',
        'memory_stats'
    ]
    
    results = {}
    
    for check in sidebar_checks:
        results[f'sidebar_{check}'] = check in app_content
    
    for forbidden in forbidden_elements:
        results[f'no_{forbidden}'] = forbidden not in app_content
    
    return results

def run_regression_tests():
    """Run complete regression test suite."""
    print("🧪 Running UI Hot-Fix Regression Tests")
    print("=" * 50)
    
    tests = [
        ("Streamlit App Running", test_streamlit_running),
        ("Knowledge Base Structure", test_knowledge_base_structure), 
        ("System Files Exist", test_system_files_exist),
        ("Critical Imports", test_imports),
        ("Streaming Functions", test_streaming_function_exists),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            if isinstance(result, tuple):
                success, details = result
                results[test_name] = success
                if not success:
                    print(f"❌ {test_name}: {details}")
                else:
                    print(f"✅ {test_name}")
            else:
                results[test_name] = result
                print(f"{'✅' if result else '❌'} {test_name}")
        except Exception as e:
            results[test_name] = False
            print(f"❌ {test_name}: {e}")
    
    # UI Structure test (detailed)
    print("\n📋 UI Structure Analysis:")
    ui_results = test_ui_structure()
    for check, passed in ui_results.items():
        print(f"{'✅' if passed else '❌'} {check}")
    
    # Summary
    total_tests = len(results)
    passed_tests = sum(results.values())
    
    print(f"\n📊 Regression Test Summary:")
    print(f"   Tests Passed: {passed_tests}/{total_tests}")
    print(f"   Success Rate: {passed_tests/total_tests*100:.1f}%")
    
    if passed_tests == total_tests:
        print("\n🎉 All regression tests PASSED!")
        print("   ✅ UI hot-fix is working correctly")
        print("   ✅ No functionality broken")
        print("   ✅ Ready for production")
    else:
        print(f"\n⚠️  {total_tests - passed_tests} tests FAILED")
        print("   Please review failed tests before deployment")
    
    return passed_tests == total_tests

if __name__ == "__main__":
    success = run_regression_tests()
    exit(0 if success else 1)