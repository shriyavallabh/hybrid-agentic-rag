#!/usr/bin/env python3
"""
Comprehensive Test Suite for Auto-Refresh Subsystem
Demonstrates the full functionality of the incremental knowledge base builder
"""
import time
from pathlib import Path
from core.auto_refresh import get_auto_refresh

def test_auto_refresh_comprehensive():
    """Comprehensive test of auto-refresh functionality."""
    print("🧪 Testing Auto-Refresh Subsystem")
    print("=" * 50)
    
    # Get auto-refresh instance
    auto_refresh = get_auto_refresh()
    
    # Test 1: System Status
    print("1️⃣ System Status Check:")
    status = auto_refresh.get_status()
    print(f"   Running: {status['running']}")
    print(f"   Currently Refreshing: {status['is_refreshing']}")
    print(f"   Total Refreshes: {status['refresh_count']}")
    print(f"   Last Refresh: {status['last_refresh'] or 'Never'}")
    print(f"   Debounce Period: {status['debounce_seconds']}s")
    
    # Test 2: Knowledge Base Statistics
    print("\n2️⃣ Knowledge Base Statistics:")
    kb_path = Path('knowledge_base')
    
    total_files = sum(1 for f in kb_path.rglob('*') if f.is_file())
    python_files = len(list(kb_path.rglob('*.py')))
    markdown_files = len(list(kb_path.rglob('*.md')))
    json_files = len(list(kb_path.rglob('*.json')))
    pdf_files = len(list(kb_path.rglob('*.pdf')))
    zip_files = len(list(kb_path.rglob('*.zip')))
    
    print(f"   Total Files: {total_files}")
    print(f"   Python Files: {python_files}")
    print(f"   Markdown Files: {markdown_files}")
    print(f"   JSON Files: {json_files}")
    print(f"   PDF Files: {pdf_files}")
    print(f"   ZIP Files: {zip_files}")
    
    # Test 3: File Index Status
    print("\n3️⃣ File Change Tracking:")
    tracker = auto_refresh.builder.tracker
    index_size = len(tracker.file_index)
    print(f"   Tracked Files: {index_size}")
    print(f"   Index File: {tracker.index_path}")
    print(f"   Index Exists: {tracker.index_path.exists()}")
    
    # Test 4: Extracted Content Check
    print("\n4️⃣ ZIP Extraction Verification:")
    extracted_dirs = list(kb_path.rglob('*_extracted'))
    print(f"   Extracted Directories: {len(extracted_dirs)}")
    for dir_path in extracted_dirs:
        files_in_dir = sum(1 for f in dir_path.rglob('*') if f.is_file())
        print(f"   {dir_path.name}: {files_in_dir} files")
    
    # Test 5: Manual Refresh Test
    print("\n5️⃣ Manual Refresh Test:")
    print("   Triggering manual refresh...")
    auto_refresh.trigger_refresh()
    
    # Wait a moment for processing
    time.sleep(2)
    
    updated_status = auto_refresh.get_status()
    print(f"   Refresh Count After Trigger: {updated_status['refresh_count']}")
    
    # Test 6: Integration Verification
    print("\n6️⃣ Integration Verification:")
    try:
        from app import get_model_information
        model_info = get_model_information()
        if not model_info.empty:
            for _, row in model_info.iterrows():
                print(f"   Model: {row['Model']}")
                print(f"   Documents: {row['Documents']}")
                print(f"   Last Updated: {row['Last Updated']}")
        else:
            print("   ❌ No model information available")
    except Exception as e:
        print(f"   ❌ Integration test failed: {e}")
    
    # Test 7: File Types Distribution
    print("\n7️⃣ File Types Distribution:")
    file_types = {}
    for file_path in kb_path.rglob('*'):
        if file_path.is_file():
            ext = file_path.suffix.lower()
            file_types[ext] = file_types.get(ext, 0) + 1
    
    # Show top 10 file types
    sorted_types = sorted(file_types.items(), key=lambda x: x[1], reverse=True)
    for ext, count in sorted_types[:10]:
        ext_display = ext if ext else '(no extension)'
        print(f"   {ext_display}: {count} files")
    
    print("\n✅ Auto-Refresh System Test Complete!")
    print(f"📊 Total Knowledge Base Growth: {total_files} files")
    print("🔄 System is actively monitoring for changes")
    
    return {
        'total_files': total_files,
        'python_files': python_files,
        'markdown_files': markdown_files,
        'json_files': json_files,
        'refresh_count': updated_status['refresh_count'],
        'system_running': updated_status['running']
    }

if __name__ == "__main__":
    results = test_auto_refresh_comprehensive()
    
    # Success criteria
    success_criteria = [
        ('System Running', results['system_running']),
        ('Has Files', results['total_files'] > 100),
        ('Has Python Files', results['python_files'] > 50),
        ('Has Documentation', results['markdown_files'] > 10),
        ('Processing Active', results['refresh_count'] > 0),
    ]
    
    print(f"\n🎯 Success Criteria:")
    passed = 0
    for criterion, result in success_criteria:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {status} {criterion}")
        if result:
            passed += 1
    
    print(f"\n📈 Overall Score: {passed}/{len(success_criteria)} criteria met")
    
    if passed == len(success_criteria):
        print("🎉 Auto-Refresh Subsystem: FULLY OPERATIONAL!")
    else:
        print("⚠️ Auto-Refresh Subsystem: Needs attention")