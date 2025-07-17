"""
Test suite for Model Knowledge Counselor
"""
import os
import sys
import pytest
import tempfile
import shutil
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.graph_builder import GraphBuilder
from core.agent_runner import AgentRunner


class TestKnowledgeGraphBuild:
    """Test knowledge graph building functionality."""
    
    def test_smoke_build(self):
        """Smoke test - run builder on knowledge_base/sample/, assert kg_bundle/graph.pkl exists & >= 1 node."""
        # Skip if no OpenAI key
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set")
        
        # Create temporary directories
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            kb_path = temp_path / "knowledge_base" / "sample_model"
            kg_path = temp_path / "kg_bundle"
            
            kb_path.mkdir(parents=True)
            
            # Create sample content
            (kb_path / "sample_doc.txt").write_text("""
            Sample Banking Model Documentation
            
            This model predicts loan default probability using customer data.
            The model uses the following datasets:
            - Customer demographics dataset
            - Transaction history data
            
            Performance metrics:
            - AUC: 0.85
            - Precision: 0.78
            - Recall: 0.82
            """)
            
            (kb_path / "model.py").write_text("""
            def calculate_risk_score(customer_data):
                # Calculate risk score based on customer features
                return risk_score
            
            def preprocess_data(raw_data):
                # Clean and prepare data for modeling
                return processed_data
            """)
            
            # Build knowledge graph
            builder = GraphBuilder(
                knowledge_base_path=str(temp_path / "knowledge_base"),
                output_path=str(kg_path),
                openai_api_key=os.getenv("OPENAI_API_KEY")
            )
            
            manifest = builder.build()
            
            # Assertions
            assert (kg_path / "graph.pkl").exists(), "graph.pkl should exist"
            assert manifest["node_count"] >= 1, "Should have at least 1 node"
            assert (kg_path / "faiss.index").exists(), "FAISS index should exist"
            assert (kg_path / "meta.json").exists(), "Metadata should exist"
    
    def test_import_lint(self):
        """Fail test if any module path contains 'reference_only'."""
        # Check all Python files in the project
        project_root = Path(__file__).parent.parent
        
        for py_file in project_root.rglob("*.py"):
            if "reference_only" in str(py_file):
                pytest.fail(f"Found reference_only in path: {py_file}")
            
            # Also check import statements
            try:
                content = py_file.read_text()
                if "reference_only" in content:
                    pytest.fail(f"Found reference_only import in: {py_file}")
            except Exception:
                # Skip files that can't be read
                pass


class TestAgentQuery:
    """Test agent querying functionality."""
    
    def test_smoke_query(self):
        """Smoke test - ask 'Which dataset does sample_model use?' expect non-empty answer in < 20s."""
        # Skip if no OpenAI key
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set")
        
        # First build a small KG
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            kb_path = temp_path / "knowledge_base" / "sample_model"
            kg_path = temp_path / "kg_bundle"
            
            kb_path.mkdir(parents=True)
            
            # Create sample content with clear dataset references
            (kb_path / "model_doc.txt").write_text("""
            Sample Model Documentation
            
            This banking model uses the Customer Demographics Dataset and
            Transaction History Dataset to predict loan defaults.
            
            The model achieves an AUC of 0.85 on the test set.
            """)
            
            # Build KG
            builder = GraphBuilder(
                knowledge_base_path=str(temp_path / "knowledge_base"),
                output_path=str(kg_path),
                openai_api_key=os.getenv("OPENAI_API_KEY")
            )
            builder.build()
            
            # Test query
            runner = AgentRunner(
                kg_path=str(kg_path),
                openai_api_key=os.getenv("OPENAI_API_KEY"),
                token_budget=10000  # Small budget for test
            )
            
            import time
            start_time = time.time()
            
            result = runner.query("Which dataset does sample_model use?")
            
            query_time = time.time() - start_time
            
            # Assertions
            assert query_time < 20, f"Query took {query_time:.1f}s, should be < 20s"
            assert result.answer.strip(), "Answer should not be empty"
            assert len(result.trace) > 0, "Should have reasoning trace"
            
            # Check if answer mentions datasets
            answer_lower = result.answer.lower()
            assert any(word in answer_lower for word in ["dataset", "data", "customer", "transaction"]), \
                "Answer should mention datasets"


class TestSystemDependencies:
    """Test system dependency checks."""
    
    def test_dependency_check(self):
        """Test that system dependencies are properly checked."""
        # This will fail if dependencies are missing
        try:
            builder = GraphBuilder(
                knowledge_base_path="dummy",
                output_path="dummy", 
                openai_api_key="dummy"
            )
            # If we get here, dependencies are available
            assert True
        except RuntimeError as e:
            if "Missing system dependencies" in str(e):
                pytest.skip(f"System dependencies missing: {e}")
            else:
                raise


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "-s"])