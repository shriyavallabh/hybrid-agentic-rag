"""
Create synthetic demo bundle for instant startup
"""
import pickle
import json
import networkx as nx
import numpy as np
import faiss
from datetime import datetime


def create_demo_bundle():
    """Create a synthetic knowledge graph for demo mode."""
    
    # Create synthetic graph
    graph = nx.MultiDiGraph()
    
    # Add nodes for 3 toy models
    nodes = [
        # Loan PD Model
        {"id": "loan_pd_1", "type": "Model", "label": "Loan PD Model v2.1", 
         "details": "Predicts probability of default for personal loans using gradient boosting", 
         "namespace": "loan_pd_model", "page": 1},
        {"id": "loan_pd_2", "type": "Dataset", "label": "Customer Demographics", 
         "details": "Age, income, employment status, credit history for 100k customers",
         "namespace": "loan_pd_model", "page": 2},
        {"id": "loan_pd_3", "type": "Metric", "label": "AUC Score", 
         "details": "Area under ROC curve = 0.87 on validation set",
         "namespace": "loan_pd_model", "page": 3},
        {"id": "loan_pd_4", "type": "CodeEntity", "label": "risk_calculator", 
         "details": "Main function that computes default probability score",
         "namespace": "loan_pd_model", "page": 4},
        
        # Fraud Detection Model
        {"id": "fraud_1", "type": "Model", "label": "Fraud Detection v3.0", 
         "details": "Real-time fraud detection using neural networks and rule engine",
         "namespace": "fraud_model", "page": 1},
        {"id": "fraud_2", "type": "Dataset", "label": "Transaction History", 
         "details": "5M transactions with fraud labels, amounts, merchants, locations",
         "namespace": "fraud_model", "page": 2},
        {"id": "fraud_3", "type": "Metric", "label": "Precision at 1%", 
         "details": "Precision = 0.92 at 1% false positive rate",
         "namespace": "fraud_model", "page": 3},
        {"id": "fraud_4", "type": "CodeEntity", "label": "fraud_scorer", 
         "details": "Neural network model that outputs fraud probability",
         "namespace": "fraud_model", "page": 4},
        
        # Credit Scoring Model  
        {"id": "credit_1", "type": "Model", "label": "Credit Score Model", 
         "details": "FICO-like credit scoring using traditional and alternative data",
         "namespace": "credit_model", "page": 1},
        {"id": "credit_2", "type": "Dataset", "label": "Credit Bureau Data", 
         "details": "Payment history, credit utilization, length of credit history",
         "namespace": "credit_model", "page": 2},
        {"id": "credit_3", "type": "Metric", "label": "Gini Coefficient", 
         "details": "Gini = 0.72 measuring model discriminatory power",
         "namespace": "credit_model", "page": 3},
        {"id": "credit_4", "type": "CodeEntity", "label": "score_calculator", 
         "details": "Combines multiple factors into final credit score 300-850",
         "namespace": "credit_model", "page": 4},
        
        # Shared components
        {"id": "shared_1", "type": "Dataset", "label": "Customer Master", 
         "details": "Central customer database with demographics and relationships",
         "namespace": "shared", "page": 1},
        {"id": "shared_2", "type": "CodeEntity", "label": "data_preprocessor", 
         "details": "Common data cleaning and feature engineering functions",
         "namespace": "shared", "page": 1},
    ]
    
    # Add all nodes
    for node in nodes:
        graph.add_node(node["id"], **node)
    
    # Add edges
    edges = [
        # Loan PD relationships
        {"source": "loan_pd_1", "target": "loan_pd_2", "type": "USES_DATASET", "confidence": 0.95},
        {"source": "loan_pd_1", "target": "shared_1", "type": "USES_DATASET", "confidence": 0.90},
        {"source": "loan_pd_1", "target": "loan_pd_3", "type": "HAS_METRIC", "confidence": 0.98},
        {"source": "loan_pd_1", "target": "loan_pd_4", "type": "CALLS", "confidence": 0.99},
        {"source": "loan_pd_4", "target": "shared_2", "type": "CALLS", "confidence": 0.85},
        
        # Fraud detection relationships
        {"source": "fraud_1", "target": "fraud_2", "type": "USES_DATASET", "confidence": 0.96},
        {"source": "fraud_1", "target": "shared_1", "type": "USES_DATASET", "confidence": 0.88},
        {"source": "fraud_1", "target": "fraud_3", "type": "HAS_METRIC", "confidence": 0.97},
        {"source": "fraud_1", "target": "fraud_4", "type": "CALLS", "confidence": 0.99},
        {"source": "fraud_4", "target": "shared_2", "type": "CALLS", "confidence": 0.82},
        
        # Credit scoring relationships
        {"source": "credit_1", "target": "credit_2", "type": "USES_DATASET", "confidence": 0.94},
        {"source": "credit_1", "target": "shared_1", "type": "USES_DATASET", "confidence": 0.91},
        {"source": "credit_1", "target": "credit_3", "type": "HAS_METRIC", "confidence": 0.96},
        {"source": "credit_1", "target": "credit_4", "type": "CALLS", "confidence": 0.98},
        {"source": "credit_4", "target": "shared_2", "type": "CALLS", "confidence": 0.87},
        
        # Cross-model relationships
        {"source": "credit_1", "target": "loan_pd_1", "type": "COMPARES_TO", "confidence": 0.75},
        {"source": "fraud_1", "target": "loan_pd_1", "type": "DEPENDS_ON", "confidence": 0.65},
    ]
    
    # Add all edges
    for edge in edges:
        graph.add_edge(edge["source"], edge["target"], 
                      type=edge["type"], confidence=edge["confidence"])
    
    # Create embeddings (random for demo)
    np.random.seed(42)
    embeddings = np.random.randn(len(nodes), 1536).astype(np.float32)
    faiss.normalize_L2(embeddings)
    
    # Create FAISS index
    index = faiss.IndexFlatIP(1536)
    index.add(embeddings)
    
    # Create node mapping
    node_mapping = {i: node["id"] for i, node in enumerate(nodes)}
    
    # Create metadata
    metadata = {
        "version": "1.0.0-demo",
        "created_at": datetime.now().isoformat(),
        "processed_files": {},
        "node_count": len(nodes),
        "edge_count": len(edges),
        "demo_mode": True
    }
    
    # Save everything
    with open("demo_bundle/graph.pkl", "wb") as f:
        pickle.dump(graph, f)
    
    faiss.write_index(index, "demo_bundle/faiss.index")
    
    with open("demo_bundle/node_mapping.pkl", "wb") as f:
        pickle.dump(node_mapping, f)
    
    with open("demo_bundle/meta.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Demo bundle created: {len(nodes)} nodes, {len(edges)} edges")


if __name__ == "__main__":
    create_demo_bundle()