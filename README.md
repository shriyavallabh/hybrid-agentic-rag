# Hybrid Agentic RAG System

A sophisticated knowledge counseling system that combines Graph-based reasoning with Retrieval-Augmented Generation (RAG) using a 4-agent architecture for comprehensive document analysis and question answering.

## 🚀 Features

### Core Architecture
- **Hybrid Graph-Guided RAG**: Combines knowledge graph structure with comprehensive content retrieval
- **4-Agent Reasoning System**: Plan → Thought → Action → Observation with self-reflection
- **Multi-Modal Retrieval**: Semantic similarity + keyword-based search + hybrid ranking
- **Cross-Model Analysis**: Specialized comparison capabilities across different models
- **Real-time Agent Thinking**: Claude-style progressive reasoning display

### User Interface
- **Clean, Professional Design**: Inspired by Claude's interface
- **Expandable Agent Sections**: See real-time thinking process
- **Model Selection**: Filter and focus on specific knowledge domains
- **Comprehensive Logging**: Detailed retrieval process visibility

## 🏗️ System Architecture

### 1. Enhanced Knowledge Graph Builder (`core/enhanced_graph_builder.py`)
- Extracts rich entities from PDFs, Python files, and notebooks
- Uses GPT-4o for intelligent entity and relationship extraction
- Builds comprehensive knowledge graphs with cross-document connections

### 2. Hybrid Graph-RAG Integration (`core/hybrid_graph_rag.py`)
- **HybridRetriever**: Combines graph structure with RAG content
- **Multi-Modal Search**: Semantic + keyword + hybrid ranking
- **Cross-Document Analysis**: Identifies relationships across models
- **Query Expansion**: Automatic synonym and related term expansion

### 3. 4-Agent Reasoning System (`core/hybrid_agent_runner.py`)
- **PLAN Agent**: Creates strategy for hybrid graph-RAG analysis
- **THOUGHT Agent**: Reasons about graph structure and content gaps
- **ACTION Agent**: Executes hybrid retrieval (graph + RAG)
- **OBSERVATION Agent**: Analyzes results and forms insights

### 4. Cross-Model Analyzer (`core/cross_model_analyzer.py`)
- Specialized system for comparing multiple models
- Multi-dimensional analysis (performance, methodology, datasets)
- Ecosystem-wide relationship detection

## 📚 Knowledge Base Structure

```
knowledge_base/
├── model_1/                    # GraphRAG v2.1 Documentation
│   ├── graphrag_model_doc.pdf
│   └── graphrag-main/         # Complete GraphRAG codebase
├── model_2/                   # Future models
└── ...
```

## 🔧 Technical Implementation

### Retrieval Process
1. **Graph Entity Discovery**: Traverse knowledge graph for relevant entities
2. **Semantic Similarity Search**: Vector-based content retrieval
3. **Keyword-Based Search**: Exact match and expanded term matching
4. **Hybrid Ranking**: Combine and boost results found by multiple methods
5. **Cross-Document Analysis**: Identify relationships across sources

### Multi-Modal Retrieval Example
```python
# Query: "context data construction"
# Semantic: Finds similar meaning chunks
# Keyword: Finds exact matches + synonyms (handle, build, create)
# Hybrid: Boosts chunks found by both methods
# Result: Higher precision and recall
```

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- OpenAI API key
- Streamlit
- Required packages (see requirements.txt)

### Installation
```bash
git clone https://github.com/shriyavallabh/hybrid-agentic-rag.git
cd hybrid-agentic-rag
pip install -r requirements.txt
```

### Environment Setup
```bash
# Create .env file
echo "OPENAI_API_KEY=your_api_key_here" > .env
```

### Running the Application
```bash
streamlit run app.py
```

## 🎯 Usage

### Basic Query
1. Select models from the sidebar
2. Enter your question in the chat input
3. Watch the 4-agent reasoning process
4. Get comprehensive answers with sources

### Advanced Features
- **Cross-Model Queries**: "Compare GraphRAG vs traditional RAG"
- **Implementation Details**: "How is context data construction handled?"
- **Performance Analysis**: "What datasets are used for evaluation?"

## 📊 System Components

### Core Files
- `app.py`: Main Streamlit application with Claude-style UI
- `core/hybrid_agent_runner.py`: 4-agent reasoning system
- `core/hybrid_graph_rag.py`: Multi-modal retrieval engine
- `core/enhanced_graph_builder.py`: Knowledge graph construction
- `core/cross_model_analyzer.py`: Multi-model comparison system

### Data Processing
- `rag_index/`: Semantic chunks and FAISS indices
- `enhanced_kg/`: Enhanced knowledge graph files
- `kg_bundle/`: Legacy knowledge graph (fallback)

## 🔍 Logging and Debugging

The system provides comprehensive logging for:
- **Query Processing**: User questions and model selection
- **Graph Traversal**: Entity discovery and scoring
- **Retrieval Process**: Semantic, keyword, and hybrid search
- **Agent Reasoning**: Detailed 4-agent thought process
- **Cross-Document Analysis**: Relationship detection

## 🏆 Key Innovations

1. **Hybrid Graph-Guided RAG**: Novel combination of graph structure with comprehensive content
2. **Multi-Modal Retrieval**: Addresses semantic similarity limitations
3. **4-Agent Architecture**: Structured reasoning with self-reflection
4. **Cross-Model Intelligence**: Specialized comparison capabilities
5. **Real-time Thinking Display**: Transparent reasoning process

## 🔬 Research Applications

- **Document Analysis**: Comprehensive understanding of technical documents
- **Cross-Model Comparison**: Systematic evaluation of different approaches
- **Knowledge Discovery**: Finding connections across large document collections
- **Implementation Research**: Understanding how systems work at code level

## 📈 Performance Features

- **Unlimited Token Budget**: No artificial usage limits
- **Efficient Caching**: Reduced API calls through intelligent caching
- **Parallel Processing**: Concurrent retrieval operations
- **Progressive Loading**: Real-time results display

## 🛠️ Configuration

### Model Configuration
- Add new models by placing documents in `knowledge_base/model_x/`
- System automatically detects and indexes new content
- Dynamic model selection in UI

### Retrieval Tuning
- Adjust semantic vs keyword balance in `hybrid_graph_rag.py`
- Modify synonym mappings for domain-specific terms
- Tune hybrid ranking weights for optimal results

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add comprehensive logging
5. Test with various query types
6. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🔗 Related Work

- **GraphRAG**: Microsoft's graph-based RAG system
- **RAG Systems**: Retrieval-Augmented Generation approaches
- **Multi-Agent Systems**: Collaborative AI reasoning
- **Knowledge Graphs**: Structured knowledge representation

## 📞 Support

For questions, issues, or contributions:
- Open an issue on GitHub
- Check the comprehensive logging for debugging
- Review the agent reasoning process for query insights

---

**Built with ❤️ for advancing knowledge discovery and AI reasoning**