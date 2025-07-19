# 🧠 Hybrid Agentic RAG System with Memory

A sophisticated knowledge counseling system that combines Graph-based reasoning with Retrieval-Augmented Generation (RAG) using a 4-agent architecture for comprehensive document analysis and question answering.

## ✨ Key Features

### 🤖 **4-Agent Reasoning System**
- **Plan**: Intelligent query planning and strategy
- **Thought**: Reasoning about graph connections and content gaps  
- **Action**: Hybrid graph + RAG retrieval execution
- **Observation**: Analysis and synthesis of findings

### 🧠 **Conversation Memory**
- **Context Retention**: Maintains conversation history across sessions
- **Follow-up Questions**: Natural "Tell me more" and "What about..." queries
- **Entity Tracking**: Remembers figures, tables, authors, models, and datasets
- **Smart Enhancement**: Automatically improves vague queries with context

### 🔍 **Hybrid Retrieval System**
- **Knowledge Graph**: 47 nodes, 1,081 edges with entity relationships
- **RAG Index**: 6,923 semantic chunks with FAISS vector search
- **Multi-Modal Search**: Semantic + keyword + hybrid ranking
- **Cross-Document Analysis**: Relationships across different models

### 🎨 **Professional UI**
- **Clean Interface**: Claude-style design with minimalist thinking display
- **Model Selection**: Filter and focus on specific knowledge domains
- **Real-time Feedback**: CSS-animated thinking process visualization
- **Memory Controls**: Conversation history and entity tracking

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- OpenAI API key
- 4GB+ RAM

### Installation
```bash
git clone https://github.com/shriyavallabh/hybrid-agentic-rag.git
cd hybrid-agentic-rag
pip install -r requirements.txt
echo "OPENAI_API_KEY=your_api_key_here" > .env
```

### Run the Application
```bash
streamlit run app.py
```

## 📁 Repository Structure

```
hybrid-agentic-rag/
├── 📊 CORE APPLICATION
│   ├── app.py                          # Main Streamlit application
│   ├── requirements.txt                # Python dependencies
│   └── core/                          # Core system modules
│       ├── conversation_memory.py      # Memory system
│       ├── cross_model_analyzer.py     # Cross-model analysis
│       ├── hybrid_agent_runner.py      # 4-agent reasoning
│       ├── hybrid_graph_rag.py         # Multi-modal retrieval
│       └── simple_graph_builder.py     # Graph construction
│
├── 📚 KNOWLEDGE BASE
│   └── knowledge_base/model_1/
│       └── graphrag_model_doc.pdf      # Documentation
│
├── 🧠 KNOWLEDGE GRAPH
│   └── enhanced_kg/
│       ├── enhanced_graph.pkl          # 47 nodes, 1,081 edges
│       └── metadata.json              # Graph metadata
│
└── 🔍 RAG INDEX
    └── rag_index/
        ├── chunks.pkl                  # 6,923 semantic chunks
        ├── faiss.index                 # Vector search index
        └── metadata.json               # Index metadata
```

## 🎯 How It Works

### 1. **Query Processing**
- User asks question in natural language
- Memory system enhances follow-up questions with context
- Question appears immediately (no blank screen)

### 2. **4-Agent Reasoning**
- **Planning**: Analyzes query structure and requirements
- **Thinking**: Reasons about graph connections and content gaps
- **Action**: Searches knowledge graph and retrieves relevant chunks
- **Observation**: Synthesizes findings into comprehensive answer

### 3. **Hybrid Retrieval**
- **Graph Search**: Finds relevant entities and relationships
- **RAG Search**: Semantic similarity search through 6,923 chunks
- **Hybrid Ranking**: Combines multiple retrieval methods
- **Content Filtering**: Prioritizes academic content over test files

### 4. **Memory System**
- **Context Tracking**: Remembers conversation history
- **Entity Extraction**: Tracks figures, tables, authors, models
- **Follow-up Enhancement**: "Tell me more" → "Tell me more about Figure 2"
- **Session Management**: Maintains context across interactions

## 🎨 User Experience

### **Conversation Flow**
```
You: What does Figure 2 show?

Thinking... (CSS-animated display)
• Planning approach...
• Analyzing query structure...
• Searching knowledge base...
• Retrieving relevant information...
• Synthesizing response...
• Analysis complete

Knowledge Counselor: Figure 2 shows the head-to-head win rate 
percentages comparing GraphRAG with other RAG methods...

You: Tell me more about it
[System automatically understands "it" refers to Figure 2]

Knowledge Counselor: Figure 2 specifically demonstrates...
```

### **Professional Interface**
- **Left Panel**: Model selection, document counts, memory controls
- **Main Area**: Clean conversation display with thinking visualization
- **Memory Section**: Conversation history and entity tracking
- **No Clutter**: Minimalist design focused on content

## 🔧 Technical Features

### **Production-Ready**
- **Clean Codebase**: Only essential files (28 total)
- **Optimized Size**: 15MB (reduced from 200MB)
- **Error Handling**: Graceful degradation and error recovery
- **Logging**: Comprehensive system monitoring

### **Advanced Capabilities**
- **Cross-Model Analysis**: Compare different models and approaches
- **Figure/Table Queries**: Smart handling of visual content references
- **Author Queries**: Intelligent author information retrieval
- **Performance Metrics**: Detailed analysis of model capabilities

### **System Requirements**
- **Memory**: 4GB+ RAM recommended
- **Storage**: 15MB disk space
- **Network**: OpenAI API access required
- **Browser**: Modern web browser for Streamlit UI

## 📊 Performance

- **Response Time**: <1 second average
- **Knowledge Graph**: 47 nodes, 1,081 edges
- **RAG System**: 6,923 chunks, FAISS-optimized
- **Memory Usage**: Efficient conversation tracking
- **Accuracy**: High-quality answers with source citations

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **OpenAI**: GPT-4o for reasoning and embeddings
- **Streamlit**: Web application framework
- **FAISS**: Efficient similarity search
- **NetworkX**: Graph processing capabilities

---

**Ready for production use** with professional interface, conversation memory, and hybrid reasoning capabilities.