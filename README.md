# 🔬 NII RAG System

An intelligent **Retrieval-Augmented Generation (RAG)** system for the National Institute of Immunology that transforms institutional website data into a conversational AI assistant.

[![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/license-Academic-green.svg)](LICENSE)
[![Chroma](https://img.shields.io/badge/Vector%20DB-Chroma-orange.svg)](https://www.trychroma.com/)

## 🎯 Overview

The NII RAG System enables natural language queries about faculty, research, publications, and institutional information from the National Institute of Immunology. Built with modular architecture and advanced retrieval strategies.

<img width="1038" height="742" alt="image" src="https://github.com/user-attachments/assets/294660eb-5684-4553-a2bb-d1c2ce1e8f80" />


### ✨ Key Features

- **🧠 Intelligent Query Routing** - Automatically classifies queries into 9 specialized domains
- **👥 Smart Faculty Recognition** - Handles names, nicknames, and ambiguous references  
- **🔍 Multi-Strategy Retrieval** - Metadata filtering, semantic search, and cross-domain search
- **💬 Conversation Memory** - Context-aware responses with chat history
- **🚀 Performance Optimization** - Multi-level LRU caching for fast responses
- **🔒 Security Protection** - Built-in prompt injection and threat detection
- **📱 Dual Interface** - Both CLI and Streamlit web interfaces

## 📊 System Statistics

- **📄 1,520+ Documents** indexed across 9 collections
- **👨‍🎓 55 Faculty Members** with comprehensive profiles
- **📚 1,316 Research Publications** fully searchable
- **🏛️ 9 Specialized Domains** for precise information retrieval
- **⚡ 180ms** average response time (cached queries)

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Layer    │    │ Processing Layer│    │Application Layer│
│                 │    │                 │    │                 │
│ collections/    │───▶│ vector_indexer/ │───▶│   niibot-v4/    │
│ vectorstores/   │    │                 │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
     Raw JSON              Chunking &            RAG Queries &
     Vector DB             Embedding            Response Gen
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Virtual environment (recommended)
- Groq API key

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/nii-rag-system.git
   cd nii-rag-system
   ```

2. **Set up virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment**
   ```bash
   cp .env.example .env
   # Edit .env and add your GROQ_API_KEY
   ```

### Data Processing (First Time Setup)

1. **Process and index your data**
   ```bash
   cd vector_indexer
   python main.py --verbose
   ```

2. **Verify indexing (optional)**
   ```bash
   python main.py --audit
   ```

### Running the Application

#### CLI Interface
```bash
cd niibot-v4
python main.py
```

#### Web Interface
```bash
cd niibot-v4
streamlit run UI.py
```

## 📁 Project Structure

```
nii-rag-system/
├── 📄 requirements.txt              # Dependencies
├── 📄 .env                         # Environment variables
│
├── 📁 vector_indexer/              # Data processing pipeline
│   ├── 📄 main.py                 # Indexing CLI
│   ├── 📁 config/                 # Configuration
│   ├── 📁 core/                   # Core processing logic
│   ├── 📁 chunkers/               # Domain-specific chunkers
│   ├── 📁 utils/                  # Utility functions
│   └── 📁 auditing/               # Quality control
│
├── 📁 niibot-v4/                  # RAG application
│   ├── 📄 main.py                 # CLI interface
│   ├── 📄 UI.py                   # Streamlit web interface
│   ├── 📄 domain_router.py        # Query routing
│   ├── 📁 config/                 # Configuration & prompts
│   ├── 📁 core/                   # Core RAG logic
│   ├── 📁 chains/                 # RAG chain implementation
│   └── 📁 utils/                  # Utility functions
│
├── 📁 collections/                # Raw JSON data
│   ├── 📁 FacultyInfo/           # Faculty profiles
│   ├── 📁 Publications/          # Research papers
│   ├── 📁 Research/              # Research summaries
│   └── 📁 [7 more collections]/
│
└── 📁 vectorstores/              # Vector databases
    ├── 📁 faculty_info/
    ├── 📁 publications/
    └── 📁 [7 more collections]/
```

## 💻 Usage Examples

### Faculty Queries
```
"Tell me about Dr. Monica Sundd"
"What is Dr. Debasisa Mohanty's research focus?"
"Dr. Vineeta Bal's latest publications"
```

### Research Queries
```
"Faculty working on cancer immunotherapy"
"Dr. Nimesh Gupta's lab team members"
"Publications on computational biology"
```

### Institutional Queries
```
"Who is the current director of NII?"
"List all faculty members"
"What PhD programs does NII offer?"
```

### Contact Queries
```
"Dr. Sarika Gupta's email address"
"Technical staff contact information"
```

## 🔧 Configuration

### Environment Variables
```bash
# .env file
GROQ_API_KEY=your_groq_api_key_here
```

### Key Configuration Files
- `vector_indexer/config/settings.py` - Data processing configuration
- `niibot-v4/config/settings.py` - RAG system configuration
- `niibot-v4/config/prompts.py` - Domain-specific prompts
- `niibot-v4/config/faculty_data.py` - Faculty knowledge base

## 📊 Technology Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Vector Database** | Chroma DB | Semantic search and storage |
| **Embedding Model** | all-MiniLM-L6-v2 | 384-dim text embeddings |
| **Language Model** | Llama 3-8B-8192 | Response generation |
| **API Service** | Groq API | High-speed LLM inference |
| **Web Framework** | Streamlit | User interface |
| **Caching** | functools.lru_cache | Performance optimization |

## 🛠️ Development

### Adding New Data Sources

1. **Add JSON files** to appropriate `collections/` folder
2. **Create/modify chunker** in `vector_indexer/chunkers/`
3. **Re-index data**:
   ```bash
   cd vector_indexer
   python main.py --collection your_collection
   ```

### Extending Functionality

1. **New Domain**: Add to `COLLECTION_CONFIG` in settings
2. **New Chunker**: Inherit from `BaseChunker`
3. **Custom Prompts**: Add domain-specific prompts
4. **Update Router**: Modify domain classification logic

### Testing

```bash
# Check data quality
cd vector_indexer
python main.py --audit

# Test specific collection
python main.py --collection faculty_info --verbose

# Monitor cache performance
cd niibot-v4
python main.py
# In chat: type "cache_stats"
```

## 📈 Performance

- **First Query**: ~2.1 seconds (full processing)
- **Cached Query**: ~180ms (LRU cache hit)
- **Average Response**: 2-3 seconds for complex queries
- **Index Size**: 1,520+ documents across 9 collections
- **Memory Usage**: ~500MB with full cache

## 🔒 Security Features

- **Prompt Injection Protection** - Pattern-based threat detection
- **Input Sanitization** - Query validation and cleaning
- **Safe Fallbacks** - Graceful handling of malicious inputs
- **Source Attribution** - Transparent information sourcing

## 🔧 Common Issues & Solutions

**Missing Vector Stores**
```bash
cd vector_indexer
python main.py --verbose  # Re-index all data
```

**API Key Setup**
```bash
# Add to .env file:
GROQ_API_KEY=your_actual_key_here
```

**Directory Issues**
```bash
cd niibot-v4      # for running the app
cd vector_indexer # for data processing
```

## 📄 License

**Academic License** - This project is developed for educational and research purposes. 

The code implementation is available for academic use. The scraped NII data remains property of the National Institute of Immunology. Please use responsibly and respect data sources.

See [LICENSE](LICENSE) file for full details.

---

<div align="center">

**🔬 National Institute of Immunology RAG System**

**Academic Project - Intelligent Conversational AI for Institutional Knowledge**

**⭐ Built by [KAMRANKHANALWI](https://github.com/KAMRANKHANALWI)**

</div>
