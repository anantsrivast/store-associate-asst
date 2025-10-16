# Store Associate AI Assistant

A production-ready demonstration of AI memory systems using **LangMem**, **LangGraph**, and **MongoDB**.

## 🎯 What This Demonstrates

This project showcases different types of memory in AI agents:

- **Short-term Memory**: Current conversation context (LangGraph checkpointer)
- **Long-term Memory**: Customer profiles and preferences (MongoDB Store)
- **Episodic Memory**: Past interaction summaries with semantic search
- **Semantic Memory**: Extracted facts and preferences
- **Consolidated Insights**: Behavioral patterns learned over time

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- MongoDB Atlas account (free tier works)
- OpenAI or Anthropic API key

### 1. Setup Environment
```bash
# Clone and open in Codespaces (or locally)
git clone <your-repo-url>
cd store-associate-assistant

# Copy environment template
cp .env.example .env

# Edit .env with your credentials
# - MONGODB_URI: Your MongoDB Atlas connection string
# - OPENAI_API_KEY or ANTHROPIC_API_KEY: Your API key
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Setup MongoDB
```bash
# Create collections and indexes
python scripts/setup_mongodb.py

# IMPORTANT: Follow the printed instructions to create
# the Atlas Vector Search index manually
```

### 4. Seed Database
```bash
# Generate synthetic data
python -m src.data.seed_database
```

### 5. Run Application
```bash
# Start Streamlit UI
streamlit run src/ui/streamlit_app.py
```

## 📚 Documentation

Detailed documentation available in `docs/`:

- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) - Complete system architecture and design
- [`docs/API.md`](docs/API.md) - API reference
- [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md) - Production deployment guide

## 🎮 Using the Application

### Chat Interface

1. Select a customer from the sidebar
2. Start chatting as a store associate
3. The agent will:
   - Remember customer preferences
   - Reference past interactions
   - Make personalized recommendations
   - Extract and store new memories

### Memory Visualization

The sidebar shows:
- **Preferences**: Stored facts about the customer
- **Episodes**: Summaries of past conversations
- **Insights**: Discovered behavioral patterns

### Session Management

- **New Session**: Start a fresh conversation
- **End Session**: Close conversation and create episode summary

## 🔧 Advanced Usage

### Manual Consolidation

Run memory consolidation manually:
```bash
# Consolidate for all customers
python scripts/run_consolidation.py

# Consolidate for specific customer
python scripts/run_consolidation.py customer_0001
```

### Clear Data

Reset all data:
```bash
python scripts/clear_data.py
```

### Jupyter Notebooks

Explore memory concepts interactively:
```bash
jupyter notebook
# Open notebooks/01_setup_and_explore.ipynb
```

## 📂 Project Structure
```
store-associate-assistant/
├── src/
│   ├── agent/          # LangGraph agent and nodes
│   ├── data/           # Synthetic data generation
│   ├── database/       # MongoDB client and schemas
│   ├── memory/         # LangMem managers and consolidation
│   └── ui/             # Streamlit interface
├── scripts/            # Utility scripts
├── notebooks/          # Interactive tutorials
├── docs/               # Documentation
└── tests/              # Tests
```

## 🧠 How Memory Works

### Flow Diagram
```
User Message
    ↓
[Check if summarization needed]
    ↓
[Retrieve relevant memories]
    ↓
[Agent processes with memory context]
    ↓
[Extract new memories (real-time)]
    ↓
[If session ends: Create episode summary]
    ↓
[Background: Consolidate into patterns]
```

### Memory Types

| Type | Storage | Lifespan | Purpose |
|------|---------|----------|---------|
| Short-term | Checkpoints | Session | Current conversation |
| Episodic | MongoDB Store | 90 days | Past interactions |
| Semantic | MongoDB Store | Indefinite | Facts & preferences |
| Consolidated | MongoDB Store | Indefinite | Learned patterns |

## 🔐 Environment Variables

Required variables in `.env`:
```bash
# MongoDB
MONGODB_URI=mongodb+srv://...
MONGODB_DB_NAME=store_assistant

# LLM (choose one or both)
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...

# Model Selection
LLM_MODEL=anthropic:claude-3-5-sonnet-latest
EMBEDDING_MODEL=openai:text-embedding-3-small

# Memory Settings (optional)
MEMORY_TTL_SECONDS=7776000  # 90 days
SUMMARIZATION_TOKEN_THRESHOLD=2000
```

## 🧪 Testing

Run tests:
```bash
pytest tests/
```

## 🤝 Contributing

See `CONTRIBUTING.md` for guidelines.
