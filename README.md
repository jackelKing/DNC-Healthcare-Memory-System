# DNC-Based Personal Causal Memory System for Healthcare

## Setup
```
pip install -r requirements.txt
ollama pull nomic-embed-text
ollama pull llama3
```

## Run
```
streamlit run ui/app.py
```

## Example Queries
- "What caused the kidney disease?"
- "Why did the patient develop neuropathy?"
- "What triggered the heart failure?"
- "What caused the cerebral thrombosis?"
  
## Architecture
```
project_root/
├── core/
│   ├── memory_matrix.py        # N×W DNC memory matrix with cosine addressing and temporal linkage
│   └── dnc_controller.py       # Embeds events, writes to memory, performs hybrid queries (70/15/15)
├── reasoning/
│   └── causal_graph.py         # NetworkX DAG, multi-hop traversal, PageRank scoring
├── retrieval/
│   └── engine.py               # Semantic search + temporal filter + causal traversal
├── explainability/
│   └── explainer.py            # JSON output + rule chains + LLM narrative
├── ollama_client/
│   └── client.py               # Retry, caching, n-gram fallback (offline support)
└── ui/
    └── app.py                 # Streamlit UI (Query, Timeline, Graph, Heatmap, Debug)
```
