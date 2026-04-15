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
core/memory_matrix.py   - N x W DNC matrix, cosine addressing, temporal linkage
core/dnc_controller.py  - embed events -> write memory -> hybrid query (70/15/15)
reasoning/causal_graph.py - NetworkX DAG, multi-hop traversal, PageRank scoring
retrieval/engine.py     - semantic search + temporal filter + causal traversal
explainability/explainer.py - JSON + rule chain + Ollama LLM narrative
ollama_client/client.py - retry, caching, n-gram fallback when offline
ui/app.py               - 5-tab Streamlit: query, timeline, graph, heatmap, debug
