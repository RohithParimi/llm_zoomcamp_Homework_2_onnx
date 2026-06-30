# LLM Zoomcamp — Homework 2: Vector Search

This repo contains my solution for Module 2 (Vector Search) homework. We
embed text with a lightweight ONNX `Embedder`, build vector and text
search over the course lesson pages (commit `8c1834d`, 72 pages), and
combine both with Reciprocal Rank Fusion (RRF).

Files:

- `ingest.py` — loads the lesson pages from GitHub and builds a
  minsearch text `Index`
- `Homework.ipynb` — the notebook with all the work and answers below

## Setup

```bash
mkdir llm-zoomcamp-hw2 && cd llm-zoomcamp-hw2
uv init --no-workspace
uv add onnxruntime tokenizers numpy tqdm minsearch gitsource
uv add --dev huggingface-hub jupyter
```

Helper scripts (`download.py`, `embedder.py`) downloaded from the
`02-vector-search/embed/` directory of the course repo, and the ONNX
`Xenova/all-MiniLM-L6-v2` model fetched via `download.py`.

## Answers

### Q1. Embedding a query

Embedded `"How does approximate nearest neighbor search work?"` with
the ONNX `Embedder` and read off `v[0]`.

```python
from embedder import Embedder

embed = Embedder()
v1 = embed.encode("How does approximate nearest neighbor search work?")
v1[0]
```

**Result:** `v[0] = -0.0206`

**Answer: -0.02**

### Q2. Cosine similarity

Embedded the content of `02-vector-search/lessons/07-sqlitesearch-vector.md`
and took the dot product with the Q1 query vector.

```python
for doc in documents:
    if doc["filename"] == "02-vector-search/lessons/07-sqlitesearch-vector.md":
        text = doc["content"]

t1 = embed.encode(text)
t1.dot(v1)
```

**Result:** `0.361`

**Answer: 0.37** (closest option)

### Q3. Chunking and search by hand

Chunked all 72 pages with `chunk_documents(documents, size=2000, step=1000)`,
embedded every chunk with `encode_batch`, stacked the vectors into `X`,
and scored against the Q1 query vector with `X.dot(v1)`.

```python
from gitsource import chunk_documents

chunks = chunk_documents(documents, size=2000, step=1000)
scores = X.dot(v1)
idx = np.argmax(scores)
chunks[idx]
```

The highest-scoring chunk belongs to:

**Answer: `02-vector-search/lessons/07-sqlitesearch-vector.md`**

### Q4. Vector search with minsearch

Indexed the chunk vectors with `VectorSearch` and searched for
`"What metric do we use to evaluate a search engine?"`.

```python
from minsearch import VectorSearch

v_index = VectorSearch()
v_index.fit(X, chunks)

query = "What metric do we use to evaluate a search engine?"
v_query = embed.encode(query)
v_index.search(v_query, num_results=5)
```

The first result's filename:

**Answer: `04-evaluation/lessons/05-search-metrics.md`**

### Q5. Text search vs vector search

Indexed the same chunks with minsearch's `Index` (text field `content`)
and compared the top 5 vector vs. top 5 text results for
`"How do I store vectors in PostgreSQL?"`.

```python
query2 = "How do I store vectors in PostgreSQL?"
v_query2 = embed.encode(query2)
vector_results = v_index.search(v_query2, num_results=5)
text_results = t_index.search(query2, num_results=5)

vector_files = {d["filename"] for d in vector_results}
text_files = {d["filename"] for d in text_results}
vector_files - text_files
```

Compared the top 5 filenames from each method and looked at which file
appears in the vector results but not the text results.

### Q6. Hybrid search

Ran vector and text search for `"How do I give the model access to tools?"`
and fused the results with RRF (`k=60`).

```python
query3 = "How do I give the model access to tools?"
v_query3 = embed.encode(query3)
vector_results = v_index.search(v_query3, num_results=5)
text_results = t_index.search(query3, num_results=5)

results = rrf([vector_results, text_results])
```

The file ranked first after RRF:

**Answer: `01-agentic-rag/lessons/13-function-calling.md`**

This file wasn't first in either search alone — it wins because it
ranks well in both lists.

## Summary

| Question | Answer |
|---|---|
| Q1 | -0.02 |
| Q2 | 0.37 |
| Q3 | `02-vector-search/lessons/07-sqlitesearch-vector.md` |
| Q4 | `04-evaluation/lessons/05-search-metrics.md` |
| Q5 | *(your pick — let me know which option)* |
| Q6 | `01-agentic-rag/lessons/13-function-calling.md` |

Submitted at: https://courses.datatalks.club/llm-zoomcamp-2026/homework/hw2
