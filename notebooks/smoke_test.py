# GraphRAG — Person B smoke test
# Run this cell-by-cell to verify your setup before touching the main code

# ── Cell 1: Mount Drive (Colab only) ─────────────────────────
from google.colab import drive
drive.mount('/content/drive')

import sys, os
PROJECT_ROOT = '/content/drive/MyDrive/GraphRAG'
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)
print("Working dir:", os.getcwd())

# ── Cell 2: Install deps ──────────────────────────────────────
# !pip install -r requirements.txt -q
# !python -m spacy download en_core_web_sm -q

# ── Cell 3: Test config loads ─────────────────────────────────
from src.config import cfg, GEMINI_API_KEY
print("Config loaded OK")
print("Chunking config:", cfg["chunking"])
print("Gemini key set:", bool(GEMINI_API_KEY))

# ── Cell 4: Test chunker ──────────────────────────────────────
from src.chunker import chunk_document

sample_text = """
The transformer architecture was introduced in the paper "Attention is All You Need".
It relies entirely on self-attention mechanisms, dispensing with recurrence and convolutions.
The encoder maps an input sequence to a sequence of continuous representations.
Given these representations, the decoder generates an output sequence one element at a time.
Self-attention allows each position in the sequence to attend to all positions in the previous layer.
This is key to capturing long-range dependencies efficiently.
""" * 20

chunks = chunk_document(sample_text, doc_id="test_001")
print(f"\nChunker test: {len(chunks)} chunks produced")
for c in chunks[:2]:
    print(f"  [{c['chunk_index']}] {c['token_count']} tokens, {c['sentences']} sentences")
    print(f"  preview: {c['text'][:100]}...")

# ── Cell 5: Test LLM client ───────────────────────────────────
from src.llm_client import LLMClient

llm = LLMClient(purpose="generation")
response = llm.generate("In one sentence, what is a knowledge graph?")
print("\nLLM test:", response)

# ── Cell 6: Test router (needs LLM) ──────────────────────────
from src.router import QueryRouter

router = QueryRouter(llm=llm)
test_queries = [
    "What datasets were used in the BERT paper?",
    "What are the main themes across all papers?",
    "Explain attention and its role in the field",
]
print("\nRouter test:")
for q in test_queries:
    d = router.route(q)
    print(f"  '{q[:50]}...' → {d.route_type.value} ({d.confidence:.2f})")

# ── Cell 7: Test vector store (needs embeddings) ──────────────
from src.vector_store import VectorStore

vs = VectorStore()
print("\nVector store stats:", vs.stats())

# Index the test chunks
vs.build_index(chunks)
print("After indexing:", vs.stats())

# Query
results = vs.query("attention mechanism transformer")
print(f"\nQuery returned {len(results)} chunks")
for r in results[:2]:
    print(f"  score={r['score']} | {r['text'][:80]}...")

print("\n✓ All smoke tests passed!")
