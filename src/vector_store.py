"""
vector_store.py — ChromaDB wrapper for chunk embeddings.

Handles:
- Building the vector index from chunks
- Similarity search at query time
- Persistent storage to disk (survives Colab restarts if Drive is mounted)

Usage:
    from src.vector_store import VectorStore
    vs = VectorStore()
    vs.build_index(chunks)                    # one-time, slow
    results = vs.query("attention mechanism") # fast
"""

import json
import time
import logging
from typing import List, Dict, Optional, Any

import chromadb
from chromadb.config import Settings

from src.config import cfg, CHROMA_DIR
from src.llm_client import LLMClient

logger = logging.getLogger(__name__)

__all__ = ["VectorStore"]

class VectorStore:
    def __init__(self):
        self.collection_name = cfg.get("chromadb", {}).get("collection_name", "graphrag_chunks")
        self.top_k = cfg.get("retrieval", {}).get("top_k", 5)

        # Persistent client — survives restarts
        self._client = chromadb.PersistentClient(
            path=str(CHROMA_DIR),
            settings=Settings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self._llm = LLMClient(purpose="generation")
        logger.info(f"Collection '{self.collection_name}' — {self._collection.count()} docs loaded")

    # ── Index building ────────────────────────────────────────

    def build_index(self, chunks: List[Dict[str, Any]], batch_size: int = 50) -> None:
        """
        Embed all chunks and insert into ChromaDB.
        Skips chunks already in the collection (safe to re-run).
        """
        existing_ids = set(self._collection.get()["ids"])
        new_chunks = [c for c in chunks if c["chunk_id"] not in existing_ids]

        if not new_chunks:
            logger.info(f"All {len(chunks)} chunks already indexed. Skipping.")
            return

        logger.info(f"Indexing {len(new_chunks)} new chunks (batches of {batch_size})...")

        for i in range(0, len(new_chunks), batch_size):
            batch = new_chunks[i:i + batch_size]
            ids        = [c["chunk_id"] for c in batch]
            texts      = [c["text"] for c in batch]
            metadatas  = [{"doc_id": c["doc_id"], "chunk_index": c["chunk_index"],
                           "token_count": c["token_count"]} for c in batch]

            # Embed batch with delay to respect rate limits
            embeddings = []
            for text in texts:
                embeddings.append(self._llm.embed(text))
                time.sleep(0.3)

            self._collection.add(
                ids=ids,
                documents=texts,
                embeddings=embeddings,
                metadatas=metadatas,
            )
            logger.info(f"Indexed {min(i+batch_size, len(new_chunks))}/{len(new_chunks)} chunks")

        logger.info(f"Done. Total in collection: {self._collection.count()}")

    # ── Query ─────────────────────────────────────────────────

    def query(self, query_text: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Retrieve top-k most similar chunks for a query.

        Returns list of:
        {
            chunk_id:    str,
            doc_id:      str,
            text:        str,
            score:       float  (cosine similarity, higher = more similar)
        }
        """
        k = top_k or self.top_k
        query_embedding = self._llm.embed(query_text)

        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )

        output = []
        if results and results.get("ids") and results["ids"][0]:
            for i in range(len(results["ids"][0])):
                output.append({
                    "chunk_id": results["ids"][0][i],
                    "doc_id":   results["metadatas"][0][i]["doc_id"],
                    "text":     results["documents"][0][i],
                    "score":    round(1 - (results["distances"][0][i] or 0.0), 4),  # cosine: 1-distance
                })

        return output

    # ── Stats ─────────────────────────────────────────────────

    def stats(self) -> Dict[str, Any]:
        count = self._collection.count()
        return {"total_chunks": count, "collection": self.collection_name}

    def reset(self) -> None:
        """Delete and recreate the collection. Use with caution."""
        self._client.delete_collection(self.collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("Collection reset.")
