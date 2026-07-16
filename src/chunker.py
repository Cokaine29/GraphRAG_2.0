"""
chunker.py - sentence-aware chunker using spaCy + tiktoken.

Splits documents into ~600-token chunks that respect sentence boundaries.
Each chunk overlaps by N sentences with the next chunk for context continuity.

Usage:
    from src.chunker import chunk_document, chunk_all_documents
    chunks = chunk_document(text, doc_id="paper_001")
"""

import json
import hashlib
import logging
from pathlib import Path
from typing import List, Dict

import tiktoken

from src.config import cfg, ROOT

logger = logging.getLogger(__name__)

__all__ = ["count_tokens", "split_into_sentences", "chunk_document", "chunk_all_documents", "load_all_chunks"]

# -- Lazy loading for spaCy --------------------------------------
_nlp = None

def _get_nlp():
    global _nlp
    if _nlp is None:
        logger.info("Loading spaCy model...")
        import spacy
        _nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
        _nlp.add_pipe("sentencizer")   # lightweight sentence splitter
    return _nlp

TOKENIZER = tiktoken.get_encoding("cl100k_base")

CHUNK_SIZE       = cfg.get("chunking", {}).get("chunk_size", 600)
OVERLAP_SENTS    = cfg.get("chunking", {}).get("overlap_sentences", 2)
MIN_CHUNK_TOKENS = cfg.get("chunking", {}).get("min_chunk_tokens", 50)


# -- Core chunking logic ---------------------------------------

def count_tokens(text: str) -> int:
    return len(TOKENIZER.encode(text))


def split_into_sentences(text: str) -> List[str]:
    """Use spaCy sentencizer to split text into sentences."""
    nlp = _get_nlp()
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    return sentences


def chunk_document(text: str, doc_id: str) -> List[Dict]:
    """
    Split a document into overlapping sentence-aware chunks.

    Returns a list of chunk dicts:
    {
        chunk_id:    str   (hash-based unique id),
        doc_id:      str,
        chunk_index: int,
        text:        str,
        token_count: int,
        sentences:   int  (number of sentences in chunk)
    }
    """
    sentences = split_into_sentences(text)
    if not sentences:
        return []

    chunks = []
    chunk_index = 0
    i = 0

    while i < len(sentences):
        current_sentences = []
        current_tokens = 0

        # Pack sentences until we hit the token limit
        j = i
        while j < len(sentences):
            sent = sentences[j]
            sent_tokens = count_tokens(sent)

            if current_tokens + sent_tokens > CHUNK_SIZE and current_sentences:
                break  # chunk is full

            current_sentences.append(sent)
            current_tokens += sent_tokens
            j += 1

        chunk_text = " ".join(current_sentences).strip()

        # Skip tiny chunks (likely noise)
        if count_tokens(chunk_text) >= MIN_CHUNK_TOKENS:
            chunk_id = _make_chunk_id(doc_id, chunk_index, chunk_text)
            chunks.append({
                "chunk_id":    chunk_id,
                "doc_id":      doc_id,
                "chunk_index": chunk_index,
                "text":        chunk_text,
                "token_count": current_tokens,
                "sentences":   len(current_sentences),
            })
            chunk_index += 1

        # Overlap: step back OVERLAP_SENTS sentences
        i = j - OVERLAP_SENTS if j - OVERLAP_SENTS > i else j

        # Safety: always advance at least 1
        if i >= j:
            i = j

    return chunks


def _make_chunk_id(doc_id: str, index: int, text: str) -> str:
    h = hashlib.md5(text.encode()).hexdigest()[:8]
    return f"{doc_id}_chunk{index:04d}_{h}"


# -- Batch processing ------------------------------------------

def chunk_all_documents(documents: List[Dict]) -> List[Dict]:
    """
    documents: list of {"doc_id": str, "text": str}
    Returns: flat list of all chunks across all documents.
    Also saves each doc's chunks to data/processed/{doc_id}_chunks.json
    """
    processed_dir = ROOT / "data" / "processed"
    all_chunks = []

    for doc in documents:
        doc_id = doc["doc_id"]
        text   = doc["text"]

        output_path = processed_dir / f"{doc_id}_chunks.json"

        # Load from cache if already chunked
        if output_path.exists():
            logger.info(f"{doc_id} - loading from cache")
            with open(output_path) as f:
                chunks = json.load(f)
        else:
            logger.info(f"{doc_id} - chunking ({count_tokens(text)} tokens)...")
            chunks = chunk_document(text, doc_id)
            with open(output_path, "w") as f:
                json.dump(chunks, f, indent=2)
            logger.info(f"{doc_id} - {len(chunks)} chunks created")

        all_chunks.extend(chunks)

    logger.info(f"Total chunks across all docs: {len(all_chunks)}")
    return all_chunks


def load_all_chunks() -> List[Dict]:
    """Load all pre-computed chunks from data/processed/."""
    processed_dir = ROOT / "data" / "processed"
    all_chunks = []
    for path in sorted(processed_dir.glob("*_chunks.json")):
        with open(path) as f:
            all_chunks.extend(json.load(f))
    return all_chunks

