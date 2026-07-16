"""
Unit tests for the chunker module.
"""

import pytest
from src.chunker import chunk_document, count_tokens, split_into_sentences

def test_count_tokens():
    text = "This is a simple test sentence for token counting."
    count = count_tokens(text)
    assert count > 0
    assert count < 20  # Should be around 9-10 tokens

def test_split_into_sentences():
    text = "Hello world! This is a test. How does it work?"
    sentences = split_into_sentences(text)
    assert len(sentences) == 3
    assert sentences[0].strip() == "Hello world!"
    assert sentences[1].strip() == "This is a test."

def test_chunk_document():
    text = "Sentence 1. Sentence 2. Sentence 3. Sentence 4. Sentence 5."
    # Temporarily override config for testing
    from src.config import cfg
    original_size = cfg['chunking']['chunk_size']
    original_overlap = cfg['chunking']['overlap_sentences']
    
    cfg['chunking']['chunk_size'] = 10
    cfg['chunking']['overlap_sentences'] = 1
    
    try:
        chunks = chunk_document(text, doc_id="test_doc")
        assert len(chunks) >= 1
        
        # Verify chunk structure
        assert "chunk_id" in chunks[0]
        assert chunks[0]["doc_id"] == "test_doc"
        assert "text" in chunks[0]
        assert "token_count" in chunks[0]
    finally:
        # Restore config
        cfg['chunking']['chunk_size'] = original_size
        cfg['chunking']['overlap_sentences'] = original_overlap
