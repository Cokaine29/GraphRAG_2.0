"""
Unit tests for the entity_resolver module.
"""

import pytest
from src.entity_resolver import EntityResolver

def test_normalize():
    resolver = EntityResolver()
    
    assert resolver.norm("The Transformer") == "transformer"
    assert resolver.norm("A Model") == "model"
    assert resolver.norm("self-attention") == "self attention"
    assert resolver.norm("BERT_model") == "bert model"
    assert resolver.norm("  Whitespace   ") == "whitespace"

def test_is_trivial():
    resolver = EntityResolver()
    
    # Stop words / generic terms
    assert resolver.is_trivial("the") == True
    assert resolver.is_trivial("method") == True
    assert resolver.is_trivial("approach") == True
    assert resolver.is_trivial("paper") == True
    
    # Too short
    assert resolver.is_trivial("a") == True
    assert resolver.is_trivial("xy") == True
    
    # Valid entities
    assert resolver.is_trivial("Transformer") == False
    assert resolver.is_trivial("Self-Attention") == False
    assert resolver.is_trivial("BERT") == False
