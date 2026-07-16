"""
Unit tests for the config module.
"""

import pytest
from pathlib import Path
from src.config import cfg, ROOT_DIR, DATA_DIR, OUTPUTS_DIR, GRAPH_PATH

def test_config_loaded():
    assert isinstance(cfg, dict)
    assert "llm" in cfg
    assert "chunking" in cfg
    assert "graph" in cfg
    assert "retrieval" in cfg

def test_directories_exist():
    # Because config.py automatically creates these, they should exist
    assert ROOT_DIR.exists()
    assert DATA_DIR.exists()
    assert OUTPUTS_DIR.exists()

def test_graph_path():
    assert GRAPH_PATH is not None
    assert str(GRAPH_PATH).endswith(".gml")
    
def test_default_values():
    # Check some sane defaults exist
    assert cfg.get("chunking", {}).get("chunk_size", 0) > 0
    assert cfg.get("retrieval", {}).get("top_k", 0) > 0
