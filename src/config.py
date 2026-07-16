"""
config.py - loads config.yaml and .env once, provides a single cfg object.
Usage: from src.config import cfg
"""

import os
import sys
import yaml
from pathlib import Path
from dotenv import load_dotenv

__all__ = ["cfg", "get", "ROOT", "CACHE_DIR", "SUMMARIES_DIR", "GRAPH_PATH", "CHROMA_DIR", "EVAL_DIR", "GEMINI_API_KEY", "GROQ_API_KEY", "CLAUDE_API_KEY", "OLLAMA_URL"]

# -- Locate project root (one level above src/) ----------------
ROOT = Path(__file__).resolve().parent.parent

load_dotenv(ROOT / ".env")

def _load_config() -> dict:
    config_path = ROOT / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)

cfg = _load_config()

# -- Convenience accessors -------------------------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GROQ_API_KEY   = os.getenv("GROQ_API_KEY", "")
CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY", "")
OLLAMA_URL     = os.getenv("OLLAMA_URL", "http://localhost:11434")

# Resolved absolute paths
CACHE_DIR      = ROOT / cfg["extraction"]["cache_dir"]
SUMMARIES_DIR  = ROOT / cfg["community"]["summary_cache_dir"]
GRAPH_PATH     = ROOT / cfg["graph"]["output_path"]
CHROMA_DIR     = ROOT / cfg["chromadb"]["persist_dir"]
EVAL_DIR       = ROOT / cfg["evaluation"]["output_dir"]

# Auto-create dirs on import
for d in [CACHE_DIR, SUMMARIES_DIR, GRAPH_PATH.parent,
          CHROMA_DIR, EVAL_DIR, ROOT / "data" / "raw",
          ROOT / "data" / "processed"]:
    Path(d).mkdir(parents=True, exist_ok=True)

def get(section: str, key: str, default=None):
    """cfg.get('retrieval', 'top_k', 5)"""
    return cfg.get(section, {}).get(key, default)

def validate_config():
    """Validates that necessary API keys are present based on config."""
    errors = []
    extract_prov = cfg.get("llm", {}).get("extraction_provider", "")
    gen_prov = cfg.get("llm", {}).get("generation_provider", "")
    
    providers = {extract_prov, gen_prov}
    
    if "gemini" in providers and not GEMINI_API_KEY:
        errors.append("GEMINI_API_KEY is missing but Gemini is configured as a provider.")
    if "groq" in providers and not GROQ_API_KEY:
        errors.append("GROQ_API_KEY is missing but Groq is configured as a provider.")
    if "claude" in providers and not CLAUDE_API_KEY:
        errors.append("CLAUDE_API_KEY is missing but Claude is configured as a provider.")
        
    if errors:
        print("\n".join(errors))
        print("Please check your .env file.")

validate_config()

