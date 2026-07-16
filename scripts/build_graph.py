"""
CLI: Build knowledge graph from document chunks using LLM extraction.

Usage: 
    python scripts/build_graph.py --input data/processed/chunks.json
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import cfg, GRAPH_PATH
from src.llm_client import LLMClient
from src.graph_builder import GraphBuilder
import networkx as nx

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Build Knowledge Graph from Chunks")
    parser.add_argument("--input", type=str, default="data/processed/attention_chunks.json",
                        help="Path to the JSON file containing document chunks")
    parser.add_argument("--output", type=str, default=str(GRAPH_PATH),
                        help="Path to save the resulting GML graph")
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        sys.exit(1)
        
    logger.info(f"Loading chunks from {input_path}")
    with open(input_path, 'r') as f:
        chunks = json.load(f)
        
    logger.info(f"Initializing GraphBuilder...")
    llm = LLMClient(purpose="extraction")
    builder = GraphBuilder(llm=llm)
    
    logger.info(f"Building graph from {len(chunks)} chunks. This may take a while...")
    graph = builder.build(chunks)
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    nx.write_gml(graph, str(output_path))
    logger.info(f"Graph successfully saved to {output_path}")

if __name__ == "__main__":
    main()
