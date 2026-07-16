"""
CLI: Run GraphRAG vs Vanilla RAG evaluation.

Usage: 
    python scripts/run_evaluation.py --questions evaluation/questions.json
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.evaluation import Evaluator
from src.llm_client import LLMClient
from src.vector_store import VectorStore
from src.graph_query import GraphQueryEngine
from src.community_detection import CommunityDetector
from src.community_summarizer import CommunitySummarizer
from src.config import GRAPH_PATH

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_questions():
    """Fallback questions if no file provided."""
    return [
        {"id": "G01", "question": "What are the primary applications and ethical challenges of integrating IoT and Blockchain in supply chain management?", "type": "GLOBAL"},
        {"id": "L01", "question": "How does the integration of IoT and Blockchain improve traceability and predictive maintenance in logistics?", "type": "LOCAL"}
    ]

def main():
    parser = argparse.ArgumentParser(description="Evaluate GraphRAG vs Vanilla RAG")
    parser.add_argument("--questions", type=str, default=None,
                        help="Path to JSON file containing list of questions")
    parser.add_argument("--output", type=str, default="evaluation/results.json",
                        help="Path to save evaluation results")
    
    args = parser.parse_args()
    
    if args.questions and os.path.exists(args.questions):
        with open(args.questions, 'r') as f:
            questions = json.load(f)
    else:
        logger.info("Using default test questions")
        questions = get_questions()
        
    logger.info("Initializing engines for evaluation...")
    
    # 1. Vector Store (Vanilla)
    vs = VectorStore()
    
    # 2. Graph Engine (GraphRAG)
    try:
        cd = CommunityDetector()
        cd.load_graph(path=Path(GRAPH_PATH))
        cd.load_partitions()
        cs = CommunitySummarizer(detector=cd)
        # Ensure summaries exist
        cs.summarize_all(resolution=1.0)
        graph_engine = GraphQueryEngine(summarizer=cs)
    except Exception as e:
        logger.error(f"Failed to initialize Graph Engine. Did you build the graph? Error: {e}")
        sys.exit(1)
        
    logger.info("Starting Evaluation...")
    evaluator = Evaluator(vector_store=vs, graph_engine=graph_engine)
    results = evaluator.run(questions)
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
        
    logger.info(f"Evaluation complete. Results saved to {output_path}")
    
    print("\n--- WIN RATES ---")
    for metric, rates in results.get("win_rates", {}).items():
        print(f"{metric.upper()}: VanillaRAG: {rates.get('vanilla_rag%')}% | GraphRAG: {rates.get('graphrag%')}% | Tie: {rates.get('tie%')}%")

if __name__ == "__main__":
    main()
