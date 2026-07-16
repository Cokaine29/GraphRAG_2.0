"""
CLI: Chunk PDFs and build vector index.

Usage: 
    python scripts/ingest_documents.py --pdf-dir data/raw
"""

import os
import sys
import glob
import argparse
import logging
import fitz  # PyMuPDF
import json
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.chunker import chunk_document
from src.vector_store import VectorStore

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def process_pdf(pdf_path: str) -> str:
    """Extract text from a PDF file."""
    logger.info(f"Extracting text from {pdf_path}")
    doc = fitz.open(pdf_path)
    pages = ['\n'.join(line for line in pg.get_text().split('\n') if len(line.strip()) > 20) for pg in doc]
    text = '\n\n'.join(p for p in pages if p.strip())
    return text

def main():
    parser = argparse.ArgumentParser(description="Ingest PDFs into Vector Store")
    parser.add_argument("--pdf-dir", type=str, default="data/raw",
                        help="Directory containing raw PDF files")
    parser.add_argument("--out-dir", type=str, default="data/processed",
                        help="Directory to save the processed chunks")
    
    args = parser.parse_args()
    
    pdf_dir = Path(args.pdf_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    if not pdf_dir.exists():
        logger.error(f"PDF directory not found: {pdf_dir}")
        sys.exit(1)
        
    pdf_files = glob.glob(f"{pdf_dir}/*.pdf")
    if not pdf_files:
        logger.warning(f"No PDFs found in {pdf_dir}")
        sys.exit(0)
        
    all_chunks = []
    
    for pdf in pdf_files:
        doc_id = Path(pdf).stem
        text = process_pdf(pdf)
        logger.info(f"Chunking {doc_id}...")
        chunks = chunk_document(text, doc_id=doc_id)
        
        # Save chunks to disk
        out_path = out_dir / f"{doc_id}_chunks.json"
        with open(out_path, 'w') as f:
            json.dump(chunks, f, indent=2)
            
        all_chunks.extend(chunks)
        
    logger.info(f"Total chunks extracted: {len(all_chunks)}")
    logger.info("Initializing Vector Store and building index...")
    
    vs = VectorStore()
    vs.build_index(all_chunks)
    
    logger.info("Ingestion complete!")

if __name__ == "__main__":
    main()
