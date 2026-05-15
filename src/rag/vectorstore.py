import os
import glob
from pathlib import Path
from typing import List, Dict, Any
import chromadb
from chromadb.utils import embedding_functions
import logging

from src.config import CHROMA_DB_DIR, DOCS_DIR

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Use ChromaDB's built-in lightweight local ONNX embedding function (MiniLM-L6-v2)
# This does not require OpenAI keys or large PyTorch downloads.
embedding_fn = embedding_functions.ONNXMiniLM_L6_V2()

def get_chroma_client():
    """Initializes and returns a persistent ChromaDB client."""
    os.makedirs(CHROMA_DB_DIR, exist_ok=True)
    return chromadb.PersistentClient(path=CHROMA_DB_DIR)

def get_or_create_collection():
    """Fetches or creates the vector database collection."""
    client = get_chroma_client()
    return client.get_or_create_collection(
        name="retention_playbooks",
        embedding_function=embedding_fn
    )

def chunk_text(text: str, chunk_size: int = 800, chunk_overlap: int = 150) -> List[str]:
    """Splits a string of text into smaller overlapping chunks."""
    words = text.split()
    chunks = []
    
    # Simple word-based chunking to preserve readability
    step = chunk_size - chunk_overlap
    for i in range(0, len(words), step):
        chunk_words = words[i:i + chunk_size]
        chunks.append(" ".join(chunk_words))
        if i + chunk_size >= len(words):
            break
            
    return chunks

def ingest_documents():
    """Reads documents from DOCS_DIR, chunks them, and adds them to ChromaDB."""
    collection = get_or_create_collection()
    
    # Check what files are in data/docs/
    doc_files = glob.glob(os.path.join(DOCS_DIR, "*.md")) + glob.glob(os.path.join(DOCS_DIR, "*.txt"))
    
    if not doc_files:
        logger.warning(f"No documents found for ingestion in {DOCS_DIR}")
        return
        
    logger.info(f"Found {len(doc_files)} documents to ingest.")
    
    all_chunks = []
    all_metadatas = []
    all_ids = []
    
    for doc_path in doc_files:
        filename = os.path.basename(doc_path)
        logger.info(f"Ingesting file: {filename}")
        
        try:
            with open(doc_path, "r", encoding="utf-8") as f:
                content = f.read()
                
            chunks = chunk_text(content)
            logger.info(f"Split {filename} into {len(chunks)} chunks.")
            
            for idx, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                all_metadatas.append({
                    "source": filename,
                    "chunk_index": idx
                })
                all_ids.append(f"{filename}_chunk_{idx}")
                
        except Exception as e:
            logger.error(f"Failed to read/chunk {doc_path}: {e}")
            
    if all_chunks:
        # Add to Chroma (overwrites duplicates automatically by ID)
        collection.upsert(
            documents=all_chunks,
            metadatas=all_metadatas,
            ids=all_ids
        )
        logger.info(f"Successfully ingested {len(all_chunks)} chunks into ChromaDB.")
    else:
        logger.warning("No chunks to insert.")

def query_vectorstore(query_text: str, n_results: int = 3) -> List[Dict[str, Any]]:
    """Queries ChromaDB for the most semantically relevant chunks."""
    collection = get_or_create_collection()
    
    # If the collection is empty, trigger a quick ingestion
    if collection.count() == 0:
        logger.info("ChromaDB collection is empty. Ingesting documents first...")
        ingest_documents()
        
    results = collection.query(
        query_texts=[query_text],
        n_results=n_results
    )
    
    formatted_results = []
    if results and "documents" in results and results["documents"]:
        documents = results["documents"][0]
        metadatas = results["metadatas"][0] if "metadatas" in results else [{}] * len(documents)
        distances = results["distances"][0] if "distances" in results else [0.0] * len(documents)
        
        for doc, meta, dist in zip(documents, metadatas, distances):
            formatted_results.append({
                "content": doc,
                "metadata": meta,
                "score": float(dist) # In Chroma, lower distance = higher similarity
            })
            
    return formatted_results
