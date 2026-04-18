from __future__ import annotations

import numpy as np

from typing import List, Any
from src.document_loader import DocsLoader
from src.embedding import EmbeddingManager
from src.vector_store import VectorStore
from src.rag_agent import RagAgent


def ingest_data(*, dir_path: str) -> None:
    """Ingest PDF files from directory to Vector Store"""
    doc_chunks, doc_chunks_texts = DocsLoader(path=dir_path).load_and_chunk()

    #generate embeddings for chunks
    gen_embeddings: np.ndarray = EmbeddingManager().generate_embeddings(texts=doc_chunks_texts)
    
    vector_store_interface = VectorStore()

    #add embeddings to vector store
    vector_store_interface.add_documents(
        documents=doc_chunks,
        embeddings=gen_embeddings,
    )

if __name__ == '__main__':
    dir_path = "langchain_rag_pipeline/docs/datasheets"
    
    #trigger the data ingestion pipeline
    ingest_data(dir_path=dir_path)
    
    #RAG Agent
    agent = RagAgent(query="Describe the pins of MS62256A")
    print(
        "[INFO] Final Agent Response",
        agent.agent_invoke(),
        sep="\n"*2,
    )
