from __future__ import annotations

import os
import uuid
import logging
import chromadb
import numpy as np

from typing import List, Any

logging.basicConfig(level=logging.NOTSET)

class VectorStore:
    def __init__(
        self,
        collection_names: str="pdf_documents",
        persist_directory: str="langchain_rag_pipeline/data/vector_store",
    ):
        self.collection_name = collection_names
        self.persist_directory = persist_directory
        self.client = None
        self.collection = None
        self._initialize_store()

    def _initialize_store(self) -> None:
        """Initialize chromadb client collection"""
        try:
            #create chromadb client
            os.makedirs(self.persist_directory, exist_ok=True)
            self.client = chromadb.PersistentClient(path=self.persist_directory)

            #get or create collection
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={'description': "PDF document embeddings for RAG"},
            )

            logging.info(f"Vector store initialized, collection: {self.collection_name}")
            logging.info(f"[INFO] Existing documents in collection: {self.collection.count()}")

        except Exception as e:
            logging.error(f"Initializing vector store: {e}")

    def add_documents(self, *, documents: List[Any], embeddings: np.ndarray) -> None:
        """Add documents and there embeddings to VectorDB"""

        if len(documents) != len(embeddings):
            raise ValueError("[ERROR] Number of documents and number of embeddings mismatch!")
        print(f"[INFO] Adding {len(documents)} documents to vectore store...")

        #data prep for chromaDB
        ids, metadatas, documents_texts, embeddings_list = [], [], [], []

        for i, (doc, embeddings) in enumerate(
            zip(documents, embeddings)
        ):
            #generate unique ID
            doc_id = f"doc_{uuid.uuid4().hex[:8]}_{i}"
            ids.append(doc_id)

            #prepare metadata
            metadata = dict(doc.metadata)
            metadata['doc_index'] = i
            metadata['content_length'] = len(doc.page_content)
            metadatas.append(metadata)

            #document content
            documents_texts.append(doc.page_content)

            #embedding
            embeddings_list.append(embeddings.tolist())

            #add to the collection
        try:
            self.collection.add(
                ids=ids,
                embeddings=embeddings_list,
                metadatas=metadatas,
                documents=documents_texts,
            )

            logging.info(f"Succesfully added {len(documents)} documents to the vector store")
            logging.info(f"Total documents in collection: {self.collection.count()}")

        except Exception as e:
            logging.error(f"Error adding documents to the vector store {e}")
