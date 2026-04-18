from __future__ import annotations

import logging

from typing import List, Dict, Any

logging.basicConfig(level=logging.NOTSET)


class RagReteiver:
    """
    Handles query based retreival from the vector store, its a interface 
    to get context from Vector store, based on user query
    """

    def __init__(self, *, vector_store: VectorStore, embedding_manager: EmbeddingManager):
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager

    def retreive(
        self,
        *,
        query: str,
        top_k: int=5,
        score_threashold: float=-.3,
    ) -> List[Dict[str, Any]]:

        """Return relevant docs for a query"""
        # query -> the query to serach
        # top_k -> number of top results to return
        # score_threashold -> minimum simililarity score threashold

        #returns list of documnets containing retrived docs and metadata
        logging.info(f"Retreiving documents for the query: {query}")
        logging.info(f"top_k:{top_k}, score_threashold:{score_threashold}")

        # generate query embeddings
        query_embedding = self.embedding_manager.generate_embeddings(texts=[query])[0]

        # serach in vectr store
        try:
            results = self.vector_store.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=top_k,
            )

            #print(f"RESULT: {results}")
            
            retreived_docs = []

            if results['documents'] and results['documents'][0]:
                documents = results['documents'][0]
                metadatas = results['metadatas'][0]
                distances = results['distances'][0]
                ids = results['ids'][0]

                for i, (doc_id, document, metadata, distance) in enumerate(
                    zip(ids, documents, metadatas, distances)
                ):
                    #convert distance to similiraty score (chromaDB uses cosine similiraty)
                    similarity_score = 1 - distance

                    if similarity_score >= score_threashold:
                        retreived_docs.append(
                            {
                                'id': doc_id,
                                'content': document,
                                'metadata': metadata,
                                'similarity_score': similarity_score,
                                'distance': distance,
                                'rank': i+1,
                            }
                        )
                
                logging.info(f"Retreived {len(retreived_docs)} after filtering")
            
            else:
                logging.info("No documents found")
            return retreived_docs
        
        except Exception as e:
            logging.error(f"Error during retrival: {e}")
            return []
