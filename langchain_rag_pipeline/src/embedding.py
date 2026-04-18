from __future__ import annotations

import logging
import numpy as np

from typing import List, Any
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

logging.basicConfig(level=logging.NOTSET)

class EmbeddingManager:
    """Handles embedding of docs using Sentence Transformer"""
    def __init__(
            self,
            #This is the Embedding model from HuggingFace
            model_name: str="all-MiniLM-L6-v2",
    ):
        self.model_name = model_name
        self.model = None,
        self._load_model()

    def _load_model(self) -> None:
        """Load the Transformer model"""
        try:
            logging.info(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            logging.info(f"Model loaded successfully,\
            total dimensions: {self.model.get_embedding_dimension()}")
    
        except Exception as e:
            logging.error(f"Error loading model {self.model_name}: {e}")

    def generate_embeddings(self, *, texts: List[str]) -> np.ndarray:
        if not self.model:
            raise ValueError("Model not found")

        logging.info(f"Generating embeddings for {len(texts)} texts...")
        embeddings = self.model.encode(texts, show_progress_bar=True)
        logging.info(f"Generated embeddings with shape: {embeddings.shape}")

        return embeddings
