from __future__ import annotations

import dotenv
import logging

from langchain.agents import create_agent
from src.vector_store import VectorStore
from src.embedding import EmbeddingManager
from src.rag_retrieval import RagReteiver
from typing import List, Dict, Any

#load env vars
dotenv.load_dotenv()

logging.basicConfig(level=logging.NOTSET)


class RagAgent:
    """Handling and initializing RAG based agent"""
    def __init__(
        self,
        *,
        query: str,
    
        retriver: RagReteiver=RagReteiver(
            vector_store=VectorStore(), 
            embedding_manager=EmbeddingManager(),
        ),

        model: str='gpt-4.1-mini',
        top_k: int=3,
    ) -> None:

        self.query = query
        self.model = model
        self.retriver = retriver
        self.top_k = top_k
        self.rag_agent = None
        self._initalize_agent()

    def _initalize_agent(self) -> None:
        """Initialize LLM agent with context from vector store"""

        #get context results from vector store
        retrive_results = self._retrive_context()

        #build context
        context = "\n\n".join(
            [doc['content'] for doc in retrive_results]
        ) if retrive_results else ""

        self.rag_agent = create_agent(
            model = self.model,
            #This is the prompt that gets combined with the context and it's called augmentation
            system_prompt = f"""
            Use the following context to answer the question concisely.

            Context:
            {context}

            Question:
            {self.query}
            """,
        )

        logging.info("Agent initialized...")

    def _retrive_context(self) -> List[Dict[str, Any]]:
        """Retrieve context from vector store"""
        return self.retriver.retreive(query=self.query)


    def agent_invoke(self) -> str:
        return self.rag_agent.invoke({
            'messages': [
                {'role': 'user', 'content': f'{self.query}'}
            ]
        })['messages'][-1].content
