from __future__ import annotations

import logging

from pathlib import Path
from typing import List, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader, DirectoryLoader

logging.basicConfig(level=logging.NOTSET)

class DocsLoader:
    """Handles loading of documents and converting it to chunks"""
    def __init__(self, *, path: str) -> None:
        if not Path(path).is_dir():
            raise FileNotFoundError(
                f"Files with specified directory: {path}, don't exist")
        else:
            self.path = path
    
    def _load(self) -> List[Any]:
        """Loads files from entire directory, Create a seprate document for each page"""
        pdf_dir_loader_docs: List = DirectoryLoader(
            path=self.path,
            glob="**/*.pdf",
            loader_cls=PyMuPDFLoader,
            show_progress=False,
        ).load()

        logging.info(f"Documents loaded from: {self.path}")
        return pdf_dir_loader_docs

    def _chunk_docs(self, *, loaded_docs: List[Any]) -> List[Any]:
        """Create chunks from loaded documents"""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " ", ""],
        )

        chunks: list = text_splitter.split_documents(loaded_docs)
        logging.info(f"Splitted {len(loaded_docs)} Docs into {len(chunks)} Chunks")
        chunk_texts = [doc.page_content for doc in chunks]

        return chunks, chunk_texts
    
    def load_and_chunk(self) -> List[Any]: 
        return self._chunk_docs(loaded_docs=self._load())
