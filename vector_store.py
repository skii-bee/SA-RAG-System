"""
VStore manager for rag system (handles embeddings, storage, and retrieval, using FAISS)
"""
import os
import pickle
from typing import List, Dict, Any
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from langchain_core.documents import Document


class VectorStore:
    """Oversee document embeddings and similarity checks."""

    def __init__(self, model_name: str = 'sentence-transformers/all-MiniLM-L6-v2'):
        """
        Initialize vector store with embedding model.
        Args:
            model_name: HuggingFace model name
        """
        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        self.index = None
        self.documents = []
        self.docs_path = "documents.pkl"
        self.index_path = "vector_store.index"

    def create_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Convert text chunks to embeddings.
        Args:
            texts: List of strings
        Returns:
            Numpy array of embeddings
        """
        print(f"Creating embeddings for {len(texts)} chunks...")
        embeddings = self.model.encode(texts, show_progress_bar=True)
        return embeddings.astype('float32')

    def build_index(self, documents: List[Document]):
        """
        Build FAISS index from documents.
        Args:
            documents: List of chunked documents
        """
        if not documents:
            print("No documents to index")
            return

        self.documents = documents
        texts = [doc.page_content for doc in documents]
        embeddings = self.create_embeddings(texts)
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)
        print(f"Index built with {self.index.ntotal} vectors")

    def save(self):
        """Save index and documents to disk."""
        if self.index is not None:
            faiss.write_index(self.index, self.index_path)
            with open(self.docs_path, 'wb') as f:
                pickle.dump(self.documents, f)
            print("Saved index and documents")

    def load(self) -> bool:
        """Load index and documents from disk."""
        if os.path.exists(self.index_path) and os.path.exists(self.docs_path):
            self.index = faiss.read_index(self.index_path)
            with open(self.docs_path, 'rb') as f:
                self.documents = pickle.load(f)
            print(f"Loaded index with {self.index.ntotal} vectors")
            return True
        return False

    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for most relevant document chunks.
        Args:
            query: User question
            k: Number of results to return
        Returns:
            List of relevant chunks with scores
        """
        if self.index is None or self.index.ntotal == 0:
            return []

        query_embedding = self.model.encode([query]).astype('float32')
        faiss.normalize_L2(query_embedding)
        scores, indices = self.index.search(query_embedding, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and idx < len(self.documents):
                results.append({
                    'content': self.documents[idx].page_content,
                    'score': float(score),
                    'metadata': self.documents[idx].metadata
                })
        return results