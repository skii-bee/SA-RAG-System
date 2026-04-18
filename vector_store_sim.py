"""
Simplified vector store using fastembed - no external downloads required on first run
"""
import os
os.environ["HF_HUB_OFFLINE"] = "1"
import numpy as np
import pickle
from typing import List, Dict, Any

class VectorStore:
    """Simple vector store using fastembed embeddings."""
    
    def __init__(self):
        """Initialize with fastembed - downloads model once on first use."""
        print("Loading embedding model (first time may download)...")
        from fastembed import TextEmbedding
        # This model is small and downloads reliably
        self.model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
        self.embeddings = []  # Store embeddings as list of numpy arrays
        self.documents = []   # Store original documents

        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.index_path = os.path.join(base_dir,"embeddings.pkl")
        self.docs_path = os.path.join(base_dir,"documents.pkl")

        self.load()  # Try to load existing data
    
    def build_index(self, documents: List[Any]):
        """Build index from documents-alias for add_documents for compatability"""
        self.add_documents(documents)
    
    def create_embeddings(self, texts: List[str]) -> List[np.ndarray]:
        """Convert text chunks to embeddings."""
        print(f"Creating embeddings for {len(texts)} chunks...")
        # fastembed returns a generator, convert to list
        embeddings = list(self.model.embed(texts))
        return embeddings
    
    def add_documents(self, documents: List[Any]):
        """Add documents to the store."""
        if not documents:
            return
        
        # Extract text from documents (handle both strings and LangChain Document objects)
        texts = []
        for doc in documents:
            if hasattr(doc, 'page_content'):
                texts.append(doc.page_content)
            else:
                texts.append(str(doc))
        
        # Create embeddings
        new_embeddings = self.create_embeddings(texts)
        
        # Store
        self.documents.extend(documents)
        self.embeddings.extend(new_embeddings)
        
        print(f"Added {len(documents)} chunks. Total: {len(self.documents)}")
        self.save()
    
    def save(self):
        """Save embeddings and documents to disk."""
        with open(self.index_path, 'wb') as f:
            pickle.dump(self.embeddings, f)
        with open(self.docs_path, 'wb') as f:
            pickle.dump(self.documents, f)
        print(f"Saved to disk")
    
    def load(self) -> bool:
        """Load embeddings and documents from disk."""
        if os.path.exists(self.index_path) and os.path.exists(self.docs_path):
            with open(self.index_path, 'rb') as f:
                self.embeddings = pickle.load(f)
            with open(self.docs_path, 'rb') as f:
                self.documents = pickle.load(f)
            print(f"Loaded {len(self.documents)} chunks from disk")
            return True
        return False

    def clear(self):
        """Clear all documents and delete saved files from disk."""
        self.documents = []
        self.embeddings = []
        if os.path.exists(self.index_path):
            os.remove(self.index_path)
        if os.path.exists(self.docs_path):
            os.remove(self.docs_path)
            print("Knowledge base cleared.")

    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for most relevant document chunks using cosine similarity.
        """
        if not self.embeddings or len(self.embeddings) == 0:
            return []
        
        # Create query embedding
        query_embedding = list(self.model.embed([query]))[0]
        
        # Calculate cosine similarities
        similarities = []
        for i, doc_emb in enumerate(self.embeddings):
            # Cosine similarity = dot product of normalized vectors
            norm_query = query_embedding / np.linalg.norm(query_embedding)
            norm_doc = doc_emb / np.linalg.norm(doc_emb)
            similarity = np.dot(norm_query, norm_doc)
            similarities.append((i, similarity))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Return top k
        results = []
        for i, score in similarities[:k]:
            doc = self.documents[i]
            content = doc.page_content if hasattr(doc, 'page_content') else str(doc)
            results.append({
                'content': content,
                'score': float(score),
                'metadata': doc.metadata if hasattr(doc, 'metadata') else {}
            })
        
        return results