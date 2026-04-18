"""
Completely offline vector store using TF-IDF.
No downloads, no internet required. Runs 100% offline.
"""

import pickle
import os
import numpy as np
from typing import List, Dict, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class VectorStore:
    """Offline vector store using TF-IDF - no external dependencies."""
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        self.documents = []
        self.vectors = None
        self.index_path = "vectors.pkl"
        self.docs_path = "docs.pkl"
        self.vec_path = "vectorizer.pkl"
        self.load()
        print("Vector store initialized (offline mode)")
    
    def build_index(self, documents: List[Any]):
        """Build index from documents."""
        self.add_documents(documents)
    
    def add_documents(self, documents: List[Any]):
        """Add documents to the store."""
        if not documents:
            return
        
        texts = []
        for doc in documents:
            if hasattr(doc, 'page_content'):
                texts.append(doc.page_content)
            else:
                texts.append(str(doc))

        self.documents.extend(documents)
    
        # Always refit on ALL documents so vocabulary is complete
        all_texts = []
        for doc in self.documents:
            if hasattr(doc, 'page_content'):
                all_texts.append(doc.page_content)
            else:
                all_texts.append(str(doc))
    
        self.vectors = self.vectorizer.fit_transform(all_texts)
    
        print(f"Added {len(documents)} chunks. Total: {len(self.documents)}")
        self.save()
    
    def save(self):
        """Save to disk."""
        with open(self.index_path, 'wb') as f:
            pickle.dump(self.vectors, f)
        with open(self.docs_path, 'wb') as f:
            pickle.dump(self.documents, f)
        with open(self.vec_path, 'wb') as f:
            pickle.dump(self.vectorizer, f)
        print("Saved to disk")
    
    def load(self) -> bool:
        """Load from disk."""
        if all(os.path.exists(p) for p in [self.index_path, self.docs_path, self.vec_path]):
            with open(self.index_path, 'rb') as f:
                self.vectors = pickle.load(f)
            with open(self.docs_path, 'rb') as f:
                self.documents = pickle.load(f)
            with open(self.vec_path, 'rb') as f:
                self.vectorizer = pickle.load(f)
            print(f"Loaded {len(self.documents)} chunks from disk")
            return True
        return False
    
    def clear(self):
         """Clear all documents and delete all saved files."""
         self.documents = []
         self.vectors = None
         self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df_=1,
            subliner_tf=True,
            )
         
         for path in [self.index_path, self.docs_path, self.vec_path]:
            if os.path.exists(path):
                os.remove(path)
         print("Cleared all documents")
    
    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Search for relevant chunks."""
        if not self.documents or self.vectors is None:
            return []
        
        query_vec = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.vectors).flatten()
        
        top_indices = similarities.argsort()[-k:][::-1]
        
        results = []
        for idx in top_indices:
            doc = self.documents[idx]
            content = doc.page_content if hasattr(doc, 'page_content') else str(doc)
            results.append({
                'content': content,
                'score': float(similarities[idx]),
                'metadata': doc.metadata if hasattr(doc, 'metadata') else {}
                })
        
        return results
