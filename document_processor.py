"""Document processor for RAG system
Handles the loading and chunking of all documents types
"""
import os
from typing import List
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


class DocumentProcessor:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """Initializes the DocumentProcessor with specified chunk size and overlap."""
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""])

    def load_document(self, file_path: str) -> List[Document]:
        """Loads a document from the specified file path.
        Args:
            file_path: Path to document (PDF, DOCX, TXT)
        Returns:
            List of Langchain Document objects
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        file_extension = os.path.splitext(file_path)[1].lower()  # FIX: was splittext

        try:
            if file_extension == '.pdf':
                loader = PyPDFLoader(file_path)
                documents = loader.load()
                print(f"Loaded PDF: {file_path} ({len(documents)} pages)")

            elif file_extension == '.docx':
                loader = Docx2txtLoader(file_path)  # FIX: was Doc2txtLoader and fole_path
                documents = loader.load()
                print(f"Loaded Word document: {file_path}")

            elif file_extension == '.txt':
                loader = TextLoader(file_path, encoding='utf-8')
                documents = loader.load()
                print(f"Loaded text file: {file_path}")

            else:
                raise ValueError(f"Unsupported file type: {file_extension}")

            return documents

        except Exception as e:
            print(f"Error loading {file_path}: {str(e)}")
            return []

    def chunk_documents(self, documents: List[Document]) -> List[Document]:  # FIX: was doocuments
        """Splits documents into smaller chunks.
        Args:
            documents: List of documents to chunk
        Returns:
            List of chunked documents
        """
        chunked_docs = self.text_splitter.split_documents(documents)
        print(f"Split into {len(chunked_docs)} chunks")
        return chunked_docs

    def process_file(self, file_path: str) -> List[Document]:  # FIX: was procces_file, also was outside class
        """Complete pipeline: load and chunk a single file.
        Args:
            file_path: Path to document
        Returns:
            List of chunked documents ready for embedding
        """
        documents = self.load_document(file_path)
        if documents:
            return self.chunk_documents(documents)
        return []