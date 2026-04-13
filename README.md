# 🇿🇦 SA Local RAG System

**A completely offline, private document Q&A system that runs on a laptop with only 8GB RAM.**

Built with Python, LangChain, Ollama, and Streamlit.

## 🔍 What This Does

Upload PDFs, Word documents, or text files. Ask questions in natural language. Get answers based **only** on your documents. No data ever leaves your machine.

## 🛠️ Tech Stack

- **Local LLM:** Ollama + Llama 3.2 (1B/3B)
- **Embeddings:** Fastembed / TF-IDF
- **Vector Search:** FAISS / Custom similarity
- **UI:** Streamlit
- **Language:** Python

## 🚀 Why I Built This

I completed my first year of an IT degree coding entirely from my phone. When I finally got a laptop, I wanted to build something that proved I could ship working AI systems under real constraints.

This project runs on **8GB RAM** and requires no internet after setup. That's not a limitation – it's a design requirement for real-world deployment.

## 📋 Features

- Upload multiple document types (PDF, DOCX, TXT)
- Automatic text chunking and embedding
- Local LLM inference (no API calls)
- Source attribution for answers
- Persistent storage of indexed documents

## 🔧 Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/sa-rag-system.git
cd sa-rag-system

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Pull Ollama models
ollama pull llama3.2:1b
ollama pull nomic-embed-text

# Run the app
streamlit run app.py
```
sa-rag-system/
├── app.py                 # Streamlit UI
├── document_processor.py  # PDF/DOCX loading & chunking
├── vector_store_simple.py # Embeddings & similarity search
├── requirements.txt       # Dependencies
└── README.md             # This file


Save the file.

---

## Step 6: Commit and Push

Now push everything to GitHub:

```bash
# Add README
git add README.md

# Commit with a meaningful message
git commit -m "Initial commit: SA Local RAG System with offline LLM support"

# Add your GitHub repository as remote
# Replace YOUR_USERNAME with your actual GitHub username
git remote add origin https://github.com/skii-bee/sa-rag-system.git

# Push to GitHub
git branch -M main
git push -u origin main
