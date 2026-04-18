"""
Main Retrieval-Augmented Generation Application with Streamlit Interface
local (runs entirely on my machine)
"""

import os
import tempfile
import streamlit as st
from document_processor import DocumentProcessor
from vector_store_offline import VectorStore
import requests
import json

# Page Configuration
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = VectorStore()
    # try to load existing index
    st.session_state.vector_store.load()

if 'processor' not in st.session_state:
    st.session_state.processor = DocumentProcessor(chunk_size=1000, chunk_overlap=200)

if 'messages' not in st.session_state:
    st.session_state.messages = []
 
def query_ollama(prompt: str, context: str, model: str = "llama3.2:1b") -> str:
    """
    Query local Ollama model with context.
    """
    import requests
    import json
    
    system_prompt = """You are a helpful assistant answering questions based ONLY on the provided context.
If the context doesn't contain the answer, say "I don't have information about that in my documents."
Be concise and accurate."""

    full_prompt = f"{system_prompt}\n\nContext:\n{context}\n\nQuestion: {prompt}\n\nAnswer:"
    
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model,
                "prompt": full_prompt,
                "stream": False,
                "options": {
                    "temperature": 0.3,
                    "num_predict": 300
                }
            },
            timeout=80  # Add timeout to prevent hanging
        )
        
        if response.status_code == 200:
            result = response.json()
            return result.get("response", "No response generated")
        else:
            return f"Ollama error: HTTP {response.status_code} - {response.text}"
            
    except requests.exceptions.ConnectionError:
        return "Cannot connect to Ollama. Make sure it's running with 'ollama serve'"
    except requests.exceptions.Timeout:
        return "Request timed out. The model might be too slow on your hardware."
    except Exception as e:
        return f"Error: {str(e)}"


# Sidebar - document oversight
with st.sidebar:
    st.title("Knowledge base nexus")
    st.markdown("Add South African documents to the query")

    # file uploader
    uploaded_files = st.file_uploader(
        "Upload documents",
        type=['pdf', 'docx', 'txt'],
        accept_multiple_files=True
    )


    if uploaded_files:
        if st.button("Process Documents", key="process_btn"):
            with st.spinner("Processing documents..."):
                all_chunks = []
                
                for uploaded_file in uploaded_files:
                    # Save uploaded file temporarily
                    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_path = tmp_file.name
                        
                    # Process document
                    chunks = st.session_state.processor.process_file(tmp_path)
                    all_chunks.extend(chunks)
                
                    # Clean up
                    os.unlink(tmp_path)
        
            # Build vector index
            if all_chunks:
                st.session_state.vector_store.build_index(all_chunks)
                st.session_state.vector_store.save()
                st.success(f"Processed {len(all_chunks)} chunks")
                st.rerun()
            else:
                st.error("No content extracted from documents")

    # document list
    if st.session_state.vector_store.documents:
        st.markdown(f"**Indexed chunks:** {len(st.session_state.vector_store.documents)}")

        # clear button
        if st.button("Clear knowledge base"):
            st.session_state.vector_store.clear()
            st.session_state.vector_store.documents = []
            st.session_state.vector_store.vectors = None
            st.session_state.messages = []
            st.rerun()

    # model selection
    st.markdown("---")
    st.markdown("**Model Settings**")
    model_choice = st.selectbox(
        "LLM Model",
        ["llama3.2:1b", "llama3.2:1b", "codellama:7b"],
        index=0,
        help="Smaller models use less RAM"
    )

    # System info
    st.markdown("---")
    st.markdown("**System Info**")
    st.markdown(f"- RAM: ~{len(st.session_state.vector_store.documents) * 0.1:.1f}MB for vectors")
    st.markdown("- Running entirely locally")

# Main chat interface
st.title("South African Local RAG System")
st.markdown("Ask questions about your uploaded documents - **100% private, runs on your laptop/mobile device**")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask a question about your documents..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # Check if documents are available
            if not st.session_state.vector_store.documents:
                response = "Please upload documents first to build the knowledge base."
            else:
                # search for relevant context
                results = st.session_state.vector_store.search(prompt, k=3)

                # TEMPORARY DEBUG
                st.write(f"**Debug Info:**")
                st.write(f"- Docs indexed: {len(st.session_state.vector_store.documents)}")
                st.write(f"- Vectors shape: {st.session_state.vector_store.vectors.shape if st.session_state.vector_store.vectors is not None else 'None'}")
                st.write(f"- Results returned: {len(results)}")
                if results:
                    for r in results:
                        st.write(f"- Score: {r['score']:.4f} | {r['content'][:80]}")

                if not results:
                    response = "No relevant documents found. Try uploading more content."
                else:
                    # combine context
                    context = "\n\n".join([r['content'] for r in results]) 

                    # Show sources (optional)
                    with st.expander("View sources"):
                        for i, r in enumerate(results):
                            st.markdown(f"**Source {i+1}** (similarity: {r['score']:.2f})")
                            st.markdown(r['content'][:300] + "...")

                    # Query LLM
                    response = query_ollama(prompt, context, model=model_choice)

            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})