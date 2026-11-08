import os
from dotenv import load_dotenv
from google import genai
import chromadb
from pypdf import PdfReader
import streamlit as st

# 1. Load environment variables and initialize Gemini client
load_dotenv()
client = genai.Client()

# Initialize ChromaDB persistent client
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(
    name="pdf_knowledge_base"
)

# 2. Streamlit Page Configuration
st.set_page_config(
    page_title="AI RAG Assistant", page_icon="🤖", layout="centered"
)

st.title("🤖 AI RAG Assistant (Powered by Gemini & ChromaDB)")
st.write(
    "Upload your PDF document or use the pre-indexed data, then ask any"
    " question to get accurate answers!"
)

# Sidebar for PDF Upload
st.sidebar.header("📁 Document Management")
uploaded_file = st.sidebar.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file is not None:
  # Save the uploaded file locally
  pdf_path = "temp_uploaded.pdf"
  with open(pdf_path, "wb") as f:
    f.write(uploaded_file.getbuffer())

  st.sidebar.success("PDF uploaded successfully!")

  # Read and index the uploaded PDF
  if st.sidebar.button("Index PDF into Database"):
    with st.spinner("Processing and indexing PDF..."):
      reader = PdfReader(pdf_path)
      full_text = ""
      for page in reader.pages:
        text = page.extract_text()
        if text:
          full_text += text + "\n"

      # Simple Chunking
      chunks = []
      chunk_size = 400
      overlap = 50
      start = 0
      while start < len(full_text):
        end = start + chunk_size
        chunks.append(full_text[start:end])
        start += chunk_size - overlap

      # Indexing into ChromaDB
      for i, chunk in enumerate(chunks):
        if not chunk.strip():
          continue
        response = client.models.embed_content(
            model="gemini-embedding-001", contents=chunk
        )
        vector = response.embeddings[0].values
        collection.add(
            documents=[chunk],
            embeddings=[vector],
            ids=[f"uploaded_chunk_{i}"],
        )
    st.sidebar.success(
        f"Indexed {len(chunks)} chunks successfully into ChromaDB!"
    )

# 3. Main Chat Interface for User Query
user_query = st.text_input(
    "Ask a question about your document:",
    placeholder="e.g., What is the main topic of the document?",
)

if st.button("Generate Answer"):
  if user_query:
    with st.spinner("Searching documents and generating response..."):
      # A. Generate embedding for user query
      query_response = client.models.embed_content(
          model="gemini-embedding-001", contents=user_query
      )
      query_vector = query_response.embeddings[0].values

      # B. Retrieve the closest matching chunk from ChromaDB
      results = collection.query(
          query_embeddings=[query_vector], n_results=2
      )

      # Check if any documents were found
      if results["documents"] and len(results["documents"][0]) > 0:
        retrieved_context = "\n".join(results["documents"][0])

        # C. Generate final answer using Gemini LLM
        rag_prompt = f"""
                You are an expert AI assistant. Answer the user's question accurately using ONLY the provided context below. 
                If the answer cannot be found in the context, say "I cannot answer this based on the provided documents."

                Context:
                {retrieved_context}

                User Question:
                {user_query}
                """

        response = client.models.generate_content(
            model="gemini-3.8-flash", contents=rag_prompt
        )

        # Display Output
        st.subheader("💡 AI Response:")
        st.write(response.text)

        with st.expander("🔍 View Retrieved Context (Source):"):
          st.write(retrieved_context)
      else:
        st.warning(
            "No relevant documents found in the database. Please index a PDF"
            " first."
        )
  else:
    st.warning("Please enter a question first.")