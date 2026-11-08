# 🤖 AI RAG Assistant (Gemini & ChromaDB)

A full-stack Retrieval-Augmented Generation (RAG) web application built with Python, Google Gemini API, ChromaDB, and Streamlit. This app allows users to upload PDF documents, index them into a local vector database, and query them using state-of-the-art LLMs with zero hallucination.

## 🚀 Features
- **Document Ingestion:** Upload custom PDF documents and process them dynamically.
- **Text Chunking & Embedding:** Splits documents into optimized chunks and generates semantic vector embeddings using `gemini-embedding-001`.
- **Vector Storage:** Persistently stores and retrieves embeddings using **ChromaDB**.
- **LLM Generation:** Uses **Gemini 3.8-flash** to generate precise, context-aware answers based strictly on the provided documents.
- **Interactive Web UI:** Built with **Streamlit** for a clean, user-friendly experience.

## 🛠️ Tech Stack
- **Python**
- **Google GenAI SDK (`google-genai`)**
- **ChromaDB**
- **Streamlit**
- **PyPDF**

## ⚙️ Installation & Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/Nada981/AI_RAG_Assistant.git
   cd AI_RAG_Assistant

```

2. Install dependencies:
```bash
pip install -r requirements.txt

```


3. Set up your environment variables:
Create a `.env` file in the root directory and add your Gemini API key:
```env
GEMINI_API_KEY=your_actual_api_key_here

```


4. Run the Streamlit application:
```bash
streamlit run app_ui.py
