import streamlit as st
import os
from dotenv import load_dotenv
from typing import List
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores.pgvector import PGVector
from langchain_community.vectorstores import DistanceStrategy
from langchain_huggingface import HuggingFaceEmbeddings
import tempfile

# Load environment variables
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

# Initialize the embedding model
embeddings = SentenceTransformer("all-MiniLM-L6-v2")

# Database connection
connection = "postgresql://neondb_owner:npg_dxtaLVWuD9n3@ep-wispy-frost-a445yv4z-pooler.us-east-1.aws.neon.tech/neondb?sslmode=require"
collection_name = "my_docs"

# Initialize the vector store
db = PGVector(
    embedding_function=embeddings,
    collection_name="docs_vectors",
    distance_strategy=DistanceStrategy.COSINE,
    connection_string=connection
)

# Custom CSS for better UI
st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        padding: 10px 20px;
        border: none;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .upload-section {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)

# App title and description
st.title("📚 PDF Embedding Generator")
st.markdown("""
    Upload your PDF documents and generate embeddings that will be stored in our vector database.
    This allows for efficient semantic search and analysis of your documents.
    """)


def get_chunks_from_pages(pages):
    documents = "\n\n".join(page.page_content for page in pages)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=350,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", "!", "?", ",", " "]
    )
    return text_splitter.split_text(documents)


def process_pdf(file):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        tmp_file.write(file.getvalue())
        tmp_file_path = tmp_file.name

    try:
        loader = PyPDFLoader(tmp_file_path)
        docs = loader.load()
        chunks = get_chunks_from_pages(docs)

        # Store embeddings in the database

        db.add_texts(chunks)

        return len(chunks)
    finally:
        os.unlink(tmp_file_path)


# File upload section
st.markdown('<div class="upload-section">', unsafe_allow_html=True)
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
st.markdown('</div>', unsafe_allow_html=True)

if uploaded_file is not None:
    if st.button("Process PDF"):
        with st.spinner("Processing your PDF..."):
            try:
                num_chunks = process_pdf(uploaded_file)
                st.success(f"✅ Successfully processed PDF! Generated {num_chunks} chunks and stored their embeddings.")
            except Exception as e:
                st.error(f"❌ An error occurred: {str(e)}")


