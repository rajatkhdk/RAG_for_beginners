import os
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

def load_documents(docs_path="docs"):
    """Load all .txt documents from the specified directory."""
    print(f"Loading documents from {docs_path}...")
    
    if not os.path.exists(docs_path):
        raise FileNotFoundError(f"The directory {docs_path} does not exist. Please create it and add your text files.")
    
    #Load all .txt files from the docs directory
    loader = DirectoryLoader(
        path=docs_path,
        glob="**/*.txt",
        loader_cls=TextLoader
    )

    documents = loader.load()

    if len(documents) == 0:
        raise ValueError(f"No text files found in the directory {docs_path}. Please add some .txt files to proceed.")
    
    # for i, doc in enumerate(documents[:2]): # Show first 2 documents for verification
    #     print(f"Loaded document {i+1}: {doc.metadata.get('source', 'Unknown source')} with {len(doc.page_content)} characters.")
    #     print(f"Content preview: {doc.page_content[:200]}...\n")
    #     print(f"Metadata: {doc.metadata}\n")

    return documents

def split_documents(documents, chunk_size=800, chunk_overlap=0):
    """Split documents into smaller chunks."""
    print("Splitting documents into chunks...")

    text_splitter = CharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = text_splitter.split_documents(documents)

    # if chunks:

    #     for i, chunk in enumerate(chunks[:5]): # Show first 5 chunks for verification
    #         print(f"Chunk {i+1}: {chunk.metadata['source']} with {len(chunk.page_content)} characters.")
    #         print(f"Content preview: {chunk.page_content[:800]}...\n")

    #     if len(chunks) > 5:
    #         print(f"...and {len(chunks) - 5} more chunks.\n")

    return chunks

def create_vector_store(chunks, persist_directory="db/chroma_db"):
    """Create and persist Chroma vector store from document chunks."""
    print("Creating embeddings and storing in ChromaDB...")

    # embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
    # embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
    
    # Create the Chroma vector store
    print(f"Persisting vector store to {persist_directory}...")
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory,
        collection_metadata={"hnsw:space": "cosine"}
    )

    print("--Finished creating vector store--")

    print(f"Vector store created and persisted at {persist_directory}.")
    return vector_store

def main():
    print("Starting ingestion pipeline...")

    # 1. Load the files
    documents = load_documents(docs_path="docs")

    # 2. Split the text into chunks
    chunks = split_documents(documents, chunk_size=800, chunk_overlap=0)
    # 3. Create embeddings and Store in a vector database
    vector_store = create_vector_store(chunks, persist_directory="db/chroma_db")


if __name__ == "__main__":
    main()