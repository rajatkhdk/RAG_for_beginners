import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage


load_dotenv()

persistent_directory="db/chroma_db"

# Load embeddings and vector store
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

db = Chroma(
    persist_directory=persistent_directory,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space": "cosine"}
)

# Search for relevant documents
query = "In what year did Tesla begin production of the Roadster?"

retriever = db.as_retriever(search_kwargs={"k": 3})

# retriever = db.as_retriever(
#     search_type="similarity_score_threshold", 
#     search_kwargs={
#         "k": 3, 
#         "score_threshold": 0.3 # Only return chunks with cosine similarity >= 0.3
#         }
#     )

relevant_docs = retriever.invoke(query)

print(f"User Query: {query}\n")

# # Display results
# print("--- Context ---")
# for i, doc in enumerate(relevant_docs, 1):
#     print(f"Document {i}:")
#     print(doc.page_content)
#     print(f"Source: {doc.metadata.get('source', 'Unknown')}")
#     print("-" * 50)



# Combine the query and relevant document context into a prompt for Groq

# Combine the query and the relevant document contents
combined_prompt = f"""Based on the following documents, please answer this question: {query}

Documents:
{chr(10).join([f"- {doc.page_content}" for doc in relevant_docs])}

Please provide a clear, helpful answer using only the information from these documents. If you can't find the answer in the documents, say "I don't have enough information to answer that question based on the provided documents."
"""

# combined_prompt = f"""
# Answer the question using the information from the documents.

# Question: {query}

# Documents:
# {chr(10).join([doc.page_content for doc in relevant_docs])}

# Answer concisely:
# """


# Create a Groq chat model instance
model = ChatGroq(groq_api_key=os.getenv("GROQ_API_KEY"), model="llama-3.3-70b-versatile", temperature=0)

# Create messages for the chat model
messages = [
    SystemMessage(content="You are a helpful assistant that provides accurate information based on the provided context."),
    HumanMessage(content=combined_prompt)
]

# Invoke the model with the messages
response = model.invoke(messages)

# Display the response
print("\n--- Groq Model Response ---")
print(response.content)

# Synthetic Questions: 

# 1. "What was NVIDIA's first graphics accelerator called?"
# 2. "Which company did NVIDIA acquire to enter the mobile processor market?"
# 3. "What was Microsoft's first hardware product release?"
# 4. "How much did Microsoft pay to acquire GitHub?"
# 5. "In what year did Tesla begin production of the Roadster?"
# 6. "Who succeeded Ze'ev Drori as CEO in October 2008?"
# 7. "What was the name of the autonomous spaceport drone ship that achieved the first successful sea landing?"
# 8. "What was the original name of Microsoft before it became Microsoft?"