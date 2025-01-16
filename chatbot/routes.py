import os
import ollama
import faiss
import numpy as np
from fastapi import APIRouter, requests
from sentence_transformers import SentenceTransformer

chatbot_router = APIRouter()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
vector_db_dir = os.path.join(BASE_DIR, 'common', 'vector_db')

# Load FAISS index and embeddings
index_file = os.path.join(vector_db_dir, 'your_faiss_index.index')
embeddings_file = os.path.join(vector_db_dir, 'your_faiss_embeddings.npy')

index = faiss.read_index(index_file)
embeddings_np = np.load(embeddings_file)


@chatbot_router.get('/chatbot-query', tags=['ChatBot'])
async def user_query(query: str):
    try:
        model = SentenceTransformer('all-MiniLM-L6-v2')
        query_embedding = model.encode([query], convert_to_tensor=True).cpu().numpy()
        D, I = index.search(query_embedding, k=5)

        matched_documents = []
        for idx in I[0]:
            matched_documents.append(embeddings_np[idx])  # Fetch the relevant embeddings

        matched_texts = []
        for idx in I[0]:
            document_text = "Document content corresponding to index"  # You can retrieve the content based on the index
            matched_texts.append(document_text)

        prompt = f"Here is the relevant information from the documents: {matched_texts}. Based on the query: '{query}', please provide a helpful answer."

        response = ollama.chat(model="llama:3.2", messages=[{"role": "user", "content": prompt}])

        return {"response": response["text"]}

    except Exception as e:
        print("USER QUERY EXC: ", e)
        return {"status": "unsuccessful"}