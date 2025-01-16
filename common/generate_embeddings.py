from sentence_transformers import SentenceTransformer
from common.utils import read_markdown_file
import numpy as np
import faiss
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

vector_db_dir = os.path.join(BASE_DIR, 'common', 'vector_db')


class GenerateVectorDatabase:

    @classmethod
    def generate_embedding(cls, file_path, file_name):
        try:
            data = read_markdown_file(file_path)
            sentences = data.split('\n')
            model = SentenceTransformer('all-MiniLM-L6-v2')

            # Generate embeddings for each sentence
            embeddings = model.encode(sentences, convert_to_tensor=True)

            # Convert embeddings to numpy array
            embeddings_np = embeddings.cpu().numpy()

            # Create a FAISS index (using L2 distance)
            dim = embeddings_np.shape[1]  # Dimension of the embeddings (e.g., 384 for MiniLM)
            index = faiss.IndexFlatL2(dim)

            # Create the directory if it doesn't exist
            os.makedirs(vector_db_dir, exist_ok=True)

            # Example: save FAISS index and embeddings
            faiss_index_file = os.path.join(vector_db_dir, f'{file_name}.index')
            embedding_file = os.path.join(vector_db_dir, f'{file_name}_embeddings.npy')

            # Add embeddings to the FAISS index
            index.add(embeddings_np)

            # Save the FAISS index and embeddings in the common/vector_db folder
            faiss.write_index(index, faiss_index_file)
            np.save(embedding_file, embeddings_np)

            print(f"FAISS index and embeddings saved to {faiss_index_file} and {embedding_file}")
            return {"status": "EMBEDDING GENERATED"}
        except Exception as e:
            print("GENERATE EMBEDDINGS EXC: ", e)
            return
