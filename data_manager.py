# File: data_manager.py
import os
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

class DataManager:
    def __init__(self, data_dir='data'):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        self.embed_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        self.index = self.load_index()
        self.id_to_post = self.load_posts()

    def load_index(self):
        index_path = os.path.join(self.data_dir, 'faiss_index.pkl')
        if os.path.exists(index_path):
            with open(index_path, 'rb') as f:
                return pickle.load(f)
        else:
            index = faiss.IndexFlatL2(768)  # 768 is the embedding dimension for MiniLM-L6-v2
            return index

    def save_index(self):
        index_path = os.path.join(self.data_dir, 'faiss_index.pkl')
        with open(index_path, 'wb') as f:
            pickle.dump(self.index, f)

    def load_posts(self):
        posts_path = os.path.join(self.data_dir, 'posts.pkl')
        if os.path.exists(posts_path):
            with open(posts_path, 'rb') as f:
                return pickle.load(f)
        return {}

    def save_posts(self):
        posts_path = os.path.join(self.data_dir, 'posts.pkl')
        with open(posts_path, 'wb') as f:
            pickle.dump(self.id_to_post, f)

    def update_embeddings(self, post_id, post_text):
        embedding = self.embed_model.encode([post_text])[0]
        self.index.add(np.array([embedding]).astype('float32'))
        self.id_to_post[str(post_id)] = post_text
        self.save_posts()

    def search_index(self, query_embedding):
        return self.index.search(query_embedding.astype('float32'), k=5)

    def get_post_text(self, post_id):
        return self.id_to_post.get(str(post_id), "Post not found")