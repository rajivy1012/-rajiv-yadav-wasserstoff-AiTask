# File: init_faiss_index.py
import os
import faiss
from data_manager import DataManager


def initialize_index():
    # Create data manager instance
    data_manager = DataManager()

    # Initialize FAISS index with correct embedding dimension
    index = faiss.IndexFlatL2(768)  # 768 for MiniLM-L6-v2

    # Set the index and save it
    data_manager.index = index
    data_manager.save_index()

    print("FAISS index initialized successfully")


if __name__ == "__main__":
    initialize_index()