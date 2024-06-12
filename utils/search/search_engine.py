import chromadb
import sys
import os
import time
import numpy as np
from chromadb.config import DEFAULT_TENANT, DEFAULT_DATABASE, Settings

root_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(root_dir)

from utils.embeddings.embeddings_engine import EmbeddingsEngine
from utils.reranking.reranker import Reranker

class SearchEngine:

    def __init__(self, collection_name="portal_db"):
        self.embeddings_model = EmbeddingsEngine("default")
        self.client = chromadb.PersistentClient(
            settings=Settings(),
            tenant=DEFAULT_TENANT,
            database=DEFAULT_DATABASE,
        )
        try:
            self.collection = self.client.get_collection(name=collection_name)
        except:
            self.create_collection(collection_name)

    def create_collection(self, collection_name:str):
        self.collection = self.client.create_collection(name=collection_name, metadata={"hnsw:space": "cosine"})

    def add(self, concatenated, description, transcription, summary, filepath):
        embedding = self.embeddings_model.embed(concatenated).tolist()
        id_string = f"{filepath}"
        self.collection.add(
            documents=[concatenated],
            embeddings=[embedding],
            ids=[f"{filepath}"],
            metadatas=[{"filepath": filepath}]
        )
    
    def query(self, query_string, query_type):
        embedding = self.embeddings_model.embed(query_string).tolist()

        if query_type != "":
            results = self.collection.query(
                query_embeddings=[embedding],
                include=["documents", "metadatas"],
                n_results=3
            )
        cleaned_results = {}
        for i in range (len(results["ids"][0])):
            cleaned_results[results["metadatas"][0][i]["filepath"]] = results["documents"][0][i] 
        return cleaned_results

    def delete_collection(self, collection_name):
        self.client.delete_collection(name=collection_name)

# engine = SearchEngine()
# engine.add("knowledge graph", "hi.mp4")
# print(engine.query("knowledge graph", "text"))