import chromadb
import sys
import os
import numpy as np

root_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(root_dir)

from utils.embeddings.embeddings_engine import EmbeddingsEngine



class ChromaSearchEngine:

    def __init__(self):
        self.embeddings_model = EmbeddingsEngine("default")
        self.id = 0

    def create_client (self):
        self.client = chromadb.Client()

    def create_collection(self, collection_name:str):
        self.collection = self.client.create_collection(name=collection_name, metadata={"hnsw:space": "cosine"})

    def add_data(self, data:list[str]):  
        for datapoint in data:
            embedding = self.embeddings_model.embed(datapoint).tolist()
            #print(embedding)
            id_string = f"id_{self.id}"
            self.collection.add(
                documents=[datapoint],
                embeddings=[embedding],
                ids=[id_string]
            )
            self.id = self.id + 1


    def query_data(self, query_string):
        embedding = self.embeddings_model.embed(query_string).tolist()
        return self.collection.query(
            query_embeddings=[embedding],
            include=["documents"],
            n_results=2
        )

    def delete_collection(self, collection_name):
        self.client.delete_collection(name=collection_name)

engine = ChromaSearchEngine()
print("Creating client")
engine.create_client()
print("Creating collection")
engine.create_collection("new_collection")
print("Creating data")
engine.add_data(["Rajan is so cool", "Rajan has a cool dog", "It is raining outside"])
print("Creating query")
print(engine.query_data("The rain is mean to dogs"))
print("Deleting collection")
engine.delete_collection("new_collection")