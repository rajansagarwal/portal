import chromadb
import sys
import os
import time
import numpy as np

root_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(root_dir)

from utils.embeddings.embeddings_engine import EmbeddingsEngine
from utils.reranking.reranker import Reranker



class ChromaSearchEngine:

    def __init__(self):
        self.embeddings_model = EmbeddingsEngine("default")

    def create_client (self):
        self.client = chromadb.Client()

    def create_collection(self, collection_name:str):
        self.collection = self.client.create_collection(name=collection_name, metadata={"hnsw:space": "cosine"})

#sample dict: frame id: ignored, audio: embed, frame: embed, video: id_{}, summary: embed

    def add_datapoint(self, type, id, text, user, filepath):
        embedding = self.embeddings_model.embed(text).tolist()
        id_string = f"{id}_{type}"
        self.collection.add(
            documents=[text],
            embeddings=[embedding],
            ids=[id_string],
            metadatas=[{"type" : type}, {"user": user}, {"filepath", filepath}]
        )

    def add_data(self, data_list):  
        for data in data_list:
            if len(data["frame_description"]) != 0:
                self.add_datapoint("frame", data["id"], data["frame_description"])
            if len(data["audio_description"]) != 0:
                self.add_datapoint("audio", data["id"], data["audio_description"])
            if len(data["summary"]) != 0:
                self.add_datapoint("summary", data["id"], data["summary"])
            #print(embedding)
        

    def query_data(self, query_string):
        embedding = self.embeddings_model.embed(query_string).tolist()
        return self.collection.query(
            query_embeddings=[embedding],
            include=["documents"],
            n_results=4
        )
    
    def query_data_by_type(self, query_string, user, type):
        embedding = self.embeddings_model.embed(query_string).tolist()
        return self.collection.query(
            query_embeddings=[embedding],
            include=["documents", "metadatas"],
            where={"type": {"$eq": type}, "user": {"$eq": type}},
            n_results=2
        )

    def delete_collection(self, collection_name):
        self.client.delete_collection(name=collection_name)

# engine = ChromaSearchEngine()
# reranker = Reranker()
# print("Creating client")
# engine.create_client()
# print("Creating collection")
# engine.create_collection("new_collection")
# print("Creating data")
# engine.add_data({"frame_description": "there are lots of birds", "audio_description": "there is a lot of escalators", "id": "store/video_1200.mp4", "summary": "birds on an escalator"})
# start = time.time()
# print("Creating OG query")
# query = "birds and escalators"
# res = engine.query_data(query)
# print(res)
# end = time.time()
# print(end-start)
# print("Reranking")
# print(reranker.rerank(res, query))
# reranking_end = time.time()
# print(reranking_end-start)
# print("Query by type")
# print(engine.query_data_by_type("birds and escalators", "summary"))
# print(engine.query_data_by_type("birds and escalators", "frame"))
# print(engine.query_data_by_type("birds and escalators", "audio"))
# print("Deleting collection")
# engine.delete_collection("new_collection")