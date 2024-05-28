import sys
import os
import numpy as np
from annoy import AnnoyIndex

# Append the root directory of your project to sys.path
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(root_dir)

from utils.embeddings.embeddings_engine import EmbeddingsEngine

class SearchEngine:
    def __init__ (self, data: list[str]):
        self.embeddings_model = EmbeddingsEngine("default")
        if len(data) != 0:
            self.data = self.embeddings_model.embed(data)
            self.text_data = data

    def add_data (self, new_data: list[str]) -> None:
        if len(self.data) != 0:
            self.data = np.concatenate([self.data, self.embeddings_model.embed(new_data)], axis=0)
            self.text_data += new_data
        else:
            self.data = self.embeddings_model.embed(new_data)
            self.text_data = new_data
    
    def query(self, query_text: str) -> list[int]:
        query_vector = self.embeddings_model.embed([query_text])[0]

        num_features = self.data.shape[1]
        annoy_index = AnnoyIndex(num_features, 'angular')

        for index, data_vector in enumerate(self.data):
            annoy_index.add_item(index, data_vector)

        annoy_index.build(10)
        nearest_neighbors = annoy_index.get_nns_by_vector(query_vector, 100, include_distances=False)
        
        print(nearest_neighbors)

        return nearest_neighbors
    
    def query_text_results (self, results: list[int]) -> list[str]:
        text_results = []
        for item in results:
            text_results.append(self.text_data[item])
        return text_results

    
# Example Usage

# search_engine = SearchEngine(["hello world", "hello rajan"])
# print(search_engine.data)
# search_engine.add_data(["Cats and dogs are super cool", "Rajan is the coolest guy"])
# nearest_neighbours = search_engine.query("rajan is super cool")
# print(search_engine.query_text_results(nearest_neighbours))