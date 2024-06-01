from sentence_transformers import SentenceTransformer, util
import numpy as np
from utils.embeddings.embeddings_engine import EmbeddingsEngine

class ClusteringEngine:
    def __init__(self, embeddings_engine, threshold=0.7):
        self.embeddings_engine = embeddings_engine
        self.events = []
        self.event_embeddings = []
        self.threshold = threshold

    def add_event_description(self, description: str):
        new_embedding = self.embeddings_engine.embed([description])[0]
        
        if self.event_embeddings:
            # Calculate similarity with existing event embeddings
            similarities = [util.cos_sim(new_embedding, emb)[0][0] for emb in self.event_embeddings]
            max_similarity = max(similarities)
            best_event_index = np.argmax(similarities)
            
            # If similarity exceeds threshold, update existing event
            if max_similarity > self.threshold:
                self.events[best_event_index].append(description)
                self.event_embeddings[best_event_index] = np.mean(
                    [self.event_embeddings[best_event_index], new_embedding], axis=0)
                return

        # If no existing event is similar enough, add as new event
        self.events.append([description])
        self.event_embeddings.append(new_embedding)

    def get_events(self):
        return self.events