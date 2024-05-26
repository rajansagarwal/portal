from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
import numpy

class EmbeddingsEngine:
    #model_name: "default" gives the default engine
    def __init__(self, model_name: str):
        if (model_name == "default"):
            self.model =  SentenceTransformer('Alibaba-NLP/gte-large-en-v1.5', trust_remote_code=True)
        else:
            self.model =  SentenceTransformer(model_name, trust_remote_code=True)

    def embed (self, sentences: list[str]) -> list[numpy.ndarray]:
        embeddings = self.model.encode(sentences)
        # print(type(embeddings))
        return embeddings

