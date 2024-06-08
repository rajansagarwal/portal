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

        # To be implemented
        # self.text_tokenizer = AutoTokenizer.from_pretrained('nomic-ai/nomic-embed-text-v1.5')
        # self.text_model = AutoModel.from_pretrained('nomic-ai/nomic-embed-text-v1.5', trust_remote_code=True)
        # self.vision_processor = AutoImageProcessor.from_pretrained("nomic-ai/nomic-embed-vision-v1.5")
        # self.vision_model = AutoModel.from_pretrained("nomic-ai/nomic-embed-vision-v1.5", trust_remote_code=True)

    def embed (self, sentences: list[str]) -> list[numpy.ndarray]:
        embeddings = self.model.encode(sentences)
        # print(type(embeddings))
        return embeddings

