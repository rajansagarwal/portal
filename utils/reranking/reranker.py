from sentence_transformers import CrossEncoder
import numpy as np

class Reranker:
    def __init__(self):
        self.model = CrossEncoder('mixedbread-ai/mxbai-rerank-large-v1')
    
    def rerank(self, chroma_results, og_query):
        # rerank the results with original query and documents returned from Chroma
        scores = self.model.predict([(og_query, doc) for doc in chroma_results["documents"][0]])
        #print(scores)
        # get the highest scoring document
        #print(chroma_results)
        arg = np.argmax(scores)
        res = {}
        for key in chroma_results:
            if (chroma_results[key] is not None):
                res[key] = chroma_results[key][0][arg]
        #print(res)
        return res