import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, AutoImageProcessor
from PIL import Image
import requests

processor = AutoImageProcessor.from_pretrained("nomic-ai/nomic-embed-vision-v1.5")
vision_model = AutoModel.from_pretrained("nomic-ai/nomic-embed-vision-v1.5", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained('nomic-ai/nomic-embed-text-v1.5')
text_model = AutoModel.from_pretrained('nomic-ai/nomic-embed-text-v1.5', trust_remote_code=True)
text_model.eval()

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def embed_image(url):
    print("EMBEDDING IMAGE")
    image = Image.open(requests.get(url, stream=True).raw)
    inputs = processor(image, return_tensors="pt")
    img_emb = vision_model(**inputs).last_hidden_state
    img_embeddings = F.normalize(img_emb[:, 0], p=2, dim=1)
    return img_embeddings

def embed_text(sentence):
    print("EMBEDDING TEXT")
    encoded_input = tokenizer(sentence, padding=True, truncation=True, return_tensors='pt')

    with torch.no_grad():
        model_output = text_model(**encoded_input)

    text_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    text_embeddings = F.layer_norm(text_embeddings, normalized_shape=(text_embeddings.shape[1],))
    text_embeddings = F.normalize(text_embeddings, p=2, dim=1)
    return text_embeddings

def distributor(type, source):
    if type == 'image':
        return embed_image(source)
    if type == 'text':
        return embed_text(source)

def similarity_search(img_embeddings, text_embeddings, sources):
    print("SIMILARITY SEARCH")
    similarity_scores = torch.matmul(img_embeddings, text_embeddings.T)
    sorted_scores, sorted_indices = torch.sort(similarity_scores, descending=True)
    for idx in sorted_indices[0]:
        source_type = sources[idx]['type']
        source_content = sources[idx]['source']
        print(f"Score: {sorted_scores[0][idx].item()}, Source ({source_type}): {source_content}")

print("STARTING")
source = [
    {
        "type": "text",
        "source": 'Cats lying in bed'
    },
    {
        "type": "image",
        "source": "https://farm1.staticflickr.com/17/20770643_d04d79280b_z.jpg"
    },
    {
        "type": "text",
        "source": 'What are cute animals to cuddle with?'
    },
    {
        "type": "text",
        "source": 'What do cats look like?'
    },
        {
        "type": "image",
        "source": "https://upload.wikimedia.org/wikipedia/commons/thumb/4/43/Cute_dog.jpg/1280px-Cute_dog.jpg"
    }
]

query = distributor('text', 'search_query: dogs')
corpus = [distributor(item['type'], item['source']) for item in source]
corpus_embeddings = torch.cat(corpus, dim=0)
similarity_search(query, corpus_embeddings, source)
