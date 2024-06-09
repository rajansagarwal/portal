from transformers import CLIPProcessor, CLIPModel
import torch
import faiss
import numpy as np
from PIL import Image
import requests
from io import BytesIO

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

source = [
    {"type": "text", "source": 'Cats lying in bed'},
    {"type": "image", "source": "https://farm1.staticflickr.com/17/20770643_d04d79280b_z.jpg"},
    {"type": "text", "source": 'What are cute animals to cuddle with?'},
    {"type": "text", "source": 'What do cats look like?'},
    {"type": "image", "source": "https://upload.wikimedia.org/wikipedia/commons/thumb/4/43/Cute_dog.jpg/1280px-Cute_dog.jpg"}
]

def encode_images(image_url):
    response = requests.get(image_url)
    img = Image.open(BytesIO(response.content)).convert("RGB")
    inputs = processor(images=img, return_tensors="pt", padding=True, truncation=True)
    outputs = model.get_image_features(**inputs)
    return outputs[0].detach().cpu().numpy()

def encode_texts(text):
    inputs = processor(text=[text], return_tensors="pt", padding=True, truncation=True)
    outputs = model.get_text_features(**inputs)
    return outputs[0].detach().cpu().numpy()

embeddings = []
for item in source:
    if item['type'] == 'image':
        embedding = encode_images(item['source'])
    else:
        embedding = encode_texts(item['source'])
    embeddings.append(embedding)

embeddings = np.vstack(embeddings)

dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings.astype('float32'))

def search(query_text, num_results=5):
    query_embeddings = encode_texts(query_text)
    _, indices = index.search(query_embeddings[np.newaxis, :], num_results)
    return [source[idx]['source'] for idx in indices[0]]

results = search("Dogs")
print(results)
