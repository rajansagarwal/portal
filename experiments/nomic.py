import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, AutoImageProcessor
from PIL import Image
import requests

processor = AutoImageProcessor.from_pretrained("nomic-ai/nomic-embed-vision-v1.5")
vision_model = AutoModel.from_pretrained("nomic-ai/nomic-embed-vision-v1.5", trust_remote_code=True)

url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
image = Image.open(requests.get(url, stream=True).raw)

inputs = processor(image, return_tensors="pt")

img_emb = vision_model(**inputs).last_hidden_state
img_embeddings = F.normalize(img_emb[:, 0], p=2, dim=1)


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

sentences = ['search_query: cat', 'search_query: gun']

tokenizer = AutoTokenizer.from_pretrained('nomic-ai/nomic-embed-text-v1.5')
text_model = AutoModel.from_pretrained('nomic-ai/nomic-embed-text-v1.5', trust_remote_code=True)
text_model.eval()

encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

with torch.no_grad():
    model_output = text_model(**encoded_input)

text_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
text_embeddings = F.layer_norm(text_embeddings, normalized_shape=(text_embeddings.shape[1],))
text_embeddings = F.normalize(text_embeddings, p=2, dim=1)

print(torch.matmul(img_embeddings, text_embeddings.T))