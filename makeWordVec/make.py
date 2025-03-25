import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import os
import pickle
import numpy as np
import torch
from transformers import CLIPModel, CLIPTokenizer

#512
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

#768
#model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")  
#tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")


  

cub_label_file = "./cub_classes.txt"
cub_labels = []
with open(cub_label_file, "r") as f:
    for line in f:
        cub_labels.append(line.replace('\n',""))
print(cub_labels)

cub_label_embeddings = []
for label in cub_labels:
    input_ids = tokenizer(label, return_tensors="pt").input_ids
    label_embedding = model.get_text_features(input_ids)
    cub_label_embeddings.append(label_embedding.squeeze().detach().numpy())

embeddings = np.array(cub_label_embeddings)
print(embeddings.shape)

with open('vit-base-cub_bird_label_embeddings.pkl', 'wb') as f:
    pickle.dump(embeddings, f)