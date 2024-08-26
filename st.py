from sentence_transformers import SentenceTransformer
import json
import numpy as np

# 1. Load a pretrained Sentence Transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

sentences = [
    "move your hand out from your body",
    "I want you to move your hand away from your body",
    "move your hand in towards your body",
    "move your hand to your side",
    "move your hand towards my side"
]

emb = model.encode(sentences)

x = model.similarity(emb,emb)
for i in x:
    for j in i:
        print("{0:.3f}".format(j.item()),end=' ')
    print()