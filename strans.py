from sentence_transformers import SentenceTransformer
import json
import numpy as np

# 1. Load a pretrained Sentence Transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

# The sentences to encode
sentences = [
]


with open("knn_db.json", "r") as f:
    db = json.load(f)
    t = np.array([item["phrase"] for item in db])
    for i in t:
        if i.lower() in ["one", "two", "three", "1", "2", "3","okay","this time","now","three go","all right","very good","good","and now"]:
            continue
        sentences.append(i)

#print(sentences)

'''

# 2. Calculate embeddings by calling model.encode()
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 384]

# 3. Calculate the embedding similarities
similarities = model.similarity(embeddings, embeddings)
print(similarities)
# tensor([[1.0000, 0.6660, 0.1046],
#         [0.6660, 1.0000, 0.1411],
#         [0.1046, 0.1411, 1.0000]])

'''




examples = ["move towards yourself", "move away from yourself"]
emb = model.encode(examples)
#print(model.similarity(emb,emb))
x = model.similarity(emb,emb)
for i in x:
    for j in i:
        print("{0:.3f}".format(j.item()),end=' ')
    print()



types = [["move towards me", "move away from you", "move your hand towards myself","move your hand away from yourself","slide outward","move away from your body","move farther","move out"],
         ["move towards you", "move away from me", "move your hand away from me","move your hand towards yourself","slide inward","move towards your body","move closer","move in"],
         ["move your hand up"],
         ["move your hand down"],
         ["doing good"]]


print(len(types))
for i in range(len(sentences)):
    best = []#[0,0,0,0,0]
    for j in range(len(types)):
        best.append(0)
    for j in range(len(types)):
        for k in range(len(types[j])):
            embeddings = model.encode([sentences[i],types[j][k]])
            best[j] = max(best[j],model.similarity(embeddings,embeddings)[0][1])
    #print(best)
    #print(types)
    id = 0
    for j in range(len(best)):
        if best[id] < best[j]:
            id = j
    types[id].append(sentences[i])
for i in types:
    print(i)
    print('\n')
