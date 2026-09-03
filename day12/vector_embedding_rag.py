# Day 12: Vector Embeddings & RAG (Retrieval-Augmented Generation)

# Step 1: What an embedding actually is

# An embedding turns text into a list of numbers (a vector) that captures its meaning — similar sentences produce similar numbers, even if the wording is completely different.

from google import genai
import os
import numpy as np

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

result = client.models.embed_content(
    model="gemini-embedding-001",
    contents="Patient has high blood pressure and irregular exercise habits."
)

print(result.embeddings[0].values[:4])
print(len(result.embeddings[0].values))




# Sentence 1
#      ↓
#  Embedding
#      ↓
# Compare with
#      ↑
#  Embedding
#      ↑
# Weather sentence
#      ↓
# Lower similarity

def get_embedding(text):
    result = client.models.embed_content(
        model="gemini-embedding-001",
        contents=text
    )
    return np.array(result.embeddings[0].values)

# It calculates how similar two vectors are.
# a = [2, 3, 4]
# b = [2, 3, 5]
# (2 × 2) + (3 × 3) + (4 × 5)
# It helps measure how much the two vectors point in the same direction.
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    # What does np.linalg.norm(a) mean?
    # This calculates the length/magnitude of vector a.
    # 5.385 × √(2² + 3² + 5²)

emb1 = get_embedding("Patient has high blood pressure and irregular exercise habits.")
emb2 = get_embedding("The patient suffers from hypertension and doesn't exercise regularly.")
emb3 = get_embedding("The weather today is sunny with a light breeze.")

print("Similarity (related sentences):", cosine_similarity(emb1, emb2))
print("Similarity (unrelated sentences):", cosine_similarity(emb1, emb3))

knowledge_base = [
    "Patient Amaka: diagnosed with hypertension, blood pressure 150/95, prescribed lisinopril.",
    "Patient David: Type 2 diabetes, blood sugar levels well-controlled with metformin.",
    "Patient Chidi: asthma, uses albuterol inhaler as needed, no recent flare-ups.",
    "Patient Bolu: seasonal allergies, prescribed antihistamines during spring months."
]
# knowledge_embeddings = [get_embedding(doc) for doc in knowledge_base]
# This looks complicated at first, but it's actually a shortcut for a for loop.

knowledge_embeddings = [get_embedding(doc) for doc in knowledge_base]
# doc means:

# "Take each document/sentence from knowledge_base, one at a time."
# knowledge_embeddings = []

# for doc in knowledge_base:
#     embedding = get_embedding(doc)
#     knowledge_embeddings.append(embedding)

# What is doc?

# knowledge_base
# │
# ├── 0 → Amaka's information
# ├── 1 → David's information
# ├── 2 → Chidi's information
# └── 3 → Bolu's information




# question → what the user wants to know
# knowledge_base → your original information/documents
# knowledge_embeddings → numerical representations of those documents
# top_k=1 → how many results you want

def search(question, knowledge_base, knowledge_embeddings, top_k=1):
    question_embedding = get_embedding(question)
    scores = [cosine_similarity(question_embedding, doc_emb) for doc_emb in knowledge_embeddings]
    best_index = np.argmax(scores)
    return knowledge_base[best_index], scores[best_index]

question = "Does anyone have asthma?"
best_match, score = search(question, knowledge_base, knowledge_embeddings)

print("Best match:", best_match)
print("Score:", score)