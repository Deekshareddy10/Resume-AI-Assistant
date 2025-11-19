import faiss
import numpy as np
from utils.embeddings import model

def build_faiss_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index

def query_faiss(index, chunks, question):
    q_emb = model.encode([question])
    D, I = index.search(q_emb, k=3)
    return [chunks[i] for i in I[0]]
