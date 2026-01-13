from http import client
import os
import numpy as np
from qdrant_client import models
from qdrant_client.http.models import ScoredPoint
from sentence_transformers import SentenceTransformer
from ..core import get_rag_client
from dotenv import load_dotenv

load_dotenv()

class MedicalNativeRAG:
    def __init__(self, embedder):
        self.rag_client = get_rag_client()
        self.collection_name = os.getenv("COLLECTION_NAME")
        # self.model_name = cfg.RAG_EMBEDDING_MODEL_NAME
        self.model = embedder
        
    def query(self, query: str, limit=5):
        embeddings = self.model.encode([query], convert_to_numpy=True).tolist()[0]
        hits = self.rag_client.search(
            collection_name=self.collection_name,
            query_vector=embeddings,
            limit=limit
        )
        return hits
    
class MedicalRerankRAG:
    def __init__(self, embedder, reranker):
        self.rag_client = get_rag_client()
        self.collection_name = os.getenv("COLLECTION_NAME")
        self.embedder = embedder
        self.reranker = reranker
        self.limit = 20
        self.top_k = 5

    # -------------------------
    # 1. Dense retrieval
    # -------------------------
    def retrieve(self, query: str, limit):
        query_embedding = self.embedder.encode(
            [query], convert_to_numpy=True
        ).tolist()[0]

        return self.rag_client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=limit
        )
        
    # -------------------------
    # 2. Re-ranking
    # -------------------------
    def rerank(self, query: str, hits, top_k=5):
        if not hits:
            return []

        pairs = [(query, h.payload["text"]) for h in hits]
        scores = self.reranker.predict(pairs)

        scored_hits = []
        for h, s in zip(hits, scores):
            final_score = s + 0.2 * h.score
            scored_hits.append((h, final_score))

        scored_hits.sort(key=lambda x: x[1], reverse=True)
        return [h for h, _ in scored_hits[:top_k]]

    # -------------------------
    # 3. Main query
    # -------------------------
    def query(self, query: str, limit1, limit2):
        hits = self.retrieve(query, limit=limit1)

        if not hits:
            return []

        reranked_hits = self.rerank(query, hits, limit2)
        return  reranked_hits

