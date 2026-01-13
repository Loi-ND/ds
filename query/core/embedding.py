from sentence_transformers import SentenceTransformer
from sentence_transformers import CrossEncoder


def get_embedding_model(model_name: str="AITeamVN/Vietnamese_Embedding_v2"):
    return SentenceTransformer(model_name)

def get_cross_encoder(model_name: str="BAAI/bge-reranker-v2-m3", max_length: int=512):
    return CrossEncoder(
            model_name,
            max_length=max_length
        )