import os
from xmlrpc import client
from qdrant_client import QdrantClient, models
from dotenv import load_dotenv

load_dotenv()

def get_rag_client():
    client = QdrantClient(url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_API_KEY"), timeout=60)
    return client

if __name__ == "__main__":
    rag_client = get_rag_client()
    
    print(rag_client.get_collections()) 