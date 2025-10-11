import os
from dotenv import load_dotenv

load_dotenv()

DATA_PATH = "data/"
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "mindmate")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", 1536))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1000))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 200))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 100))
MAX_WORKERS = int(os.getenv("MAX_WORKERS", 5))
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "aws")
PINECONE_REGION = os.getenv("PINECONE_ENVIRONMENT", "us-east-1")