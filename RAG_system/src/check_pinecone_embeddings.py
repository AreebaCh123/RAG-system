from pinecone import Pinecone
import os
from dotenv import load_dotenv
load_dotenv()

# Initialize Pinecone connection
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("mindmate")  # replace with your actual index name

# Fetch some vectors
response = index.fetch(ids=["some_vector_id"])  # if you know specific IDs

print(response)
print(pc.list_indexes().names())       # lists all indexes
index = pc.Index("mindmate")    # pick the correct one
print(index.describe_index_stats())    # check stored vectors
