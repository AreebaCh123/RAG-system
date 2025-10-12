
import os
import hashlib
from tqdm import tqdm
from pinecone import Pinecone, ServerlessSpec
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from concurrent.futures import ThreadPoolExecutor, as_completed

from config import (
    DATA_PATH,
    INDEX_NAME,
    EMBEDDING_MODEL,
    EMBEDDING_DIMENSION,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    BATCH_SIZE,
    MAX_WORKERS,
    PINECONE_CLOUD,
    PINECONE_REGION)

def generate_chunk_id(document_source, chunk_text):
    hasher = hashlib.sha256()
    hasher.update(f"{document_source}{chunk_text}".encode('utf-8'))
    return hasher.hexdigest()

def process_batch(batch_chunks, index, embeddings):
    chunk_ids = [generate_chunk_id(chunk.metadata.get("source", ""), chunk.page_content) for chunk in batch_chunks]
    
    try:
        existing_ids_response = index.fetch(ids=chunk_ids)
        # Handle different response formats
        if hasattr(existing_ids_response, 'vectors'):
            existing_ids = set(existing_ids_response.vectors.keys())
        elif hasattr(existing_ids_response, 'get'):
            existing_ids = set(existing_ids_response.get('vectors', {}).keys())
        else:
            existing_ids = set()
    except Exception as e:
        print(f"Error fetching existing IDs: {e}")
        return 0

    chunks_to_upsert = []
    for chunk, chunk_id in zip(batch_chunks, chunk_ids):
        if chunk_id not in existing_ids:
            chunk.metadata["chunk_id"] = chunk_id
            chunks_to_upsert.append(chunk)

    if not chunks_to_upsert:
        return 0

    try:
        embedded_texts = embeddings.embed_documents([chunk.page_content for chunk in chunks_to_upsert])
        
        vectors_to_upsert = []
        for chunk, embedding in zip(chunks_to_upsert, embedded_texts):
            vector = {
                "id": chunk.metadata["chunk_id"],
                "values": embedding,
                "metadata": {
                    "text": chunk.page_content,
                    "source": chunk.metadata.get("source", "Unknown"),
                    "page": chunk.metadata.get("page", 0)
                }
            }
            vectors_to_upsert.append(vector)
            
        index.upsert(vectors=vectors_to_upsert)
        return len(vectors_to_upsert)
    except Exception as e:
        print(f"Error embedding or upserting batch: {e}")
        return 0

def ingest_data():
    # --- 1. Load & 2. Split Documents ---
    print("Loading and splitting documents...")
    loader = DirectoryLoader(DATA_PATH, glob="*.pdf", loader_cls=PyPDFLoader, show_progress=True)
    documents = loader.load()
    if not documents:
        print(f"No documents found in '{DATA_PATH}'. Please add your PDF files.")
        return
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = text_splitter.split_documents(documents)
    print(f"Loaded and split documents into {len(chunks)} chunks.")

    # --- 3. Initialize Embeddings and Pinecone ---
    print("Initializing embeddings model and Pinecone connection...")
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    
    if INDEX_NAME not in pc.list_indexes().names():
        print(f"Creating new Pinecone index: '{INDEX_NAME}'")
        pc.create_index(
            name=INDEX_NAME,
            dimension=EMBEDDING_DIMENSION,
            metric='cosine',
            spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION)
        )
    
    index = pc.Index(INDEX_NAME)
    print("Embeddings and Pinecone initialized.")
    
    batches = [chunks[i:i + BATCH_SIZE] for i in range(0, len(chunks), BATCH_SIZE)]

    total_upserted_count = 0
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_batch = {executor.submit(process_batch, batch, index, embeddings): batch for batch in batches}
        for future in tqdm(as_completed(future_to_batch), total=len(batches), desc="Processing batches"):
            try:
                upserted_count = future.result()
                total_upserted_count += upserted_count
            except Exception as exc:
                print(f'A batch generated an exception: {exc}')

    print(f" Data ingestion complete. Upserted {total_upserted_count} new chunks.")

if __name__ == "__main__":
    ingest_data()