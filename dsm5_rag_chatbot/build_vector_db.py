from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# ---- Step 1: Load PDF ----
print("📘 Loading DSM-5-TR PDF...")
loader = PyPDFLoader(r"C:\Users\hp\Downloads\DSM-5-TR.pdf")
documents = loader.load()
print(f"✅ Total pages loaded: {len(documents)}")

# ---- Step 2: Split into Chunks ----
print("✂️ Splitting text into chunks...")
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(documents)
print(f"✅ Total chunks created: {len(chunks)}")

# ---- Step 3: Create Embeddings ----
print("🧠 Creating embeddings...")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# ---- Step 4: Build Vector DB ----
print("📦 Building FAISS vector database...")
db = FAISS.from_documents(chunks, embeddings)

# ---- Step 5: Save Database Locally ----
db.save_local("faiss_dsm5_index")
print("✅ FAISS vector database saved as 'faiss_dsm5_index/'")
