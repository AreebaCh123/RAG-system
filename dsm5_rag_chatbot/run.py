from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatOpenAI   # ✅ updated import

import os
os.environ["OPENAI_API_KEY"] = "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"
os.environ["OPENAI_API_BASE"] = "https://openrouter.ai/api/v1"

# Load FAISS
print("📂 Loading FAISS vector database...")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local("faiss_dsm5_index", embeddings, allow_dangerous_deserialization=True)
print("✅ FAISS database loaded successfully!")

# ✅ use ChatOpenAI (modern version)
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.2
)

# Create retriever
retriever = db.as_retriever(search_kwargs={"k": 3})

# Build RAG chain
rag_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff"
)

# Chat loop
print("\n💬 DSM-5 RAG Chatbot Ready! Ask your questions below:")
while True:
    query = input("\nYou: ")
    if query.lower() in ["exit", "quit", "bye"]:
        print("👋 Goodbye!")
        break
    response = rag_chain.invoke({"query": query})   # ✅ use .invoke() instead of .run()
    print(f"🤖 Chatbot: {response['result']}")
