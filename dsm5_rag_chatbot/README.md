# DSM-5-TR RAG Chatbot

A **Retrieval-Augmented Generation (RAG)** chatbot built using **FAISS**, **LangChain**, and **HuggingFace embeddings**, designed to answer questions based on DSM-5-TR content. This project demonstrates how a basic RAG system works by combining vector databases and language models for intelligent retrieval-based responses.

---

## Features

- **RAG Workflow:** Combines document retrieval and language model generation.
- **Vector Database:** Uses **FAISS** for efficient similarity search.
- **Embeddings:** Utilizes **HuggingFace `all-MiniLM-L6-v2` embeddings**.
- **LLM Integration:** Works with **OpenAI / OpenRouter GPT-4o-mini**.
- **Interactive Chat:** Provides a simple chat loop interface for user queries.
- **Educational Purpose:** Helps understand basic RAG systems workflow.

---

## Installation

1. **Clone the repository:**

```bash
git clone https://github.com/your-username/dsm5_rag_chatbot.git
cd dsm5_rag_chatbot
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
pip install -r requirements.txt
