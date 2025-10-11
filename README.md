# 🧠 MindMate - Mental Health RAG System

A sophisticated Retrieval-Augmented Generation (RAG) system designed for mental health support, featuring advanced document retrieval, reranking, and confidence scoring capabilities.

## ✨ Features

- **Therapeutic AI**: Evidence-based mental health support using DSM-5-TR and other clinical documents
- **Advanced Retrieval**: Dynamic k-value selection and cross-encoder reranking for better relevance
- **Confidence Scoring**: Multi-factor confidence assessment with hallucination detection
- **Streaming Responses**: Real-time response generation for better user experience
- **Conversation Logging**: Automatic chat history logging with confidence metrics
- **Crisis Awareness**: Built-in safety features and appropriate disclaimers

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- OpenAI API key
- Pinecone API key (for vector database)
- LangSmith API key (optional, for tracing)

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd RAG-system
```

### 2. Install Dependencies

```bash
# Install all required packages
pip install -r requirements.txt

# Optional: Install sentence-transformers for enhanced reranking
pip install sentence-transformers
```

### 3. Environment Configuration

Copy the environment template and configure your API keys:

```bash
# Copy the template file
cp env.template .env

# Edit the .env file with your actual API keys
# Use your preferred text editor (nano, vim, VS Code, etc.)
```

Or create a `.env` file manually with the following variables:

```bash
# Required API Keys
OPENAI_API_KEY=your_openai_api_key_here
PINECONE_API_KEY=your_pinecone_api_key_here

# Optional: LangSmith for tracing
LANGSMITH_API_KEY=your_langsmith_api_key_here

# Pinecone Configuration
PINECONE_INDEX_NAME=mindmate
PINECONE_CLOUD=aws
PINECONE_ENVIRONMENT=us-east-1

# Embedding Configuration
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_DIMENSION=1536

# Document Processing
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
BATCH_SIZE=100
MAX_WORKERS=5
```

### 4. Add Your Documents

Place your PDF documents in the `data/` directory. The system will automatically process them during ingestion.

### 5. Ingest Documents

```bash
python src/ingest.py
```

This will:
- Load and split your PDF documents
- Create embeddings using OpenAI
- Store vectors in Pinecone
- Handle duplicate detection

### 6. Run the Chatbot

```bash
python main.py
```

Or run the enhanced chatbot directly:

```bash
python src/chatbot.py
```

## 📁 Project Structure

```
RAG-system/
├── data/                          # Place your PDF documents here
│   └── DSM-5-TR.pdf              # Example document
├── src/
│   ├── chatbot.py                # Enhanced chatbot with streaming
│   ├── config.py                 # Configuration settings
│   ├── ingest.py                 # Document ingestion pipeline
│   ├── prompt.py                 # System prompts and templates
│   └── pipeline/                 # Additional pipeline components
├── chat_logs/                    # Conversation logs (auto-generated)
├── main.py                       # Main entry point
├── requirements.txt              # Python dependencies
├── env.template                  # Environment variables template
└── README.md                     # This file
```



### Advanced Features

#### Reranking (Optional)
Install sentence-transformers for enhanced document reranking:
```bash
pip install sentence-transformers
```

#### LangSmith Tracing (Optional)
Set up LangSmith for request tracing and debugging:
1. Get API key from [LangSmith](https://smith.langchain.com/)
2. Add `LANGSMITH_API_KEY` to your `.env` file

## 🛠️ Development

### Testing Your Setup

Run the verification script to ensure everything is configured correctly:

```bash
python main.py
```

This will:
- Check for required API keys
- Test OpenAI connection
- Verify Pinecone connectivity
- Run a sample LLM call

### Adding New Documents

1. Place PDF files in the `data/` directory
2. Run the ingestion script:
   ```bash
   python src/ingest.py
   ```
3. The system will automatically detect and process new documents

### Customizing the System

- **Prompts**: Modify `src/prompt.py` for different response styles
- **Retrieval**: Adjust parameters in `src/config.py`
- **Models**: Change LLM or embedding models in the respective files

## 🔒 Security & Privacy

- API keys are stored in `.env` file (not committed to git)
- Conversation logs are stored locally
- No data is sent to external services except OpenAI and Pinecone
- All processing happens in your local environment

## ⚠️ Important Disclaimers

- **Not a replacement for therapy**: This system provides support but is not a substitute for professional mental health care
- **Crisis situations**: For emergencies, contact local emergency services
- **Data privacy**: Be mindful of sensitive information in conversations
- **API costs**: Monitor your OpenAI and Pinecone usage

## 🐛 Troubleshooting

### Common Issues

1. **"OPENAI_API_KEY not found"**
   - Ensure your `.env` file exists and contains the correct API key
   - Check that the `.env` file is in the project root directory

2. **"Pinecone index not found"**
   - Run the ingestion script first: `python src/ingest.py`
   - Check your Pinecone API key and region settings

3. **"No documents found"**
   - Add PDF files to the `data/` directory
   - Ensure files have `.pdf` extension

4. **Import errors**
   - Install missing dependencies: `pip install -r requirements.txt`
   - For reranking: `pip install sentence-transformers`

### Getting Help

- Check the console output for detailed error messages
- Verify all API keys are correct and have proper permissions
- Ensure your Python environment has all required packages

## 📊 Performance Tips

- Use smaller `CHUNK_SIZE` for more precise retrieval
- Increase `BATCH_SIZE` for faster ingestion
- Enable reranking for better relevance (requires sentence-transformers)
- Monitor API usage to manage costs

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is for educational and research purposes. Please ensure compliance with OpenAI and Pinecone terms of service.

---

**Remember**: This system is designed to provide support and information, but it is not a replacement for professional mental health care. Always consult with qualified healthcare providers for serious mental health concerns.
