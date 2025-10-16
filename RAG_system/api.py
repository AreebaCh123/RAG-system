"""
FastAPI Server for MindMate Chatbot
Integrates with existing chatbot.py in src/
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

print("\n" + "="*70)
print("🔄 INITIALIZING MINDMATE API")
print("="*70)

# Initialize FastAPI
app = FastAPI(
    title="MindMate API",
    description="Mental Health Support Chatbot",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ======================== DATA MODELS ========================
class ChatRequest(BaseModel):
    user_id: str = "user1"
    message: str
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    confidence_level: str
    confidence_score: float
    is_crisis: bool
    sources: List[str] = []

class HealthResponse(BaseModel):
    status: str
    version: str
    chatbot_ready: bool
    error: Optional[str] = None

# ======================== INITIALIZE CHATBOT ========================
CHATBOT_READY = False
retriever = None
llm = None
validator = None
chat_history = []
past_conversations = ""
CHATBOT_ERROR = None

print("\n📦 Step 1: Importing chatbot functions...")
try:
    from chatbot import (
        get_enhanced_conversational_rag_chain,
        detect_crisis_intent_semantic,
        stream_response_with_validation,
        load_all_chat_logs,
    )
    print("✅ Successfully imported chatbot functions")
except ImportError as e:
    print(f"❌ IMPORT ERROR: {e}")
    CHATBOT_ERROR = str(e)

if not CHATBOT_ERROR:
    print("\n📚 Step 2: Loading RAG chain (this may take 30-60 seconds)...")
    try:
        retriever, llm, validator = get_enhanced_conversational_rag_chain(use_reranking=True)
        print("✅ RAG chain loaded")
    except Exception as e:
        print(f"❌ RAG CHAIN ERROR: {e}")
        CHATBOT_ERROR = str(e)

if not CHATBOT_ERROR:
    print("\n📖 Step 3: Loading chat history...")
    try:
        chat_history, past_conversations, files_loaded, total_messages = load_all_chat_logs()
        print(f"✅ Chat history loaded ({files_loaded} files, {total_messages} messages)")
    except Exception as e:
        print(f"❌ CHAT HISTORY ERROR: {e}")
        CHATBOT_ERROR = str(e)

if not CHATBOT_ERROR:
    CHATBOT_READY = True
    print("\n" + "="*70)
    print("✅ CHATBOT FULLY INITIALIZED AND READY")
    print("="*70 + "\n")
else:
    print("\n" + "="*70)
    print(f"⚠️ CHATBOT NOT READY: {CHATBOT_ERROR}")
    print("="*70 + "\n")

# ======================== ENDPOINTS ========================

@app.get("/", tags=["Info"])
async def root():
    """Welcome endpoint"""
    return {
        "name": "MindMate API",
        "version": "1.0.0",
        "docs": "/docs",
        "status": "running",
        "chatbot_ready": CHATBOT_READY
    }

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Check API and chatbot status"""
    status = "🟢 OK" if CHATBOT_READY else "🟡 DEGRADED"
    return HealthResponse(
        status=status,
        version="1.0.0",
        chatbot_ready=CHATBOT_READY,
        error=CHATBOT_ERROR
    )

@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(request: ChatRequest):
    """Chat with MindMate"""
    
    if not CHATBOT_READY:
        raise HTTPException(
            status_code=503,
            detail=f"Chatbot not initialized: {CHATBOT_ERROR}"
        )
    
    if not request.message or not request.message.strip():
        raise HTTPException(
            status_code=400,
            detail="Message cannot be empty"
        )
    
    try:
        user_msg = request.message.strip()
        print(f"\n📨 Processing: {user_msg[:50]}...")
        
        # Crisis detection
        is_crisis = detect_crisis_intent_semantic(user_msg)
        
        if is_crisis:
            print("🚨 CRISIS DETECTED!")
            return ChatResponse(
                response=(
                    "🚨 I hear that you're going through something really difficult.\n\n"
                    "You are NOT alone - your life matters.\n\n"
                    "**Please reach out immediately:**\n\n"
                    "🇵🇰 Pakistan: **03111-774444** (Umang, 24/7)\n"
                    "🇺🇸 USA: **988** (Crisis Lifeline)\n"
                    "🌍 Text **HOME** to **741741**"
                ),
                confidence_level="HIGH",
                confidence_score=100.0,
                is_crisis=True,
                sources=[]
            )
        
        # Get response
        print("🤖 Generating response...")
        result = stream_response_with_validation(
            user_msg,
            chat_history,
            retriever,
            llm,
            validator,
            past_conversations
        )
        
        response_text = result.get('answer', 'I apologize, I could not generate a response.')
        confidence = result.get('confidence', {})
        context_docs = result.get('context_docs', [])
        
        sources = list(set([
            doc.metadata.get('source', 'Unknown')
            for doc in context_docs[:3]
        ]))
        
        print(f"✅ Response generated")
        
        return ChatResponse(
            response=response_text,
            confidence_level=confidence.get('level', 'UNKNOWN'),
            confidence_score=float(confidence.get('overall_confidence', 0.0)),
            is_crisis=False,
            sources=sources
        )
        
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# ======================== RUN ========================
if __name__ == "__main__":
    import uvicorn
    
    print("\n" + "="*70)
    print("🚀 STARTING FASTAPI SERVER")
    print("="*70)
    print(f"📝 Swagger Docs: http://127.0.0.1:8000/docs")
    print(f"🔗 API URL: http://127.0.0.1:8000")
    print("="*70 + "\n")
    
    try:
        uvicorn.run(
            app,
            host="127.0.0.1",
            port=8000,
            reload=False,  # Disabled reload to avoid re-initialization issues
            log_level="info"
        )
    except Exception as e:
        print(f"\n❌ FAILED TO START SERVER: {e}")
        print(f"\nTroubleshooting:")
        print(f"1. Port 8000 already in use? Kill it: lsof -ti:8000 | xargs kill -9")
        print(f"2. Missing dependencies? pip install fastapi uvicorn")
        print(f"3. Check .env file exists and has required keys")