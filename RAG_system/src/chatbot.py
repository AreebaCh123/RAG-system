import os
import re
from datetime import datetime
from typing import List, Dict
from dotenv import load_dotenv


from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.documents import Document

from prompt import build_system_prompt, CONTEXTUALIZE_PROMPT
import numpy as np

from config import INDEX_NAME, EMBEDDING_MODEL
from twilio.rest import Client

# ======================== CrossEncoder (Optional) ========================
try:
    from sentence_transformers import CrossEncoder
    CROSSENCODER_AVAILABLE = True
except ImportError:
    CROSSENCODER_AVAILABLE = False
    print("⚠️  Warning: sentence_transformers not available. Reranking disabled.")
    print("   Run: pip install --upgrade sentence-transformers")

# Load .env file
load_dotenv()

# Twilio credentials
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_PHONE_NUMBER = os.getenv("TWILIO_PHONE_NUMBER")

# Initialize Twilio client
twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
# Example user profile
users = {
    "user1": {
        "name": "Alice",
        "emergency_contacts": ["+923295502805"]
    }
}

# ======================== Crisis Detection ========================
def detect_crisis_intent(user_input: str) -> bool:
    """Return True only if message clearly expresses self-harm intent."""
    text = user_input.lower()
    crisis_patterns = [
        r"\bi want to (kill|hurt|harm) myself\b",
        r"\bi (feel|am) suicidal\b",
        r"\bi can't go on\b",
        r"\bi don't want to live\b",
        r"\bi wish i were dead\b",
        r"\bi am going to end my life\b",
        r"\bi plan to (kill|hurt|harm) myself\b"
    ]
    for pattern in crisis_patterns:
        if re.search(pattern, text):
            # Exclude negated statements like "not harm myself"
            if "not" in text or "never" in text:
                return False
            return True
    return False


def send_sms(phone_number: str, message: str):
    """Send SMS using Twilio."""
    twilio_client.messages.create(
        body=message,
        from_=TWILIO_PHONE_NUMBER,
        to=phone_number
    )


def handle_crisis(user_id: str, message: str) -> str:
    """Send alerts to emergency contacts and provide support options."""
    for contact in users[user_id]["emergency_contacts"]:
        send_sms(contact, f"⚠️ ALERT: User '{user_id}' may be in crisis. Message: '{message}'")

    options = ["Call Emergency Helpline", "Contact Saved Helpline", "Chat with AI for support"]
    return f"It seems you are in distress. Please consider these options: {options}"

# ======================== Advanced RAG Retriever ========================
class AdvancedRAGRetriever:
    def __init__(self, vectorstore, base_k=10, use_reranking=True):
        self.vectorstore = vectorstore
        self.base_k = base_k
        self.use_reranking = use_reranking and CROSSENCODER_AVAILABLE
        if self.use_reranking:
            print(" Loading cross-encoder model...")
            self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
            print("✅ Cross-encoder loaded")
        else:
            self.reranker = None
            if use_reranking and not CROSSENCODER_AVAILABLE:
                print("ℹ Reranking requested but CrossEncoder not available. Using basic retrieval.")

    def calculate_dynamic_k(self, query: str) -> int:
        word_count = len(query.split())
        question_words = ['how', 'why', 'what', 'when', 'where', 'explain', 'describe']
        has_question_words = any(word in query.lower() for word in question_words)
        has_multiple_concepts = ',' in query or ' and ' in query or ' or ' in query
        k = 5
        if word_count > 15:
            k += 3
        elif word_count > 10:
            k += 2
        if has_question_words:
            k += 2
        if has_multiple_concepts:
            k += 3
        return max(3, min(12, k))

    def rerank_documents(self, query: str, documents: List[Document], top_k=4) -> List[Document]:
        if not documents or not self.use_reranking or self.reranker is None:
            return documents[:top_k]
        pairs = [[query, doc.page_content] for doc in documents]
        scores = self.reranker.predict(pairs)
        doc_score_pairs = list(zip(documents, scores))
        doc_score_pairs.sort(key=lambda x: x[1], reverse=True)
        reranked_docs = []
        for doc, score in doc_score_pairs[:top_k]:
            doc.metadata['rerank_score'] = float(score)
            reranked_docs.append(doc)
        return reranked_docs

    def retrieve(self, query: str) -> List[Document]:
        dynamic_k = self.calculate_dynamic_k(query)
        retriever = self.vectorstore.as_retriever(search_kwargs={'k': dynamic_k})
        documents = retriever.invoke(query)
        final_k = min(4, len(documents))
        return self.rerank_documents(query, documents, top_k=final_k)


# ======================== Response Validator ========================
class ResponseValidator:
    def __init__(self, llm):
        self.llm = llm

    def check_hallucination(self, response: str, context_docs: List[Document]) -> Dict:
        context_text = "\n".join([doc.page_content for doc in context_docs])
        validation_prompt = f"""You are a fact-checker. Analyze if the RESPONSE contains information that is NOT supported by the CONTEXT.

CONTEXT:
{context_text[:2000]}  

RESPONSE:
{response}

Analyze:
1. Are there specific claims in the response not found in the context?
2. Does the response add medical advice or diagnoses not in the context?
3. Does the response invent statistics, studies, or facts?

Respond in this exact format:
HALLUCINATION: [YES/NO]
CONFIDENCE: [0-100]
ISSUES: [List specific unsupported claims, or "None"]
"""
        try:
            validation_response = self.llm.invoke(validation_prompt)
            return self._parse_validation_response(validation_response.content)
        except Exception:
            return {"hallucination": False, "confidence": 50, "issues": []}

    def _parse_validation_response(self, response: str) -> Dict:
        hallucination = "YES" in response.split("HALLUCINATION:")[1].split("\n")[0].upper()
        confidence_match = re.search(r'CONFIDENCE:\s*(\d+)', response)
        confidence = int(confidence_match.group(1)) if confidence_match else 50
        try:
            issues_text = response.split("ISSUES:")[1].strip()
            issues = [issues_text] if issues_text.lower() != "none" else []
        except:
            issues = []
        return {
            "hallucination": hallucination,
            "confidence": confidence,
            "issues": issues
        }

    def calculate_confidence_score(self, response: str, context_docs: List[Document], validation_result: Dict) -> Dict:
        scores = []
        factors = {}
        rerank_scores = [doc.metadata.get('rerank_score', 0) for doc in context_docs]
        if rerank_scores and any(rerank_scores):
            avg_rerank = np.mean(rerank_scores)
            rerank_confidence = min(100, max(0, (avg_rerank + 5) * 10))
            scores.append(rerank_confidence)
            factors['rerank_score'] = round(rerank_confidence, 2)
        total_context_length = sum(len(doc.page_content) for doc in context_docs)
        context_confidence = min(100, (total_context_length / 500) * 100)
        scores.append(context_confidence)
        factors['context_coverage'] = round(context_confidence, 2)
        validation_confidence = validation_result.get('confidence', 50)
        scores.append(validation_confidence)
        factors['validation_score'] = validation_confidence
        hedge_words = ['might', 'could', 'possibly', 'perhaps', 'may', 'suggest', 'consider']
        hedge_count = sum(response.lower().count(word) for word in hedge_words)
        hedge_penalty = min(20, hedge_count * 5)
        factors['hedge_penalty'] = hedge_penalty
        base_score = np.mean(scores) if scores else 50
        final_score = max(0, base_score - hedge_penalty)
        return {
            'overall_confidence': round(final_score, 2),
            'factors': factors,
            'level': self._get_confidence_level(final_score)
        }

    def _get_confidence_level(self, score: float) -> str:
        if score >= 80:
            return "HIGH"
        elif score >= 60:
            return "MEDIUM"
        elif score >= 40:
            return "LOW"
        else:
            return "VERY_LOW"


# ======================== Initialize RAG Chain ========================
def get_enhanced_conversational_rag_chain(use_reranking=True):
    print("🔗 Initializing enhanced conversational RAG chain...")
    streaming_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3, streaming=True)
    validation_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, streaming=False)
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    vectorstore = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings)
    advanced_retriever = AdvancedRAGRetriever(vectorstore, use_reranking=use_reranking)
    validator = ResponseValidator(validation_llm)
    print("✅ Enhanced conversational RAG chain initialized successfully.")
    return advanced_retriever, streaming_llm, validator


def stream_response_with_validation(query: str, chat_history: list, retriever, llm, validator, past_conversations: str = "") -> Dict:
    context_docs = retriever.retrieve(query)
    context_text = "\n\n---\n\n".join([f"[Source: {doc.metadata.get('source', 'Unknown')}]\n{doc.page_content}" for doc in context_docs])
    system_prompt = build_system_prompt(context_text)

    if chat_history:
        previous_context = "\n".join([f"User: {msg.content}" if isinstance(msg, HumanMessage) else f"MindMate: {msg.content}" for msg in chat_history[-10:]])
        system_prompt += f"\n\nCurrent Session Context:\n{previous_context}\n"

    if past_conversations:
        max_past_length = 3000
        truncated_past = past_conversations
        if len(past_conversations) > max_past_length:
            truncated_past = past_conversations[-max_past_length:] + "\n... (truncated for length)"
        system_prompt += f"\n\nPast Conversation History:\n{truncated_past}\n"

    messages = [{"role": "system", "content": system_prompt}]
    for msg in chat_history:
        if isinstance(msg, HumanMessage):
            messages.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage):
            messages.append({"role": "assistant", "content": msg.content})
    messages.append({"role": "user", "content": query})

    print("\n🤖 MindMate: ", end="", flush=True)
    full_response = ""
    try:
        for chunk in llm.stream(messages):
            if chunk.content:
                print(chunk.content, end="", flush=True)
                full_response += chunk.content
        print()
    except Exception as e:
        print(f"\n❌ Streaming error: {e}")
        response = llm.invoke(messages)
        full_response = response.content
        print(f"\n🤖 MindMate: {full_response}")

    validation_result = validator.check_hallucination(full_response, context_docs)
    confidence_result = validator.calculate_confidence_score(full_response, context_docs, validation_result)

    return {
        "answer": full_response,
        "context_docs": context_docs,
        "validation": validation_result,
        "confidence": confidence_result
    }


# ======================== Load Previous Chat Logs ========================
def load_all_chat_logs() -> tuple:
    chat_history = []
    past_conversations = ""
    log_dir = "chat_logs"
    files_loaded = 0
    total_messages = 0
    
    if not os.path.exists(log_dir):
        return chat_history, past_conversations, files_loaded, total_messages

    txt_files = [f for f in os.listdir(log_dir) if f.endswith(".txt")]
    txt_files.sort(key=lambda x: os.path.getctime(os.path.join(log_dir, x)))
    
    for file in txt_files:
        file_path = os.path.join(log_dir, file)
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                file_content = f.read()
                past_conversations += f"\n\n--- Conversation from {file} ---\n{file_content}"
                for line in file_content.split('\n'):
                    if line.startswith("You: "):
                        content = line.replace("You: ", "").strip()
                        if content:
                            chat_history.append(HumanMessage(content=content))
                            total_messages += 1
                    elif line.startswith("MindMate: "):
                        content = line.replace("MindMate: ", "").strip()
                        if content:
                            chat_history.append(AIMessage(content=content))
                            total_messages += 1
                files_loaded += 1
        except Exception as e:
            print(f"⚠️ Warning: Could not read {file}: {e}")
    
    return chat_history, past_conversations, files_loaded, total_messages


# ======================== Main Chat Loop ========================
if __name__ == "__main__":
    retriever, llm, validator = get_enhanced_conversational_rag_chain(use_reranking=True)
    chat_history, past_conversations, files_loaded, total_messages = load_all_chat_logs()
    GLOBAL_PAST_CONVERSATIONS = past_conversations

    print("\n" + "="*70)
    print("🧠 MindMate - Your Mental Health Companion")
    print("="*70)
    rerank_status = " Enabled" if CROSSENCODER_AVAILABLE else "❌ Disabled"
    print(f"✨ Features: Therapeutic AI | Evidence-Based Support | Crisis Awareness")
    print(f"🔄 Reranking: {rerank_status}")
    print(f"💾 Memory Loaded: {files_loaded} files, {total_messages} messages")
    print("\n💡 This chatbot provides support but is NOT a replacement for therapy.")
    print("Type 'exit' to end the conversation.\n")

    os.makedirs("chat_logs", exist_ok=True)
    log_file = f"chat_logs/conversation_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.txt"

    while True:
        user_input = input("\n💬 You: ").strip()
        if user_input.lower() in ['exit', 'quit', 'bye']:
            print("\n🌿 MindMate: Thank you for sharing with me today. Remember, you're not alone. Take care. 💙")
            break
        if not user_input:
            continue

        try:
            # ===== Crisis Detection =====
            if detect_crisis_intent(user_input):
                answer = handle_crisis("user1", user_input)
                validation = {"hallucination": False}
                confidence = {"overall_confidence": 100, "level": "HIGH"}
            else:
                result = stream_response_with_validation(user_input, chat_history, retriever, llm, validator, GLOBAL_PAST_CONVERSATIONS)
                answer = result['answer']
                validation = result['validation']
                confidence = result['confidence']

            confidence_emoji = "🟢" if confidence['level'] == "HIGH" else "🟡" if confidence['level'] == "MEDIUM" else "🔴"
            print(f"\n{confidence_emoji} Confidence: {confidence['level']} ({confidence['overall_confidence']}%)")
            if validation.get('hallucination'):
                print(f"⚠️  Warning: Potential unsupported claims detected")

            with open(log_file, "a", encoding="utf-8") as f:
                f.write(f"You: {user_input}\n")
                f.write(f"MindMate: {answer}\n")
                f.write(f"Confidence: {confidence['level']} ({confidence['overall_confidence']}%)\n")
                f.write(f"Hallucination Check: {'FLAGGED' if validation.get('hallucination') else '✅ PASSED'}\n")
                f.write("-"*70 + "\n\n")

            chat_history.append(HumanMessage(content=user_input))
            chat_history.append(AIMessage(content=answer))
            if len(chat_history) > 50:
                chat_history = chat_history[-50:]

        except Exception as e:
            print(f"\n❌ MindMate: Sorry, I'm having trouble processing that right now.")
            print(f"   Debug: {str(e)}")
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(f"ERROR: {e}\n\n")
