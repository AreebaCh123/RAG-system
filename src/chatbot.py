import os
import re
from datetime import datetime
from typing import List, Dict, Iterator
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

try:
    from sentence_transformers import CrossEncoder
    CROSSENCODER_AVAILABLE = True
except ImportError:
    CROSSENCODER_AVAILABLE = False
    print("⚠️  Warning: sentence_transformers not available. Reranking disabled.")
    print("   Run: pip install --upgrade sentence-transformers")


class AdvancedRAGRetriever:
    """Enhanced retriever with dynamic k-value and reranking capabilities"""
    
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
        """
        Rerank documents using cross-encoder for better relevance
        """
        if not documents or not self.use_reranking or self.reranker is None:
            return documents[:top_k]
        
        # Prepare pairs for cross-encoder
        pairs = [[query, doc.page_content] for doc in documents]
        
        # Get relevance scores
        scores = self.reranker.predict(pairs)
        
        # Sort documents by score
        doc_score_pairs = list(zip(documents, scores))
        doc_score_pairs.sort(key=lambda x: x[1], reverse=True)
        
        # Add scores to metadata
        reranked_docs = []
        for doc, score in doc_score_pairs[:top_k]:
            doc.metadata['rerank_score'] = float(score)
            reranked_docs.append(doc)
            
        return reranked_docs
    
    def retrieve(self, query: str) -> List[Document]:
        """
        Main retrieval method with dynamic k and reranking
        """
        dynamic_k = self.calculate_dynamic_k(query)
        
        retriever = self.vectorstore.as_retriever(search_kwargs={'k': dynamic_k})
        documents = retriever.invoke(query)
        
        # Rerank to get top most relevant
        final_k = min(4, len(documents))
        reranked_docs = self.rerank_documents(query, documents, top_k=final_k)
        
        return reranked_docs


class ResponseValidator:
    """Validates responses to detect hallucinations and assess confidence"""
    
    def __init__(self, llm):
        self.llm = llm
        
    def check_hallucination(self, response: str, context_docs: List[Document]) -> Dict:
        """
        Detect potential hallucinations by checking if response claims are grounded in context
        """
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
            result = self._parse_validation_response(validation_response.content)
            return result
        except Exception as e:
            return {"hallucination": False, "confidence": 50, "issues": []}
    
    def _parse_validation_response(self, response: str) -> Dict:
        """Parse the validation response"""
        hallucination = "YES" in response.split("HALLUCINATION:")[1].split("\n")[0].upper()
        
        try:
            confidence_match = re.search(r'CONFIDENCE:\s*(\d+)', response)
            confidence = int(confidence_match.group(1)) if confidence_match else 50
        except:
            confidence = 50
            
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
    
    def calculate_confidence_score(self, response: str, context_docs: List[Document], 
                                   validation_result: Dict) -> Dict:
        """
        Calculate overall confidence score based on multiple factors
        """
        scores = []
        factors = {}
        
        # Factor 1: Reranking scores
        rerank_scores = [doc.metadata.get('rerank_score', 0) for doc in context_docs]
        if rerank_scores and any(rerank_scores):
            avg_rerank = np.mean(rerank_scores)
            rerank_confidence = min(100, max(0, (avg_rerank + 5) * 10))
            scores.append(rerank_confidence)
            factors['rerank_score'] = round(rerank_confidence, 2)
        
        # Factor 2: Context relevance
        total_context_length = sum(len(doc.page_content) for doc in context_docs)
        context_confidence = min(100, (total_context_length / 500) * 100)
        scores.append(context_confidence)
        factors['context_coverage'] = round(context_confidence, 2)
        
        # Factor 3: Validation confidence
        validation_confidence = validation_result.get('confidence', 50)
        scores.append(validation_confidence)
        factors['validation_score'] = validation_confidence
        
        # Factor 4: Hedging language detection
        hedge_words = ['might', 'could', 'possibly', 'perhaps', 'may', 'suggest', 'consider']
        hedge_count = sum(response.lower().count(word) for word in hedge_words)
        hedge_penalty = min(20, hedge_count * 5)
        factors['hedge_penalty'] = hedge_penalty
        
        # Calculate final score
        base_score = np.mean(scores) if scores else 50
        final_score = max(0, base_score - hedge_penalty)
        
        return {
            'overall_confidence': round(final_score, 2),
            'factors': factors,
            'level': self._get_confidence_level(final_score)
        }
    
    def _get_confidence_level(self, score: float) -> str:
        """Convert numeric score to confidence level"""
        if score >= 80:
            return "HIGH"
        elif score >= 60:
            return "MEDIUM"
        elif score >= 40:
            return "LOW"
        else:
            return "VERY_LOW"



def get_enhanced_conversational_rag_chain(use_reranking=True):
    """
    Creates and returns an enhanced conversational RAG chain with streaming support.
    """
    print("🔗 Initializing enhanced conversational RAG chain...")

    streaming_llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.3, 
        streaming=True
    )
    
    # Non-streaming LLM for validation
    validation_llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        streaming=False
    )
    
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

    # Initialize enhanced retriever
    vectorstore = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings)
    advanced_retriever = AdvancedRAGRetriever(vectorstore, use_reranking=use_reranking)
    
    # Initialize validator
    validator = ResponseValidator(validation_llm)

    print("✅ Enhanced conversational RAG chain initialized successfully.")
    return advanced_retriever, streaming_llm, validator


def stream_response_with_validation(query: str, chat_history: list, 
                                    retriever, llm, validator) -> Dict:
    """
    Process query with streaming and post-validation using therapeutic prompts
    """
    # Step 1: Retrieve documents with dynamic k and reranking
    context_docs = retriever.retrieve(query)
    
    # Step 2: Build context string from retrieved documents
    context_text = "\n\n---\n\n".join([
        f"[Source: {doc.metadata.get('source', 'Unknown')}]\n{doc.page_content}"
        for doc in context_docs
    ])
    
    # Step 3: Build complete therapeutic system prompt
    system_prompt = build_system_prompt(context_text)
    
    # Step 4: Prepare messages for LLM
    messages = [
        {"role": "system", "content": system_prompt}
    ]
    
    # Add chat history
    for msg in chat_history:
        if isinstance(msg, HumanMessage):
            messages.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage):
            messages.append({"role": "assistant", "content": msg.content})
    
    # Add current query
    messages.append({"role": "user", "content": query})
    
    # Step 5: Stream the response
    print("\n🤖 MindMate: ", end="", flush=True)
    
    full_response = ""
    try:
        # Stream tokens
        for chunk in llm.stream(messages):
            if chunk.content:
                print(chunk.content, end="", flush=True)
                full_response += chunk.content
        
        print()  # New line after streaming
        
    except Exception as e:
        print(f"\n❌ Streaming error: {e}")
        # Fallback to non-streaming
        response = llm.invoke(messages)
        full_response = response.content
        print(f"\n🤖 MindMate: {full_response}")
    
    # Step 6: Post-streaming validation
    print("\n🔍 Analyzing response...", end="", flush=True)
    
    validation_result = validator.check_hallucination(full_response, context_docs)
    confidence_result = validator.calculate_confidence_score(
        full_response, context_docs, validation_result
    )
    
    print("\r" + " " * 30 + "\r", end="", flush=True)  # Clear message
    
    return {
        "answer": full_response,
        "context_docs": context_docs,
        "validation": validation_result,
        "confidence": confidence_result
    }

if __name__ == "__main__":
    # Initialize enhanced system with streaming
    retriever, llm, validator = get_enhanced_conversational_rag_chain(use_reranking=True)
    chat_history = []
    
    print("\n" + "="*70)
    print("🧠 MindMate - Your Mental Health Companion")
    print("="*70)
    rerank_status = " Enabled" if CROSSENCODER_AVAILABLE else "❌ Disabled (install sentence-transformers)"
    print(f"✨ Features: Therapeutic AI | Evidence-Based Support | Crisis Awareness")
    print(f"🔄 Reranking: {rerank_status}")
    print(f"📊 Confidence Scoring | Hallucination Detection")
    print("\n💡 This chatbot provides support but is NOT a replacement for therapy.")
    print("Type 'exit' to end the conversation.\n")
    
    # Create logs directory
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
            # Process with streaming and validation
            result = stream_response_with_validation(
                user_input, chat_history, retriever, llm, validator
            )
            
            answer = result['answer']
            confidence = result['confidence']
            validation = result['validation']
            
            # Display confidence information
            confidence_emoji = "🟢" if confidence['level'] == "HIGH" else "🟡" if confidence['level'] == "MEDIUM" else "🔴"
            print(f"\n{confidence_emoji} Confidence: {confidence['level']} ({confidence['overall_confidence']}%)")
            
            # Warning if hallucination detected
            if validation['hallucination']:
                print(f"⚠️  Warning: Potential unsupported claims detected")
            
            # Log to file
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(f"You: {user_input}\n")
                f.write(f"MindMate: {answer}\n")
                f.write(f"Confidence: {confidence['level']} ({confidence['overall_confidence']}%)\n")
                f.write(f"Hallucination Check: {'FLAGGED' if validation['hallucination'] else '✅ PASSED'}\n")
                f.write("-" * 70 + "\n\n")
            
            # Update chat history
            chat_history.append(HumanMessage(content=user_input))
            chat_history.append(AIMessage(content=answer))
            
            # Trim history to prevent memory issues
            if len(chat_history) > 20:
                chat_history = chat_history[-20:]
                
        except Exception as e:
            print(f"\n❌ MindMate: Sorry, I'm having trouble processing that right now.")
            print(f"   Debug: {str(e)}")
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(f"ERROR: {e}\n\n")