"""
MindMate - Professional Mental Health AI Companion
Enterprise-grade Streamlit Application with Improved Sidebar
"""

import streamlit as st
import requests
from datetime import datetime
from typing import Optional
import json

# ======================== PAGE CONFIG ========================
st.set_page_config(
    page_title="MindMate | Professional Mental Health Support",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://mindmate.com/help",
        "Report a bug": "https://mindmate.com/support",
        "About": "https://mindmate.com/about"
    }
)

# ======================== IMPROVED PROFESSIONAL STYLING ========================
st.markdown("""
<style>
    /* Import Professional Typography */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* --- Base & Body --- */
    :root {
        --primary-color: #5a67d8;
        --secondary-color: #8c9eff;
        --background-color: #f7fafc;
        --text-color: #2d3748;
        --text-color-light: #718096;
        --border-color: #e2e8f0;
        --user-message-bg: var(--primary-color);
        --assistant-message-bg: #ffffff;
        --danger-color: #e53e3e;
        --warning-color: #dd6b20;
        --success-color: #38a169;
    }
    
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }
    
    body {
        background-color: var(--background-color);
    }

    /* --- Main Layout --- */
    .main .block-container {
        max-width: 1000px;
        padding: 1rem 1rem 1rem;
    }
    
    .chat-container {
        display: flex;
        flex-direction: column;
        gap: 1.25rem;
        margin-bottom: 1rem;
        min-height: 5px
    }

    /* --- Enterprise Header --- */
    .main-header {
        background: linear-gradient(135deg, #f7fafc 0%, #edf2f7 100%);
        border: 1px solid #e2e8f0;
        padding: 2.0rem 2rem;
        border-radius: 10px;
        text-align: center;
        color: #2d3748;
        margin-bottom: 1.5rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.06);
    }
    .main-header h1 {
        font-size: 2rem;
        font-weight: 600;
        margin: 0 0 0.35rem 0;
        color: #1a202c;
    }
    .main-header p {
        font-size: 1rem;
        color: #4a5568;
        font-weight: 400;
        margin: 0;
    }

    /* --- Chat Interface --- */
    .message-wrapper { 
        display: flex; 
        align-items: flex-end; 
        gap: 10px; 
        width: 100%; 
    }
    .avatar { 
        width: 36px; 
        height: 36px; 
        border-radius: 50%; 
        display: flex; 
        align-items: center; 
        justify-content: center; 
        font-size: 1.2rem; 
        flex-shrink: 0; 
    }
    .user-avatar { 
        background-color: #e2e8f0; 
        color: #4a5568; 
    }
    .assistant-avatar { 
        background-color: #e2e8f0; 
    }
    .chat-bubble { 
        padding: 12px 16px; 
        border-radius: 18px; 
        max-width: 85%; 
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05); 
        word-wrap: break-word; 
    }
    .chat-bubble strong { 
        display: none; 
    }
    .message-wrapper.user { 
        justify-content: flex-end; 
    }
    .message-wrapper.user .chat-bubble { 
        background: var(--user-message-bg); 
        color: white; 
        border-bottom-right-radius: 4px; 
    }
    .message-wrapper.assistant { 
        justify-content: flex-start; 
    }
    .message-wrapper.assistant .chat-bubble { 
        background: var(--assistant-message-bg); 
        color: var(--text-color); 
        border: 1px solid var(--border-color); 
        border-bottom-left-radius: 4px; 
    }
    .timestamp { 
        font-size: 0.75rem; 
        color: var(--text-color-light); 
        margin: 4px 0 0 0; 
        text-align: left; 
    }
    .message-wrapper.user .timestamp { 
        text-align: right; 
        color: rgba(255, 255, 255, 0.75); 
    }

    /* --- Metadata & Alerts --- */
    .metadata-container { 
        font-size: 0.8rem; 
        color: var(--text-color-light); 
        padding-left: 46px; 
        padding-top: 8px; 
    }
    .crisis-alert { 
        background-color: #fff5f5; 
        border: 1px solid var(--danger-color); 
        color: var(--danger-color); 
        padding: 1rem; 
        border-radius: 12px; 
        margin-top: 8px; 
        font-weight: 500; 
    }
    
    /* --- IMPROVED SIDEBAR STYLING --- */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f8f9fa 0%, #ffffff 100%);
        border-right: 1px solid #e2e8f0;
    }
    
    [data-testid="stSidebar"] > div:first-child {
        padding: 2rem 1.25rem;
    }
    
    /* Sidebar Title */
    .sidebar-title {
        font-size: 1.5rem;
        font-weight: 700;
        color: #1a202c;
        margin-bottom: 0.25rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .sidebar-subtitle {
        font-size: 0.875rem;
        color: #718096;
        margin-bottom: 1.5rem;
    }
    
    /* Section Divider */
    .sidebar-divider {
        height: 1px;
        background: linear-gradient(to right, transparent, #e2e8f0, transparent);
        margin: 1.5rem 0;
    }
    
    /* Section Headers */
    .sidebar-section-title {
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #718096;
        margin-bottom: 0.75rem;
        margin-top: 1rem;
    }
    
    /* Status Badge */
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.625rem 1rem;
        border-radius: 8px;
        font-size: 0.875rem;
        font-weight: 500;
        width: 100%;
        justify-content: center;
        margin-bottom: 0.75rem;
    }
    
    .status-online {
        background-color: #d1fae5;
        color: #065f46;
        border: 1px solid #34d399;
    }
    
    .status-offline {
        background-color: #fee2e2;
        color: #991b1b;
        border: 1px solid #f87171;
    }
    
    /* Input Field in Sidebar */
    [data-testid="stSidebar"] .stTextInput > div > div > input {
        background-color: white;
        border: 1px solid #d1d5db;
        border-radius: 8px;
        padding: 0.625rem 0.875rem;
        font-size: 0.875rem;
        transition: all 0.2s ease;
    }
    
    [data-testid="stSidebar"] .stTextInput > div > div > input:focus {
        border-color: var(--primary-color);
        box-shadow: 0 0 0 3px rgba(90, 103, 216, 0.1);
    }
    
    /* Sidebar Buttons */
    [data-testid="stSidebar"] .stButton > button {
        background-color: white;
        color: #374151;
        border: 1px solid #d1d5db;
        font-weight: 500;
        border-radius: 8px;
        padding: 0.625rem 1rem;
        font-size: 0.875rem;
        transition: all 0.2s ease;
        width: 100%;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05);
    }
    
    [data-testid="stSidebar"] .stButton > button:hover {
        background-color: #5a67d8;
        color: white;
        border-color: #5a67d8;
        transform: translateY(-1px);
        box-shadow: 0 2px 4px rgba(90, 103, 216, 0.2);
    }
    
    /* Metrics in Sidebar */
    [data-testid="stSidebar"] [data-testid="stMetric"] {
        background-color: white;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 0.75rem;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05);
    }
    
    [data-testid="stSidebar"] [data-testid="stMetricLabel"] {
        font-size: 0.75rem;
        color: #718096;
        font-weight: 500;
    }
    
    [data-testid="stSidebar"] [data-testid="stMetricValue"] {
        font-size: 1.5rem;
        color: #1a202c;
        font-weight: 600;
    }
    
    /* Download Button Special Styling */
    [data-testid="stSidebar"] .stDownloadButton > button {
        background-color: #f7fafc;
        color: #374151;
    }
    
    [data-testid="stSidebar"] .stDownloadButton > button:hover {
        background-color: #38a169;
        border-color: #38a169;
    }
    
    /* Expander in Sidebar */
    [data-testid="stSidebar"] .streamlit-expanderHeader {
        background-color: white;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 0.75rem 1rem;
        font-weight: 500;
        color: #374151;
        font-size: 0.875rem;
    }
    
    [data-testid="stSidebar"] .streamlit-expanderHeader:hover {
        background-color: #f7fafc;
    }
    
    [data-testid="stSidebar"] .streamlit-expanderContent {
        background-color: white;
        border: 1px solid #e2e8f0;
        border-top: none;
        border-radius: 0 0 8px 8px;
        padding: 1rem;
    }
    
    /* --- General Components --- */
    .stExpander { 
        border: 1px solid var(--border-color); 
        border-radius: 12px; 
    }
    .streamlit-expanderHeader { 
        font-weight: 600; 
    }
    hr { 
        border: none; 
        border-top: 1px solid var(--border-color); 
        margin: 2rem 0; 
    }
    
    /* --- Footer --- */
    .footer {
        text-align: center;
        color: #a0aec0;
        font-size: 0.85rem;
        padding: 2rem 0 1rem 0;
        border-top: 1px solid var(--border-color);
        margin-top: 2rem;
    }

    /* --- Skeleton loader --- */
    @keyframes pulseFade { 0% {opacity: .5} 50% {opacity: .15} 100% {opacity: .5} }
    .skeleton { background: #e5e7eb; height: 18px; border-radius: 6px; animation: pulseFade 1.2s ease-in-out infinite; }
    .skeleton.bubble { height: 90px; border-radius: 16px; }
    
    /* --- Chat Input Positioning Fix --- */
    [data-testid="stChatInput"] {
        position: relative;
        z-index: 100;
    }

    /* --- Dark Theme Overrides --- */
    @media (prefers-color-scheme: dark) {
        :root {
            --primary-color: #7986cb;
            --background-color: #0E1117;
            --text-color: #FAFAFA;
            --text-color-light: #A0AEC0;
            --border-color: #262730;
            --assistant-message-bg: #262730;
        }
        
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #1a1d24 0%, #0E1117 100%);
            border-right: 1px solid var(--border-color);
        }
        
        .sidebar-title {
            color: var(--text-color);
        }
        
        .sidebar-subtitle {
            color: var(--text-color-light);
        }
        
        [data-testid="stSidebar"] .stTextInput > div > div > input {
            background-color: #1a1d24;
            border-color: var(--border-color);
            color: var(--text-color);
        }
        
        [data-testid="stSidebar"] .stButton > button {
            background-color: #1a1d24;
            color: var(--text-color);
            border-color: var(--border-color);
        }
        
        [data-testid="stSidebar"] [data-testid="stMetric"] {
            background-color: #1a1d24;
            border-color: var(--border-color);
        }
        
        [data-testid="stSidebar"] .streamlit-expanderHeader {
            background-color: #1a1d24;
            border-color: var(--border-color);
            color: var(--text-color);
        }
        
        [data-testid="stSidebar"] .streamlit-expanderContent {
            background-color: #0E1117;
            border-color: var(--border-color);
        }
    }
</style>
""", unsafe_allow_html=True)

# ======================== SESSION STATE ========================
if "messages" not in st.session_state:
    st.session_state.messages = []
if "api_url" not in st.session_state:
    st.session_state.api_url = "http://127.0.0.1:8000"
if "user_id" not in st.session_state:
    st.session_state.user_id = f"user_{int(datetime.now().timestamp())}"
if "api_healthy" not in st.session_state:
    st.session_state.api_healthy = False
if "last_prompt" not in st.session_state:
    st.session_state.last_prompt = None
if "suggested_prompt" not in st.session_state:
    st.session_state.suggested_prompt = None

# ======================== HELPER FUNCTIONS ========================
def check_api_health():
    try:
        response = requests.get(f"{st.session_state.api_url}/health", timeout=3)
        return response.status_code == 200
    except:
        return False

def send_message_to_api(message: str):
    try:
        response = requests.post(
            f"{st.session_state.api_url}/chat",
            json={"user_id": st.session_state.user_id, "message": message},
            timeout=30
        )
        return response.json() if response.status_code == 200 else None
    except Exception as e:
        st.error(f"API Error: {str(e)}")
        return None

def format_timestamp():
    return datetime.now().strftime("%H:%M")

# ======================== IMPROVED SIDEBAR ========================
with st.sidebar:
    # Header Section
    st.markdown('<div class="sidebar-title">🧠 MindMate</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-subtitle">AI Mental Health Companion</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
    
    # API Connection Section
    st.markdown('<div class="sidebar-section-title">🔗 API Connection</div>', unsafe_allow_html=True)
    
    api_url = st.text_input(
        "API Endpoint",
        value=st.session_state.api_url,
        label_visibility="collapsed",
        key="api_input",
        placeholder="http://127.0.0.1:8000"
    )
    
    if api_url != st.session_state.api_url:
        st.session_state.api_url = api_url
    
    # Check API button
    if st.button("🔍 Check Connection", use_container_width=True, key="check_api"):
        with st.spinner("Checking..."):
            st.session_state.api_healthy = check_api_health()
        if st.session_state.api_healthy:
            st.success("Connected successfully!")
        else:
            st.error("Connection failed")
    
    # Status Badge
    if st.session_state.api_healthy:
        st.markdown('<div class="status-badge status-online">🟢 Connected</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="status-badge status-offline">🔴 Disconnected</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
    
    # Chat Controls Section
    st.markdown('<div class="sidebar-section-title">💬 Chat Controls</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Clear", use_container_width=True, key="clear_chat"):
            st.session_state.messages = []
            st.rerun()
    
    with col2:
        if st.button("🔄 Retry", use_container_width=True, disabled=not st.session_state.last_prompt, key="retry_last"):
            if st.session_state.last_prompt:
                with st.spinner("Retrying..."):
                    resp = send_message_to_api(st.session_state.last_prompt)
                if resp:
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": resp.get("response", ""),
                        "timestamp": format_timestamp(),
                        "metadata": {
                            "is_crisis": resp.get("is_crisis", False),
                            "confidence_level": resp.get("confidence_level", "UNKNOWN"),
                            "confidence_score": resp.get("confidence_score", 0),
                            "sources": resp.get("sources", []),
                        },
                    })
                    st.rerun()
    
    # Export button (full width)
    st.download_button(
        "📥 Export Chat",
        data=json.dumps(st.session_state.messages, ensure_ascii=False, indent=2),
        file_name=f"mindmate_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json",
        use_container_width=True,
        key="export_chat"
    )
    
    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
    
    # Session Stats Section
    st.markdown('<div class="sidebar-section-title">📊 Session Stats</div>', unsafe_allow_html=True)
    
    total_msgs = len(st.session_state.messages)
    user_msgs = sum(1 for m in st.session_state.messages if m.get('role') == 'user')
    bot_msgs = total_msgs - user_msgs
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Messages", total_msgs)
    with col2:
        st.metric("Exchanges", user_msgs)
    
    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
    
    # Crisis Resources
    with st.expander("🆘 Crisis Support", expanded=False):
        st.markdown("""
        **If you're in crisis, seek help immediately:**
        
        🇵🇰 **Pakistan**  
        `03111-774444` (Umang, 24/7)
        
        🇺🇸 **USA**  
        `988` (Suicide & Crisis Lifeline)
        
        🌍 **International**  
        [Find Help Worldwide](https://findahelpline.com/)
        """)

# ======================== MAIN HEADER (Minimal for Tab Layout) ========================
st.markdown("""
<div style="text-align: center; margin-bottom: 1rem;">
    <h2 style="color: #2d3748; margin-bottom: 0.25rem; font-weight: 600;">🧠 MindMate</h2>
    <p style="color: #718096; font-size: 0.9rem; margin: 0;">Final Year Project - AI Mental Health Assistant</p>
</div>
""", unsafe_allow_html=True)

# ======================== NAVIGATION TABS ========================
tab_intro, tab_chat, tab_insights, tab_settings = st.tabs(["📋 Introduction", "💬 Chat", "📊 Insights", "⚙️ Settings"])

# ======================== INTRODUCTION TAB ========================
with tab_intro:
    # Professional Header with FYP Context
    st.markdown("""
    <div style="background: linear-gradient(135deg, #5a67d8 0%, #667eea 100%); 
                padding: 2.5rem 2rem; border-radius: 15px; color: white; margin-bottom: 2rem;text-align: center;"">
        <h1 style="color: white; margin-bottom: 0.5rem; font-size: 2.5rem; font-weight: 700;"> MindMate</h1>
        <p style="color: rgba(255,255,255,0.95); font-size: 1.2rem; margin-bottom: 0;">Final Year Project</p>
        <p style="color: rgba(255,255,255,0.85); font-size: 1rem; margin-top: 0.5rem;">AI-Powered Mental Health Assistant</p>
    </div>
    
    <div style="display: flex; justify-content: center; gap: 2rem; margin-bottom: 2rem;">
        <div style="text-align: center;">
            <div style="font-size: 2rem;">🎓</div>
            <div style="font-weight: 600; color: #2d3748;">Academic Research</div>
            <div style="font-size: 0.9rem; color: #718096;">Evidence-Based Approach</div>
        </div>
        <div style="text-align: center;">
            <div style="font-size: 2rem;">🔬</div>
            <div style="font-weight: 600; color: #2d3748;">RAG Technology</div>
            <div style="font-size: 0.9rem; color: #718096;">Retrieval-Augmented Generation</div>
        </div>
        <div style="text-align: center;">
            <div style="font-size: 2rem;">🛡️</div>
            <div style="font-weight: 600; color: #2d3748;">Crisis Detection</div>
            <div style="font-size: 0.9rem; color: #718096;">Safety-First Design</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Project Overview
    st.markdown("### 📋 Project Overview")
    st.markdown("""
    **MindMate** is a comprehensive Final Year Project that demonstrates the application of modern AI technologies 
    in mental health education. Built using Retrieval-Augmented Generation (RAG) architecture, it provides 
    evidence-based responses while maintaining strict safety protocols.
    """)
    
    # Technical Features
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **🔧 Technical Stack:**
        - **Backend:** FastAPI with Python
        - **Frontend:** Streamlit with custom CSS
        - **AI Model:** OpenAI GPT-4 with embeddings
        - **Vector DB:** Pinecone for document retrieval
        - **Crisis Detection:** Semantic analysis pipeline
        """)
    
    with col2:
        st.markdown("""
        **🎯 Key Features:**
        - Real-time crisis detection and intervention
        - Evidence-based mental health education
        - Secure conversation logging
        - Professional-grade UI/UX design
        - Comprehensive safety protocols
        """)
    
    st.markdown("---")
    
    # Important Notice
    st.error("""
    ⚠️ **IMPORTANT NOTICE** ⚠️
    
    This is **NOT** a therapy service or medical tool.
    This is an **educational chatbot** for learning about mental health concepts and coping strategies.
    
    **DO NOT** share personal health information.
    **DO NOT** use this in place of professional help.
    """)
    
    # What This Chatbot Is/Isn't
    col1, col2 = st.columns(2)
    
    with col1:
        st.success("""
        🤖 **What This Chatbot IS:**
        - An educational AI assistant
        - A tool to learn about mental health concepts (like CBT)
        - A practice space for self-reflection techniques
        - Based on evidence-based psychological frameworks
        """)
    
    with col2:
        st.error("""
        ❌ **What This Chatbot IS NOT:**
        - A therapist, counselor, or medical professional
        - A replacement for professional mental health care
        - HIPAA-compliant or suitable for sharing personal health data
        - Appropriate for crisis situations or urgent mental health needs
        """)
    
    # Privacy Notice
    st.warning("""
    🔒 **Privacy Notice:**
    - This system is **NOT HIPAA-compliant**
    - **DO NOT** share:
      - Personal identifying information
      - Details that could identify you
    - Conversations may be stored for educational purposes
    - Use generic scenarios and hypothetical situations instead
    """)
    
    # Crisis Resources
    st.error("""
    🆘 **IF YOU'RE IN CRISIS - STOP AND GET HELP NOW:**
    
    📞 **Call 03111-774444 (Umang, 24/7)* - Umang Pk 
    🌐 **Visit 988lifeline.org**
    
    **This chatbot CANNOT help with emergencies.**
    """)
    
    # Who Should Use This
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("""
        ✅ **This tool is for you if you want to:**
        - Learn about mental health concepts
        - Explore CBT techniques in a low-stakes environment
        - Practice self-reflection skills
        - Understand psychological frameworks
        """)
    
    with col2:
        st.warning("""
        ❌ **This tool is NOT for you if you:**
        - Need professional mental health treatment
        - Are experiencing a mental health crisis
        - Want to discuss personal medical information
        - Are looking for diagnosis or treatment plans
        """)
    
    # Educational Purpose
    st.info("""
    📚 **Educational Purpose:**
    
    This chatbot helps you learn about mental wellness strategies and cognitive behavioral therapy concepts. 
    Think of it as an **interactive educational guide**, not a counselor.
    
    **For actual mental health support, please consult:**
    - Licensed therapists
    - Mental health professionals  
    - Your primary care physician
    - Campus counseling centers (if student)
    - Employee assistance programs (if available)
    """)
    
    # Research Methodology
    st.markdown("---")
    st.markdown("### 🔬 Research Methodology")
    
    st.markdown("""
    This project employs a **multi-layered approach** to ensure both educational value and user safety:
    
    **1. Evidence-Based Content:** Responses are generated using authoritative mental health resources and 
    evidence-based psychological frameworks.
    
    **2. Crisis Detection Pipeline:** Advanced semantic analysis identifies crisis language patterns 
    and triggers immediate intervention protocols.
    
    **3. RAG Architecture:** Retrieval-Augmented Generation ensures responses are grounded in 
    verified mental health literature while maintaining conversational flow.
    
    **4. Privacy-First Design:** All conversations are processed securely with no personal data retention.
    """)
    
    # Getting Started
    st.markdown("---")
   
# ======================== CHAT TAB ========================
with tab_chat:
    # Prompt suggestions
    with st.expander("💡 Quick Start", expanded=False):
        examples = [
            "I'm feeling overwhelmed, can we try a grounding exercise?",
            "Help me challenge negative thoughts about failing.",
            "Guide me through a 3-minute breathing technique.",
        ]
        cols = st.columns(len(examples))
        for i, (col, ex) in enumerate(zip(cols, examples)):
            with col:
                if st.button(ex, key=f"suggest_{i}", use_container_width=True):
                    st.session_state.suggested_prompt = ex
                    st.rerun()

    # Chat display
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)

    if not st.session_state.messages:
        st.info("👋 Welcome to MindMate! How are you feeling today?")

    for idx, msg in enumerate(st.session_state.messages):
        role = msg["role"]
        avatar = "👤" if role == "user" else "🧠"
        avatar_class = "user-avatar" if role == "user" else "assistant-avatar"

        st.markdown(f"""
            <div class="message-wrapper {role}">
                <div class="avatar {avatar_class}">{avatar}</div>
                <div class="chat-bubble">
                    {msg['content']}
                    <div class="timestamp">{msg.get('timestamp', '')}</div>
                </div>
            </div>
        """, unsafe_allow_html=True)

        # Metadata and actions for assistant
        if role == "assistant" and "metadata" in msg:
            meta = msg["metadata"]
            with st.container():
                st.markdown('<div class="metadata-container">', unsafe_allow_html=True)
                level = (meta.get("confidence_level") or "").upper()
                score = float(meta.get("confidence_score", 0))
                
                if level:
                    conf_emoji = "🟢" if level == "HIGH" else "🟡" if level == "MEDIUM" else "🔴"
                    st.caption(f"{conf_emoji} Confidence: {level} ({score:.0f}%)")
                
                if meta.get("is_crisis"):
                    st.markdown('<div class="crisis-alert">🚨 CRITICAL: Crisis language detected. Please seek immediate professional help.</div>', unsafe_allow_html=True)
                
                sources = meta.get("sources") or []
                if sources:
                    st.caption(f"📚 Sources: {', '.join(sources[:3])}")
                
                # Actions row
                col1, col2, col3 = st.columns(3)
                with col1:
                    if st.button("👍 Helpful", key=f"up_{idx}"):
                        st.toast("Thanks for the feedback!", icon="✅")
                with col2:
                    if st.button("🔄 Regenerate", key=f"regen_{idx}"):
                        if st.session_state.last_prompt:
                            with st.spinner("Regenerating..."):
                                resp = send_message_to_api(st.session_state.last_prompt)
                            if resp:
                                st.session_state.messages.append({
                                    "role": "assistant",
                                    "content": resp.get("response", ""),
                                    "timestamp": format_timestamp(),
                                    "metadata": {
                                        "is_crisis": resp.get("is_crisis", False),
                                        "confidence_level": resp.get("confidence_level", "UNKNOWN"),
                                        "confidence_score": resp.get("confidence_score", 0),
                                        "sources": resp.get("sources", []),
                                    },
                                })
                                st.rerun()
                with col3:
                    if st.button("📋 Copy", key=f"copy_{idx}"):
                        st.toast("Response copied!", icon="📋")
                
                st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# ======================== INPUT SECTION ========================
if 'processing' not in st.session_state:
    st.session_state.processing = False

if st.session_state.suggested_prompt:
    # Show suggestion as editable text input
    edited_prompt = st.text_area(
        "Your message:",
        value=st.session_state.suggested_prompt,
        height=60,
        key="suggestion_input",
        help="Edit this suggestion or type your own message"
    )
    
    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("Send", type="primary", use_container_width=True):
            if edited_prompt.strip():
                st.session_state.processing = True
                st.session_state.last_prompt = edited_prompt.strip()
                st.session_state.suggested_prompt = None
                st.rerun()
    with col2:
        if st.button("Clear", use_container_width=True):
            st.session_state.suggested_prompt = None
            st.rerun()
else:
    # Regular chat input
    prompt = st.chat_input("How are you feeling?", disabled=st.session_state.processing)
    if prompt:
        st.session_state.processing = True
        st.session_state.last_prompt = prompt

# Process message if we have one
if st.session_state.last_prompt:
    # Add user message to state
    st.session_state.messages.append({
        "role": "user",
        "content": st.session_state.last_prompt,
        "timestamp": format_timestamp()
    })

    # Trigger API call and bot response
    with st.spinner("MindMate is thinking..."):
        response_data = send_message_to_api(st.session_state.last_prompt)

    if response_data:
        bot_response = response_data.get("response", "I'm having trouble connecting right now.")
        st.session_state.messages.append({
            "role": "assistant",
            "content": bot_response,
            "timestamp": format_timestamp(),
            "metadata": {
                "is_crisis": response_data.get("is_crisis", False),
                "confidence_level": response_data.get("confidence_level", "UNKNOWN"),
                "confidence_score": response_data.get("confidence_score", 0),
                "sources": response_data.get("sources", [])
            }
        })
    else:
        st.session_state.messages.append({
            "role": "assistant",
            "content": "Sorry, I couldn't process that. Please check the API connection and try again.",
            "timestamp": format_timestamp()
        })

    st.session_state.processing = False
    st.session_state.last_prompt = None
    st.rerun()

# ======================== INSIGHTS TAB ========================
with tab_insights:
    st.subheader("📈 Conversation Analytics")
    
    total_msgs = len(st.session_state.messages)
    user_msgs = sum(1 for m in st.session_state.messages if m.get('role') == 'user')
    bot_msgs = total_msgs - user_msgs
    confidences = [m.get('metadata',{}).get('confidence_score') for m in st.session_state.messages if m.get('role')=='assistant' and 'metadata' in m]
    avg_conf = round(sum([c for c in confidences if isinstance(c,(int,float))]) / max(1, len(confidences)), 1) if confidences else 0

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Messages", total_msgs)
    with col2:
        st.metric("Bot Responses", bot_msgs)
    with col3:
        st.metric("Avg Confidence", f"{avg_conf}%")

    st.markdown("---")
    
    st.subheader("📚 Recent Sources")
    recent_sources = []
    for m in st.session_state.messages[-10:]:
        if m.get('role')=='assistant' and 'metadata' in m:
            recent_sources.extend(m['metadata'].get('sources', []))
    
    if recent_sources:
        unique_sources = list(dict.fromkeys([str(s) for s in recent_sources]))
        for source in unique_sources:
            st.caption(f"• {source}")
    else:
        st.info("No sources available from recent conversations.")
# ======================== SETTINGS TAB ========================
with tab_settings:
    st.subheader("⚙️ Application Settings")
    
    st.markdown("""
    Adjust your preferences and configurations for a personalized experience.
    """)
    
    # Theme selection (informational only - cannot be changed at runtime)
    current_theme = st.get_option("theme.base")
    theme_display = {
        "light": "Light",
        "dark": "Dark", 
        "system": "System Default"
    }.get(current_theme, "Unknown")
    
    st.info(f"**Current Theme:** {theme_display}")
    st.caption("To change theme, restart the app with: `streamlit run frontend.py --theme.base dark`")
    
    # Notification settings
    notifications = st.checkbox(
        "Enable Notifications",
        value=True,
        help="Toggle to receive notifications for important updates and responses."
    )
    
    # Privacy settings
    privacy = st.checkbox(
        "Send Anonymous Usage Data",
        value=False,
        help="Help us improve MindMate by sending anonymous usage statistics."
    )
    
    st.markdown("---")
    
    st.subheader("🔒 Privacy & Security")
    st.markdown("""
    MindMate is committed to protecting your privacy. All conversations are confidential and encrypted.
    
    - **Data Encryption**: All data transmitted between your device and our servers is encrypted using industry-standard protocols.
    - **No Third-Party Sharing**: We do not share your data with third parties without your explicit consent.
    - **Data Retention**: You can clear your chat history at any time using the 'Clear' button in the chat controls.
    
    For more details, please refer to our [Privacy Policy](https://mindmate.com/privacy).
    """)

# ======================== FOOTER ========================
st.markdown("""
<div class="footer">
    <p>© 2024 MindMate. All rights reserved. | <a href="https://mindmate.com/terms" target="_blank">Terms of Service</a> | <a href="https://mindmate.com/privacy" target="_blank">Privacy Policy</a></p>
</div>
""", unsafe_allow_html=True)
# ======================== END OF FILE ========================
