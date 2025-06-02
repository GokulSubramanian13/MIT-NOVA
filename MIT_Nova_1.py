
import streamlit as st
from groq import Groq
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
import base64
import time
import os

# Initialize Groq Client
client = Groq(api_key="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")

# Load vector DBs
try:
    json_db = Chroma(
        persist_directory="./json_db", 
        embedding_function=SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    )
    pdf_db = Chroma(
        persist_directory="./pdf_db", 
        embedding_function=SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    )
except Exception as e:
    st.error(f"Error loading databases: {str(e)}")
    st.stop()

# RAG Function
def rag_query(json_db, pdf_db, user_query, threshold=0.2, k=7):
    try:
        json_hits = json_db.similarity_search_with_relevance_scores(user_query, k=k)
        filtered_json = [doc for doc, score in json_hits if score >= threshold]

        pdf_hits = pdf_db.similarity_search_with_relevance_scores(user_query, k=k)
        filtered_pdf = [doc for doc, score in pdf_hits if score >= threshold]

        all_docs = filtered_json + filtered_pdf
        if not all_docs:
            return "Sorry, I couldn't find relevant information. Please rephrase your question."

        context = "\n".join([doc.page_content for doc in all_docs])

        response = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[
                {"role": "system", "content": "Answer professionally based on the context."},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {user_query}"}
            ],
            temperature=0.3,
            max_tokens=1000
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"An error occurred: {str(e)}"

# Theme Setup
def set_custom_theme(dark_mode):
    theme = {
        "bg": "#343541" if dark_mode else "#ffffff",
        "sidebar": "#202123" if dark_mode else "#f7f7f8",
        "text": "#ffffff" if dark_mode else "#343541",
        "text-secondary": "#acacbe" if dark_mode else "#565869",
        "card-user": "#40414f" if dark_mode else "#f7f7f8",
        "card-bot": "#444654" if dark_mode else "#ffffff",
        "border": "#565869" if dark_mode else "#e5e5e5",
        "button": "#10a37f" if dark_mode else "#10a37f",
        "button-hover": "#1a7f64" if dark_mode else "#0d8b6b",
        "input": "#40414f" if dark_mode else "#ffffff",
        "toggle-text": "#ffffff" if dark_mode else "#343541",  # Fixed toggle text color
        "toggle-bg": "#565869" if dark_mode else "#e5e5e5",  # Added toggle background
        "logo-filter": "invert(100%)" if not dark_mode else "none",
    }
    
    st.markdown(f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@600&display=swap');
        
        :root {{
            --bg-color: {theme["bg"]};
            --sidebar-color: {theme["sidebar"]};
            --text-color: {theme["text"]};
            --text-secondary: {theme["text-secondary"]};
            --card-user: {theme["card-user"]};
            --card-bot: {theme["card-bot"]};
            --border-color: {theme["border"]};
            --button-bg: {theme["button"]};
            --button-hover: {theme["button-hover"]};
            --input-bg: {theme["input"]};
            --toggle-text: {theme["toggle-text"]};
            --toggle-bg: {theme["toggle-bg"]};
            --logo-filter: {theme["logo-filter"]};
        }}
        .stApp {{
            background-color: var(--bg-color) !important;
        }}
        .stTextInput>div>div>input {{
            background-color: var(--input-bg) !important;
            color: var(--text-color) !important;
            border: 1px solid var(--border-color) !important;
            border-radius: 6px !important;
            padding: 12px !important;
            box-shadow: none !important;
        }}
        .stTextInput>div>div>input::placeholder {{
            color: var(--text-secondary) !important;
            opacity: 1 !important;
        }}
        .stButton>button {{
            background-color: var(--button-bg) !important;
            color: white !important;
            border-radius: 6px !important;
            transition: all 0.3s !important;
            border: none !important;
        }}
        .stButton>button:hover {{
            background-color: var(--button-hover) !important;
            transform: none !important;
            box-shadow: none !important;
        }}
        [data-testid="stSidebar"] {{
            background-color: var(--sidebar-color) !important;
        }}
        .chat-message {{
            padding: 24px;
            color: var(--text-color);
            display: flex;
            max-width: 800px;
            margin: 0 auto;
        }}
        .chat-message-user {{
            background-color: var(--card-user);
            border-top: 1px solid var(--border-color);
            border-bottom: 1px solid var(--border-color);
        }}
        .chat-message-bot {{
            background-color: var(--card-bot);
            border-bottom: 1px solid var(--border-color);
        }}
        .chat-message-content {{
            max-width: 700px;
            margin: 0 auto;
            padding-left: 72px;
        }}
        .chat-message-avatar {{
            width: 36px;
            height: 36px;
            border-radius: 2px;
            display: flex;
            align-items: center;
            justify-content: center;
            margin-right: 16px;
            flex-shrink: 0;
        }}
        .chat-message-avatar-user {{
            background-color: #ab68ff;
            color: white;
        }}
        .chat-message-avatar-bot {{
            background-color: #10a37f;
            color: white;
        }}
        .new-chat-btn {{
            border: 1px solid var(--border-color) !important;
            margin-bottom: 16px !important;
        }}
        .history-item {{
            padding: 8px 12px;
            border-radius: 4px;
            margin: 4px 0;
            cursor: pointer;
            font-size: 14px;
            color: var(--text-secondary) !important;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
            transition: all 0.2s;
            background-color: transparent;
        }}
        .history-item:hover {{
            background-color: var(--card-user) !important;
            color: var(--text-color) !important;
        }}
        .history-item.active {{
            background-color: var(--card-user) !important;
            color: var(--text-color) !important;
        }}
        .history-item::selection {{
            background: var(--button-bg);
            color: white;
        }}
        .spinner {{
            margin: 0 auto;
            width: 24px;
            height: 24px;
            border: 3px solid rgba(255,255,255,.3);
            border-radius: 50%;
            border-top-color: #fff;
            animation: spin 1s ease-in-out infinite;
        }}
        .sidebar-logo {{
            padding: 16px 0;
            margin-bottom: 16px;
            text-align: center;
            border-bottom: 1px solid var(--border-color);
        }}
        .sidebar-logo img {{
            max-width: 80%;
            height: auto;
            filter: var(--logo-filter);
            background-color: white;
            padding: 5px;
            border-radius: 4px;
        }}
        /* Toggle fixes */
        .stToggle label p {{
            color: var(--toggle-text) !important;
            font-size: 14px !important;
        }}
        .stToggle button {{
            background-color: var(--toggle-bg) !important;
        }}
        .stToggle button:hover {{
            background-color: var(--toggle-bg) !important;
        }}
        /* Header styling */
        .chat-header {{
            text-align: center;
            padding: 16px 0;
            margin-bottom: 16px;
            border-bottom: 1px solid var(--border-color);
        }}
        .chat-header h1 {{
            color: var(--text-color);
            font-size: 24px;
            margin: 0;
            font-family: 'Poppins', sans-serif;
            font-weight: 600;
            letter-spacing: 0.5px;
        }}
        .chat-header p {{
            color: var(--text-secondary);
            font-size: 14px;
            margin: 4px 0 0;
        }}
        @keyframes spin {{
            to {{ transform: rotate(360deg); }}
        }}
    </style>
    """, unsafe_allow_html=True)

# ====== Session State ======
if "conversation" not in st.session_state:
    st.session_state.conversation = []
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False
if "current_chat" not in st.session_state:
    st.session_state.current_chat = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = {}

# ====== Page Config ======
st.set_page_config(
    page_title="MIT Nova",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply theme
set_custom_theme(st.session_state.dark_mode)

# Load logo
try:
    with open("logo.png", "rb") as image_file:
        logo = base64.b64encode(image_file.read()).decode()
    logo_html = f'<img src="data:image/png;base64,{logo}" alt="MIT Nova Logo">'
except Exception as e:
    st.error(f"Error loading logo: {str(e)}")
    logo_html = '<h3>MIT Nova</h3>'

# ====== Sidebar ======
with st.sidebar:
    # Logo at the top of sidebar
    st.markdown(f"""
    <div class="sidebar-logo">
        {logo_html}
    </div>
    """, unsafe_allow_html=True)
    
    # New Chat Button
    if st.button("+ New chat", key="new_chat", use_container_width=True, type="primary"):
        st.session_state.current_chat = str(time.time())
        st.session_state.conversation = []
        st.session_state.chat_history[st.session_state.current_chat] = []
    
    # Dark Mode Toggle - properly styled
    new_dark_mode = st.toggle(
        "Dark mode", 
        value=st.session_state.dark_mode,
        key="dark_mode_toggle"
    )
    if new_dark_mode != st.session_state.dark_mode:
        st.session_state.dark_mode = new_dark_mode
        st.rerun()
    
    # Conversation History
    st.markdown("---")
    st.markdown("#### Chats")
    
    # If no current chat, create one
    if not st.session_state.current_chat:
        st.session_state.current_chat = str(time.time())
        st.session_state.chat_history[st.session_state.current_chat] = []
    
    # Display chat history
    for chat_id in reversed(list(st.session_state.chat_history.keys())):
        # Get first non-empty message for title
        chat_title = "New chat"
        for msg in st.session_state.chat_history[chat_id]:
            if msg[0]:  # user message
                chat_title = msg[0][:30] + ("..." if len(msg[0]) > 30 else "")
                break
        
        # Create clickable chat item
        if st.session_state.current_chat == chat_id:
            st.markdown(f'<div class="history-item active">{chat_title}</div>', unsafe_allow_html=True)
        else:
            if st.markdown(f'<div class="history-item">{chat_title}</div>', unsafe_allow_html=True):
                st.session_state.current_chat = chat_id
                st.session_state.conversation = st.session_state.chat_history[chat_id]
                st.rerun()

# ====== Main Interface ======
# Chat header with title and description
st.markdown(f"""
<div class="chat-header">
    <h1>MIT Nova</h1>
    <p>Your AI assistant for company policies and HR information</p>
</div>
""", unsafe_allow_html=True)

# Chat container
chat_container = st.container()

# Display conversation
with chat_container:
    for i, (query, answer) in enumerate(st.session_state.conversation):
        # User message
        st.markdown(f"""
        <div class="chat-message chat-message-user">
            <div class="chat-message-avatar chat-message-avatar-user">U</div>
            <div class="chat-message-content">{query}</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Bot message
        st.markdown(f"""
        <div class="chat-message chat-message-bot">
            <div class="chat-message-avatar chat-message-avatar-bot">N</div>
            <div class="chat-message-content">{answer}</div>
        </div>
        """, unsafe_allow_html=True)

# Input area at bottom
with st.container():
    st.markdown("<div style='height: 100px;'></div>", unsafe_allow_html=True)  # Spacer
    
    # Chat input form
    with st.form("chat_input_form", clear_on_submit=True):
        col1, col2 = st.columns([6, 1])
        with col1:
            query = st.text_input(
                "Message MIT Nova...",
                placeholder="Ask about company policies or HR information...",
                label_visibility="collapsed",
                key="query_input"
            )
        with col2:
            submit_button = st.form_submit_button("Send", use_container_width=True)
    
    if submit_button and query.strip():
        # Add user message to conversation
        st.session_state.conversation.append((query, ""))
        st.session_state.chat_history[st.session_state.current_chat] = st.session_state.conversation
        
        # Display user message immediately
        with chat_container:
            st.markdown(f"""
            <div class="chat-message chat-message-user">
                <div class="chat-message-avatar chat-message-avatar-user">U</div>
                <div class="chat-message-content">{query}</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Display typing indicator
        with chat_container:
            typing_indicator = st.markdown("""
            <div class="chat-message chat-message-bot">
                <div class="chat-message-avatar chat-message-avatar-bot">N</div>
                <div class="chat-message-content">
                    <div class="spinner"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Get response
        answer = rag_query(json_db, pdf_db, query)
        
        # Update conversation
        st.session_state.conversation[-1] = (query, answer)
        st.session_state.chat_history[st.session_state.current_chat] = st.session_state.conversation
        
        # Remove typing indicator and display answer
        typing_indicator.empty()
        with chat_container:
            st.markdown(f"""
            <div class="chat-message chat-message-bot">
                <div class="chat-message-avatar chat-message-avatar-bot">N</div>
                <div class="chat-message-content">{answer}</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Rerun to update the chat history sidebar
        st.rerun()

# Footer
st.markdown("""
<div style="text-align: center; color: var(--text-secondary); font-size: 12px; padding: 16px;">
    MIT Nova may produce inaccurate information about people, places, or facts.
</div>
""", unsafe_allow_html=True)
 
