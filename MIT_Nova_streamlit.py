import sys
import pysqlite3
sys.modules['sqlite3'] = pysqlite3

import streamlit as st
from groq import Groq
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.document_loaders import JSONLoader, PyPDFLoader
from langchain.text_splitter import TokenTextSplitter
import base64
import time
import os
from pathlib import Path

# Initialize Groq Client (use Streamlit secrets)
client = Groq(api_key=st.secrets["GROQ_API_KEY"])

# Database configuration
DB_CONFIG = {
    "json": {
        "data_path": "./company_policies.json",
        "db_path": "./streamlit_json_db",
        "jq_schema": ".[] | {question: .question, answer: .answer}"
    },
    "pdf": {
        "data_path": "./Employee_Handbook.pdf",
        "db_path": "./streamlit_pdf_db"
    }
}

# Initialize vector databases
@st.cache_resource(show_spinner=False)
def initialize_databases():
    """Initialize or load Chroma vector databases"""
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    databases = {}
    
    for db_type, config in DB_CONFIG.items():
        db_path = config["db_path"]
        data_path = config["data_path"]
        
        # Create directory if it doesn't exist
        os.makedirs(db_path, exist_ok=True)
        
        # Check if database exists
        if Path(db_path).exists() and any(Path(db_path).iterdir()):
            st.info(f"Loading existing {db_type.upper()} database...")
            databases[db_type] = Chroma(
                persist_directory=db_path,
                embedding_function=embeddings
            )
        else:
            try:
                st.info(f"Creating new {db_type.upper()} database...")
                
                # Load documents
                if db_type == "json":
                    loader = JSONLoader(
                        file_path=data_path,
                        jq_schema=config["jq_schema"],
                        text_content=False
                    )
                else:  # pdf
                    loader = PyPDFLoader(data_path)
                
                docs = loader.load()
                
                # Process documents
                splitter = TokenTextSplitter(chunk_size=300, chunk_overlap=40)
                chunks = splitter.split_documents(docs)
                
                # Create vector store
                databases[db_type] = Chroma.from_documents(
                    documents=chunks,
                    embedding=embeddings,
                    persist_directory=db_path
                )
                databases[db_type].persist()
                
            except Exception as e:
                st.error(f"Error creating {db_type} database: {str(e)}")
                st.stop()
    
    return databases["json"], databases["pdf"]

# RAG Function
def rag_query(json_db, pdf_db, user_query, threshold=0.2, k=7):
    try:
        # Search both databases
        json_hits = json_db.similarity_search_with_relevance_scores(user_query, k=k)
        pdf_hits = pdf_db.similarity_search_with_relevance_scores(user_query, k=k)
        
        # Filter by relevance threshold
        filtered_json = [doc for doc, score in json_hits if score >= threshold]
        filtered_pdf = [doc for doc, score in pdf_hits if score >= threshold]
        
        # Combine results
        all_docs = filtered_json + filtered_pdf
        if not all_docs:
            return "Sorry, I couldn't find relevant information. Please rephrase your question."
        
        # Generate context
        context = "\n".join([doc.page_content for doc in all_docs])
        
        # Query LLM
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
        "sidebar": "#3E4145" if dark_mode else "#898b8e",
        "text": "#ffffff" if dark_mode else "#343541",
        "text-secondary": "#acacbe" if dark_mode else "#565869",
        "card-user": "#40414f" if dark_mode else "#f7f7f8",
        "card-bot": "#444654" if dark_mode else "#ffffff",
        "border": "#565869" if dark_mode else "#e5e5e5",
        "button": "#10a37f" if dark_mode else "#10a37f",
        "button-hover": "#1a7f64" if dark_mode else "#0d8b6b",
        "input": "#40414f" if dark_mode else "#ffffff",
        "toggle-text": "#343541" if dark_mode else "#343541",
        "toggle-bg": "#565869" if dark_mode else "#e5e5e5",
        "logo-filter": "none",
        "input-cursor": "#ffffff" if dark_mode else "#10a37f",
        "input-selection-bg": "#444f60" if dark_mode else "#d2d6d4",
        "input-selection-text": "#ffffff" if dark_mode else "#212529",
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
            caret-color: var(--input-cursor) !important;
        }}
        .stTextInput>div>div>input::placeholder {{
            color: var(--text-secondary) !important;
            opacity: 1 !important;
        }}
        .stTextInput>div>div>input::selection {{
            background-color: var(--input-selection-bg) !important;
            color: var(--input-selection-text) !important;
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
        .stTextArea>div>textarea {{
            color: var(--text) !important;
            caret-color: var(--input-cursor) !important;
        }}
        .stTextArea>div>textarea::selection {{
            background-color: var(--input-selection-bg) !important;
            color: var(--input-selection-text) !important;
            }}
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
if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = True
if 'current_chat' not in st.session_state:
    st.session_state.current_chat = str(time.time())
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = {}
if 'conversation' not in st.session_state:
    st.session_state.conversation = []

# Initialize current chat if not exists
if st.session_state.current_chat not in st.session_state.chat_history:
    st.session_state.chat_history[st.session_state.current_chat] = []

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
        logo_html = f'<div style="background: white; padding: 5px; border-radius: 4px; display: inline-block;"><img src="data:image/png;base64,{logo}" alt="MIT Nova Logo" style="max-width: 100%; height: auto;"></div>'
except Exception as e:
    st.error(f"Error loading logo: {str(e)}")
    logo_html = '<h3>MIT Nova</h3>'

def create_new_chat():
    new_chat_id = str(time.time())
    st.session_state.current_chat = new_chat_id
    st.session_state.conversation = []
    st.session_state.chat_history[new_chat_id] = []

def switch_chat(chat_id):
    st.session_state.current_chat = chat_id
    st.session_state.conversation = st.session_state.chat_history[chat_id]

# ====== Sidebar ======
with st.sidebar:
    # Logo at the top of sidebar
    st.markdown(f"""
    <div class="sidebar-logo">
        {logo_html}
    </div>
    """, unsafe_allow_html=True)
   
    # New Chat Button
    if st.button("+ New chat", key="new_chat_button", use_container_width=True, type="primary"):
        create_new_chat()

    # Dark Mode Toggle
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
    st.markdown("#### Chat History:")

    # Custom CSS for chat history
    st.markdown("""
    <style>
        section[data-testid="stSidebar"] button.chat {
            border: 1px solid #e0e0e0 !important;
            border-radius: 8px !important;
            padding: 10px 14px !important;
            margin: 4px 0 !important;
            font-size: 15px !important;
            text-align: left !important;
            width: 100% !important;
            transition: background-color 0.2s !important;
        }
        
        section[data-testid="stSidebar"] button.chat:hover {
            background-color: #f5f5f5 !important;
        }
        
        section[data-testid="stSidebar"] button.active {
            border: 2px solid #10a37f !important;
            background-color: white !important;
            font-weight: 600 !important;
        }
    </style>
    """, unsafe_allow_html=True)

    # Display chat history
    for chat_id in reversed(list(st.session_state.chat_history.keys())):
        # Get chat title
        chat_title = next((msg[0][:30] + ("..." if len(msg[0]) > 30 else "") 
                        for msg in st.session_state.chat_history[chat_id] if msg[0]), "New chat")
        
        # Create button with custom class
        btn_class = "chat" + (" active" if st.session_state.current_chat == chat_id else "")
        st.button(
            chat_title,
            key=f"chat_{chat_id}",
            on_click=switch_chat,
            args=(chat_id,),
            use_container_width=True,
            help=btn_class
        )

# ====== Main Interface ======
# Chat header with title and description
st.markdown(f"""
<div class="chat-header">
    <h1>MIT Nova</h1>
    <p>Your AI assistant for company policies and HR information</p>
</div>
""", unsafe_allow_html=True)

# Initialize databases (with loading spinner)
with st.spinner("Loading databases..."):
    json_db, pdf_db = initialize_databases()

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

# Footer
st.markdown("""
<div style="text-align: center; color: var(--text-secondary); font-size: 12px; padding: 16px;">
    MIT Nova – A new star in internal assistance 🌟
</div>
""", unsafe_allow_html=True)