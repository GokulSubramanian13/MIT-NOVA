import streamlit as st
from groq import Groq
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
import base64
import time

# Initialize Groq Client
client = Groq(api_key="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")

# Load vector DBs
json_db = Chroma(persist_directory="./json_db", embedding_function=SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2"))
pdf_db = Chroma(persist_directory="./pdf_db", embedding_function=SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2"))

# RAG Function
def rag_query(json_db, pdf_db, user_query, threshold=0.2, k=7):
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

# Theme Setup
def set_custom_theme(dark_mode):
    theme = {
        "bg": "#0e1117" if dark_mode else "#ffffff",
        "text": "#f0f2f6" if dark_mode else "#2a3f5f",
        "card": "#1e2130" if dark_mode else "#f8f9fa",
        "border": "#2d3746" if dark_mode else "#e0e0e0",
        "button": "#6366f1" if dark_mode else "#4f46e5",
        "button_hover": "#4f46e5" if dark_mode else "#4338ca"
    }
    
    st.markdown(f"""
    <style>
        :root {{
            --bg-color: {theme["bg"]};
            --text-color: {theme["text"]};
            --card-bg: {theme["card"]};
            --border-color: {theme["border"]};
            --button-bg: {theme["button"]};
            --button-hover: {theme["button_hover"]};
        }}
        .stApp {{
            background-color: var(--bg-color) !important;
            color: var(--text-color) !important;
        }}
        .stTextInput>div>div>input {{
            background-color: var(--card-bg) !important;
            color: var(--text-color) !important;
            border-color: var(--border-color) !important;
            border-radius: 12px !important;
        }}
        .stButton>button {{
            background-color: var(--button-bg) !important;
            color: white !important;
            border-radius: 12px !important;
            transition: all 0.3s !important;
        }}
        .stButton>button:hover {{
            background-color: var(--button-hover) !important;
            transform: translateY(-1px) !important;
        }}
        .chat-bubble {{
            background-color: var(--card-bg) !important;
            border-radius: 12px !important;
            padding: 16px !important;
            margin: 8px 0 !important;
            border: 1px solid var(--border-color) !important;
        }}
    </style>
    """, unsafe_allow_html=True)

# ====== Session State ======
if "conversation" not in st.session_state:
    st.session_state.conversation = []
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False

# ====== Page Config ======
st.set_page_config(
    page_title="MIT Nova",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ====== Sidebar ======
with st.sidebar:
    # Logo with white background
    try:
        logo = base64.b64encode(open("logo.png", "rb").read()).decode()
        st.markdown(
            f"""
            <div style="
                background-color: white;
                padding: 10px;
                border-radius: 8px;
                display: inline-block;
                margin-bottom: 20px;
            ">
                <img src="data:image/png;base64,{logo}" width="180">
            </div>
            """,
            unsafe_allow_html=True
        )
    except:
        st.image("logo.png", width=150)
    
    st.markdown("---")
    
    # Dark Mode Toggle
    st.session_state.dark_mode = st.toggle("🌙 Dark Mode", value=st.session_state.dark_mode)
    
    # Conversation History
    st.markdown("### Conversation History")
    if st.session_state.conversation:
        for i, (q, a) in enumerate(st.session_state.conversation):
            if st.button(f"🗨️ {q[:25]}...", key=f"hist_{i}"):
                st.session_state.current_query = q
    else:
        st.caption("No history yet")

# Apply theme
set_custom_theme(st.session_state.dark_mode)

# ====== Main Interface ======
# Custom font header
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Roboto+Condensed:wght@700&display=swap');
.custom-header {
    font-family: 'Roboto Condensed', sans-serif;
    font-size: 28px !important;
    color: #4f46e5;
    margin-bottom: 10px;
    display: flex;
    align-items: center;
    gap: 10px;
}
</style>
""", unsafe_allow_html=True)

# Header with icon
st.markdown(
    """
    <div class="custom-header">
        <span>MIT NOVA - A new star in internal assistance</span>
        <span>🌟</span>
    </div>s
    """,
    unsafe_allow_html=True
)

st.caption("Your AI assistant for company policies and HR information")

# Chat input
query = st.text_input(
    "Ask your question...",
    placeholder="E.g.,what is the shift allowance from 2pm to 10pm ?",
    label_visibility="collapsed"
)

if st.button("Ask ➔", use_container_width=True) or query:
    if query.strip():
        with st.spinner("🔍 Searching ..."):
            answer = rag_query(json_db, pdf_db, query)
            st.session_state.conversation.append((query, answer))
        
        # Display answer
        st.markdown(f"""
        <div class="chat-bubble">
            <h4 style='margin-top:0;color:var(--text-color)'>Answer</h4>
            <div style='margin-bottom:0'>{answer}</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Feedback
        st.markdown("**Was this helpful?**")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("👍 Yes", use_container_width=True):
                st.toast("Thank you!")
        with col2:
            if st.button("👎 No", use_container_width=True):
                st.toast("We'll improve this answer")
    else:
        st.markdown("""
        <style>
            .custom-warning {
                background-color: #fff3cd;
                color: #856404;
                border-left: 4px solid #ffc107;
                padding: 12px;
                border-radius: 4px;
                margin: 16px 0;
            }
        </style>
        <div class="custom-warning">
            ⚠️ Please enter a question
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.caption("© 2025 MIT Nova | Powered by Groq & LangChain")