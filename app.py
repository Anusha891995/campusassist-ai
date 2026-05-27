# =========================================
# CAMPUSASSIST AI
# PROFESSIONAL RAG CHATBOT
# STREAMLIT APPLICATION
# FILE: app.py
# =========================================

import streamlit as st
from dotenv import load_dotenv
load_dotenv()

import os
import time

# =========================================
# LANGCHAIN IMPORTS
# =========================================

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader

# =========================================
# PAGE CONFIGURATION
# =========================================

st.set_page_config(
    page_title="CampusAssist AI",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =========================================
# CUSTOM CSS
# =========================================

st.markdown("""
<style>

/* ======================================
HIDE STREAMLIT DEFAULT UI
====================================== */

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

header {
    visibility: hidden;
}

/* ======================================
MAIN APP BACKGROUND
====================================== */

.stApp {

    background:
    linear-gradient(
        135deg,
        #020617,
        #0f172a,
        #111827,
        #020617
    );

    color: white;
}

/* ======================================
APP CONTAINER
====================================== */

.block-container {
    padding-top: 1rem;
    padding-bottom: 2rem;
}

/* ======================================
LOGO SECTION
====================================== */

.logo-container {

    display: flex;
    justify-content: center;
    align-items: center;

    margin-top: 10px;
}

/* ======================================
ANIMATED LOGO
====================================== */

.logo-circle {

    width: 120px;
    height: 120px;

    border-radius: 50%;

    background:
    linear-gradient(
        135deg,
        #38bdf8,
        #2563eb
    );

    display: flex;
    justify-content: center;
    align-items: center;

    font-size: 52px;

    box-shadow:
    0 0 20px rgba(56,189,248,0.5),
    0 0 50px rgba(56,189,248,0.4);

    animation: pulse 2.5s infinite;
}

/* ======================================
PULSE ANIMATION
====================================== */

@keyframes pulse {

    0% {
        transform: scale(1);
    }

    50% {
        transform: scale(1.08);
    }

    100% {
        transform: scale(1);
    }
}

/* ======================================
FLOATING ANIMATION
====================================== */

@keyframes floating {

    0% {
        transform: translateY(0px);
    }

    50% {
        transform: translateY(-10px);
    }

    100% {
        transform: translateY(0px);
    }
}

/* ======================================
GRADIENT TEXT ANIMATION
====================================== */

@keyframes gradientMove {

    0% {
        background-position: 0% center;
    }

    100% {
        background-position: 300% center;
    }
}

/* ======================================
FADE ANIMATION
====================================== */

@keyframes fadeIn {

    from {
        opacity: 0;
        transform: translateY(20px);
    }

    to {
        opacity: 1;
        transform: translateY(0px);
    }
}

/* ======================================
COLLEGE NAME
====================================== */

.college-name {

    text-align: center;

    font-size: 30px;

    font-weight: 700;

    color: #e2e8f0;

    letter-spacing: 3px;

    margin-top: 20px;

    animation:
    floating 4s ease-in-out infinite;
}

/* ======================================
MAIN TITLE
====================================== */

.main-title {

    text-align: center;

    font-size: 72px;

    font-weight: 900;

    margin-top: 10px;

    background:
    linear-gradient(
        90deg,
        #38bdf8,
        #60a5fa,
        #2563eb,
        #38bdf8
    );

    background-size: 300% auto;

    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;

    animation:
    gradientMove 6s linear infinite,
    floating 4s ease-in-out infinite;
}

/* ======================================
SUBTITLE
====================================== */

.subtitle {

    text-align: center;

    color: #cbd5e1;

    font-size: 20px;

    margin-top: 8px;

    margin-bottom: 40px;

    animation: fadeIn 1.5s ease;
}

/* ======================================
CHAT CONTAINER
====================================== */

.stChatMessage {

    background:
    rgba(255,255,255,0.05);

    border:
    1px solid rgba(255,255,255,0.08);

    border-radius: 20px;

    padding: 18px;

    margin-bottom: 15px;

    backdrop-filter: blur(12px);

    box-shadow:
    0 0 20px rgba(0,0,0,0.2);

    animation: fadeIn 0.5s ease;
}

/* ======================================
CHAT TEXT
====================================== */

.stChatMessage p {

    font-size: 17px;

    line-height: 1.8;
}

/* ======================================
INPUT BOX
====================================== */

.stChatInput input {

    background:
    rgba(15,23,42,0.95) !important;

    color: white !important;

    border:
    1px solid #38bdf8 !important;

    border-radius: 20px !important;

    padding: 16px !important;

    font-size: 16px !important;

    box-shadow:
    0 0 20px rgba(56,189,248,0.2);

    transition: 0.3s ease;
}

/* ======================================
INPUT HOVER
====================================== */

.stChatInput input:focus {

    border:
    1px solid #60a5fa !important;

    box-shadow:
    0 0 30px rgba(56,189,248,0.6);
}

/* ======================================
PLACEHOLDER
====================================== */

.stChatInput input::placeholder {
    color: #94a3b8;
}

/* ======================================
SPINNER COLOR
====================================== */

.stSpinner > div {
    border-top-color: #38bdf8 !important;
}

/* ======================================
FOOTER
====================================== */

.footer {

    text-align: center;

    color: #94a3b8;

    margin-top: 50px;

    font-size: 15px;
}

/* ======================================
RESPONSIVE DESIGN
====================================== */

@media (max-width: 768px) {

    .main-title {
        font-size: 48px;
    }

    .college-name {
        font-size: 22px;
    }

    .subtitle {
        font-size: 16px;
    }

    .logo-circle {

        width: 90px;
        height: 90px;

        font-size: 40px;
    }
}

</style>
""", unsafe_allow_html=True)

# =========================================
# HEADER SECTION
# =========================================

st.markdown("""
<div class="logo-container">
    <div class="logo-circle">
        🎓
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="college-name">
    SANTHIRAM ENGINEERING COLLEGE
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-title">
    CampusAssist AI
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="subtitle">
    Intelligent AI Assistant for Smart Campus Support
</div>
""", unsafe_allow_html=True)

# =========================================
# LOAD RAG PIPELINE
# =========================================

@st.cache_resource
def load_rag_pipeline():

    documents = []

    folder_path = "data"

    for file in os.listdir(folder_path):

        if file.endswith(".pdf"):

            pdf_path = os.path.join(folder_path, file)

            loader = PyPDFLoader(pdf_path)

            documents.extend(loader.load())

    # =====================================
    # TEXT SPLITTING
    # =====================================

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    docs = splitter.split_documents(documents)

    # =====================================
    # EMBEDDINGS
    # =====================================

    embedding = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # =====================================
    # VECTOR DATABASE
    # =====================================

    db = FAISS.from_documents(
        docs,
        embedding
    )

    # =====================================
    # RETRIEVER
    # =====================================

    retriever = db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )

    # =====================================
    # GEMINI MODEL
    # =====================================

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite",
        temperature=0.2,
        google_api_key="AIzaSyDQSG-mWujBtdFiwIES1A3VitDWMjx31IA"
    )

    # =====================================
    # PROMPT TEMPLATE
    # =====================================

    prompt_template = """

    You are CampusAssist AI.

    Answer questions only from the provided context.

    If exact answer is not available,
    try to provide the closest relevant answer from context.

    If nothing relevant exists, say:
    "I don't have enough information."

    Give short and clear answers.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """

    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    # =====================================
    # RETRIEVAL QA CHAIN
    # =====================================

    qa_chain = RetrievalQA.from_chain_type(

        llm=llm,

        chain_type="stuff",

        retriever=retriever,

        return_source_documents=True,

        chain_type_kwargs={
            "prompt": PROMPT
        }
    )

    return qa_chain

# =========================================
# INITIALIZE PIPELINE
# =========================================

with st.spinner("🚀 Initializing CampusAssist AI..."):

    qa_chain = load_rag_pipeline()

# =========================================
# CHAT HISTORY
# =========================================

if "messages" not in st.session_state:

    st.session_state.messages = []

# =========================================
# DISPLAY CHAT HISTORY
# =========================================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

# =========================================
# USER INPUT
# =========================================

user_question = st.chat_input(
    "Ask anything about your campus documents..."
)

# =========================================
# PROCESS QUESTION
# =========================================

if user_question:

    # USER MESSAGE

    st.session_state.messages.append({
        "role": "user",
        "content": user_question
    })

    with st.chat_message("user"):

        st.markdown(user_question)

    # ASSISTANT RESPONSE

    with st.chat_message("assistant"):

        with st.spinner("🤖 CampusAssist AI is Thinking..."):

            result = qa_chain.invoke(
                {"query": user_question}
            )

            answer = result["result"]

            # =================================
            # TYPING EFFECT
            # =================================

            message_placeholder = st.empty()

            full_response = ""

            for word in answer.split():

                full_response += word + " "

                time.sleep(0.03)

                message_placeholder.markdown(
                    full_response + "▌"
                )

            message_placeholder.markdown(
                full_response
            )

    # SAVE RESPONSE

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })

# =========================================
# FOOTER
# =========================================

st.markdown("<hr>", unsafe_allow_html=True)

st.markdown("""
<div class="footer">

🚀 Powered by Gemini AI • LangChain • FAISS • Streamlit

<br><br>

© 2026 CampusAssist AI — Santhiram Engineering College

</div>
""", unsafe_allow_html=True)