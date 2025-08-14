import os
import shutil
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama as OllamaLLM
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

st.set_page_config(page_title="RAG Chatbot", layout="wide", initial_sidebar_state="expanded")

MODEL_NAME = "qwen3:8b" 
EMBED_MODEL = "nomic-embed-text"
VECTORSTORE_PATH = "vectorstore"
DOCUMENTS_PATH = "documents"

os.makedirs(DOCUMENTS_PATH, exist_ok=True)
os.makedirs(VECTORSTORE_PATH, exist_ok=True)

@st.cache_resource
def initialize_models():
    print("Menginisialisasi model...")
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)
    llm = OllamaLLM(model=MODEL_NAME)
    print("Model berhasil diinisialisasi.")
    return embeddings, llm

embeddings, llm = initialize_models()

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "messages" not in st.session_state:
    st.session_state.messages = []

def process_documents(uploaded_files):
    all_chunks = []
    for uploaded_file in uploaded_files:
        file_path = os.path.join(DOCUMENTS_PATH, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=150)
        chunks = text_splitter.split_documents(docs)
        all_chunks.extend(chunks)

    if not all_chunks:
        return None
    vector_store = FAISS.from_documents(all_chunks, embeddings)
    return vector_store

def get_rag_chain(vector_store):
    retriever = vector_store.as_retriever(search_kwargs={'k': 4}) 
    prompt_template = """
    Anda adalah asisten AI yang ahli dalam membaca dan memahami dokumen.
    Jawablah pertanyaan pengguna secara akurat **hanya** berdasarkan informasi dari konteks yang diberikan di bawah ini.
    Jika informasi yang relevan tidak ditemukan di dalam konteks, jawab dengan sopan: "Maaf, saya tidak dapat menemukan informasi mengenai hal tersebut di dalam dokumen yang Anda berikan."
    Jangan mencoba menjawab dari pengetahuan umum Anda.

    KONTEKS DOKUMEN:
    {context}

    PERTANYAAN PENGGUNA:
    {question}

    JAWABAN:
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain

with st.sidebar:
    st.markdown("## ⚙️ Settings")
    st.markdown("---")
    
    uploaded_files = st.file_uploader(
        "Unggah file PDF Anda", 
        type=["pdf"], 
        accept_multiple_files=True,
        label_visibility="collapsed"
    )

    if st.button("📄 Document Process", use_container_width=True, disabled=not uploaded_files):
        with st.spinner("🧠 Analyzing documents..."):
            st.session_state.vector_store = process_documents(uploaded_files)
            st.session_state.messages = [{"role": "assistant", "content": "Halo! Dokumen Anda sudah siap. Apa yang ingin Anda tanyakan?"}]
            st.success("✅ Document successfully processed!")
    
    st.markdown("---")
    if st.button("🗑️ Delete Database & Chat", use_container_width=True):
        if os.path.exists(VECTORSTORE_PATH): shutil.rmtree(VECTORSTORE_PATH)
        if os.path.exists(DOCUMENTS_PATH): shutil.rmtree(DOCUMENTS_PATH)
        os.makedirs(DOCUMENTS_PATH, exist_ok=True)
        os.makedirs(VECTORSTORE_PATH, exist_ok=True)
        st.session_state.vector_store = None
        st.session_state.messages = []
        st.success("🗑️ Database and chat history successfully deleted!")
        st.rerun()
        
    st.markdown("---")

st.title("💬 RAG Chatbot")
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask Anything"):
    if st.session_state.vector_store is not None:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("🤖 Thinking..."):
                rag_chain = get_rag_chain(st.session_state.vector_store)
                answer = st.write_stream(rag_chain.stream(prompt))

        st.session_state.messages.append({"role": "assistant", "content": answer})
    else:
        st.warning("⚠️ Please upload and process the documents first before asking..")

