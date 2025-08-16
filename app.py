import os
import shutil
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama as OllamaLLM
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

load_dotenv()

st.set_page_config(page_title="RAG Chatbot", layout="wide", initial_sidebar_state="expanded")

LLM_MODEL = os.environ["LLM_MODEL"]
EMBEDDING_MODEL = os.environ["EMBEDDING_MODEL"]
VECTORSTORE_PATH = os.environ["VECTORSTORE_PATH"]
DOCUMENTS_PATH = os.environ["DOCUMENTS_PATH"]
os.makedirs(DOCUMENTS_PATH, exist_ok=True)
os.makedirs(VECTORSTORE_PATH, exist_ok=True)

@st.cache_resource
def initialize_models():
    print(f"Initializing models... LLM={LLM_MODEL}, Embeddings={EMBEDDING_MODEL}")
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    llm = OllamaLLM(model=LLM_MODEL)
    print("Models successfully initialized.")
    return embeddings, llm

embeddings_model, llm_model = initialize_models()

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
        
        try:
            loader = PyPDFLoader(file_path)
            docs = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=150)
            chunks = text_splitter.split_documents(docs)
            all_chunks.extend(chunks)
        except Exception as e:
            st.error(f"Error processing file {uploaded_file.name}: {e}")
            continue

    if not all_chunks:
        return None
    
    vector_store = FAISS.from_documents(all_chunks, embeddings_model)
    return vector_store

def get_rag_chain(vector_store):
    retriever = vector_store.as_retriever(search_kwargs={'k': 4}) 
    
    prompt_template = """
    You are an expert AI assistant for reading and understanding documents.
    Answer the user's question accurately **only** based on the information from the context provided below.
    If the relevant information is not found within the context, politely reply: "I'm sorry, I couldn't find any information about that in the documents you provided."
    Do not try to answer from your general knowledge.

    DOCUMENT CONTEXT:
    {context}

    USER'S QUESTION:
    {question}

    ANSWER:
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm_model
        | StrOutputParser()
    )
    return rag_chain

with st.sidebar:
    st.markdown("## ⚙️ Settings")

    uploaded_files = st.file_uploader(
        "Upload your PDF files", 
        type=["pdf"], 
        accept_multiple_files=True,
        label_visibility="collapsed"
    )

    if st.button("📄 Process Documents", use_container_width=True, disabled=not uploaded_files):
        with st.spinner("🧠 Analyzing documents..."):
            st.session_state.vector_store = process_documents(uploaded_files)
            if st.session_state.vector_store:
                st.session_state.messages = [{"role": "assistant", "content": "Hello! you can ask anything"}]
                st.success("✅ Documents successfully processed!")
            else:
                st.error("⚠️ No documents were processed. Please check your files.")
    
    if st.button("🗑️ Delete Database & Chat", use_container_width=True):
        if os.path.exists(VECTORSTORE_PATH): shutil.rmtree(VECTORSTORE_PATH)
        if os.path.exists(DOCUMENTS_PATH): shutil.rmtree(DOCUMENTS_PATH)
        os.makedirs(DOCUMENTS_PATH, exist_ok=True)
        os.makedirs(VECTORSTORE_PATH, exist_ok=True)
        st.session_state.vector_store = None
        st.session_state.messages = []
        st.success("🗑️ Database and chat history successfully deleted!")
        st.rerun()
            
st.title("💬 RAG Chatbot")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask anything about your documents"):
    if st.session_state.vector_store is not None:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("🤖 Thinking..."):
                rag_chain = get_rag_chain(st.session_state.vector_store)
                response = st.write_stream(rag_chain.stream(prompt))

        st.session_state.messages.append({"role": "assistant", "content": response})
    else:
        st.warning("⚠️ Please upload and process documents before asking a question.")