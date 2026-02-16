import streamlit as st
import os
from dotenv import load_dotenv
from pathlib import Path

from langchain_community.document_loaders import JSONLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

# -----------------------
# Load NVIDIA API Key
# -----------------------
load_dotenv()
os.environ["NVIDIA_API_KEY"] = os.getenv("NVIDIA_API_KEY")

# -----------------------
# Load JSON Documents
# -----------------------
json_dir = Path("./Inputdata")
all_docs = []

for json_file in json_dir.glob("*.json"):
    loader = JSONLoader(
        file_path=json_file,
        jq_schema='.updates[] | .plugin.description + " " + .plugin.synopsis + " Host: " + .asset.hostname',
        text_content=True
    )
    docs = loader.load()
    all_docs.extend(docs)

# -----------------------
# Vector Store (LOCAL EMBEDDINGS)
# -----------------------
def vector_embedding():
    if "vectors" not in st.session_state:

        embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=700,
            chunk_overlap=50
        )

        final_docs = splitter.split_documents(all_docs[:30])

        st.session_state.vectors = FAISS.from_documents(
            final_docs,
            embeddings
        )

# -----------------------
# Streamlit UI
# -----------------------
st.title("RAG Demo - Local Embeddings + NVIDIA LLM")

# NVIDIA LLM (Only this calls NVIDIA API)
llm = ChatNVIDIA(
    model="ai-llama-3_1-70b-instruct",
    temperature=0
)

prompt = ChatPromptTemplate.from_template(
"""
Answer ONLY from the provided context.

Context:
{context}

Question:
{input}
"""
)

question = st.text_input("Enter your question")

if st.button("Create Vector Database"):
    vector_embedding()
    st.success("Vector Store Ready")

if question:
    if "vectors" not in st.session_state:
        st.warning("Create vector database first.")
    else:
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        response = retrieval_chain.invoke({"input": question})

        st.write(response["answer"])

        with st.expander("Retrieved Documents"):
            for doc in response["context"]:
                st.write(doc.page_content)
                st.write("-------------------------")
