import streamlit as st
from utils.loader import load_pdf_documents
from utils.vector_store import create_vector_store
from utils.llm import get_llm

st.set_page_config(page_title="RAG App", page_icon="📘")
st.title("RAG App (Updated Version)")

uploaded_file = st.file_uploader("Upload PDF", type="pdf")

if uploaded_file:
    file_path = f"data/{uploaded_file.name}"
    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())

    documents = load_pdf_documents(file_path)
    vector_store = create_vector_store(documents)
    st.success("Vector store created successfully!")

    # ASK QUESTION
    question = st.text_input("Ask a question about your PDF")
    if question:
        llm = get_llm()
        response = llm.invoke(question)
        st.write("### Answer:")
        st.write(response)
