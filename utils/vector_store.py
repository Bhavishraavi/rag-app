from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from utils.embedding.embedding import get_embeddings

def create_vector_store(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=200
    )

    chunks = text_splitter.split_documents(documents)

    embeddings = get_embeddings()

    vector_store = FAISS.from_documents(chunks, embedding=embeddings)

    return vector_store
