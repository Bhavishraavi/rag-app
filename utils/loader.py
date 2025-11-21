from PyPDF2 import PdfReader
from langchain_core.documents import Document

def load_pdf_documents(file_path):
    reader = PdfReader(file_path)
    docs = []

    for page in reader.pages:
        text = page.extract_text()
        if text:
            docs.append(Document(page_content=text))

    return docs
