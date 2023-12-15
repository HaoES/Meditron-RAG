import streamlit as st
from pypdf import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Qdrant
from langchain.vectorstores import FAISS
import os


def get_vectors(chunks):
    embeddings = HuggingFaceEmbeddings(model_name="NeuML/pubmedbert-base-embeddings")
    vectorstore = FAISS.from_texts(texts=chunks, embeddings=embeddings)
    return vectorstore


def get_pdf_text(docs):
    text = ""
    for doc in docs:
        pdf_reader = PdfReader(doc)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def main():
    st.set_page_config(page_title="Med Prep", page_icon=":medical_symbol:")
    st.header("Med Prep :medical_symbol:")
    st.text_input("Ask questions about your documents:")

    with st.sidebar:
        st.subheader("Your documents")
        docs = st.file_uploader(
            "Upload your course materials here then click 'Process':",
            accept_multiple_files=True,
            type="pdf",
        )
        if st.button("Process"):
            with st.spinner("Processing files ..."):
                # get pdf files:
                raw_text = get_pdf_text(docs)
                # get chunks of texts
                chunks = get_chunks(raw_text)
                # get vectorstore
                vectorstore.add_texts(chunks)


if __name__ == "__main__":
    main()
