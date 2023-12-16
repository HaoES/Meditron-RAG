import streamlit as st
from pypdf import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import HuggingFaceHub
from langchain.chains.question_answering import load_qa_chain

import os


def get_vectors(chunks):
    embeddings = HuggingFaceEmbeddings(model_name="NeuML/pubmedbert-base-embeddings")
    vectorstore = FAISS.from_texts(texts=chunks, embedding=embeddings)
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


def handle_query(query):
    vects = st.session_state.vects
    docs = vects.similarity_search(query=query, k=3)
    llm = HuggingFaceHub(
        repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
        model_kwargs={"tempearture": 0.0, "max_length": 2048},
    )
    chain = load_qa_chain(llm=llm, chain_type="stuff")
    response = chain.run(input_documents=docs, question=query)
    st.write(response)


def main():
    st.set_page_config(page_title="Med Prep", page_icon=":medical_symbol:")
    st.header("Med Prep :medical_symbol:")

    # create session state object
    if "vects" not in st.session_state:
        st.session_state.vects = None

    # receiving user's query
    query = st.text_input("Ask questions about your documents:")
    if query:
        handle_query(query)

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
                vects = get_vectors(chunks)
                st.session_state.vects = vects


if __name__ == "__main__":
    main()
