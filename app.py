import streamlit as st
from pypdf import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import HuggingFaceHub
from langchain import PromptTemplate
import os

template = """ 
You are a tutor helping me study for my medical exam using the provided context. 
{query}
"""

prompt = PromptTemplate.from_template(template)


def get_vectors(chunks):
    embeddings = HuggingFaceEmbeddings(model_name="NeuML/pubmedbert-base-embeddings")
    vectorstore = FAISS.from_texts(texts=chunks, embedding=embeddings)
    return vectorstore


def get_pdf_text(doc):
    text = ""
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


def process_query(query):
    question = str(prompt.format(query=query))
    response = st.session_state.conversation({"question": question})
    st.write(response)


def get_conv(vects):
    llm = HuggingFaceHub(
        repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
        model_kwargs={"tempearture": 0.0, "max_length": 2048},
    )
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm, retriever=vects.as_retriever(), memory=memory
    )
    return conversation_chain


def main():
    st.set_page_config(page_title="Med Prep", page_icon=":medical_symbol:")
    st.header("Med Prep :medical_symbol:")

    # create session state object
    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    # receiving user's query
    query = st.text_input("Ask questions about your document:")
    if query:
        process_query(query)

    with st.sidebar:
        st.subheader("Your document")
        doc = st.file_uploader(
            "Upload your course material here then click 'Process':",
            type="pdf",
        )
        if st.button("Process"):
            with st.spinner("Processing file ..."):
                # get pdf files:
                raw_text = get_pdf_text(doc)
                # get chunks of texts
                chunks = get_chunks(raw_text)
                # get vectorstore
                vects = get_vectors(chunks)
                # get conversation
                st.session_state.conversation = get_conv(vects)
                st.write("File Processed!, You can start learning!")


if __name__ == "__main__":
    main()
