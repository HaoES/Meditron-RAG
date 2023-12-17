import streamlit as st
from pypdf import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import HuggingFaceHub
from langchain import PromptTemplate
from htmlTemplates import css, bot_template, user_template

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
    # fix this later
    question = str(prompt.format(query=query))
    response = st.session_state.conversation({"question": question})
    st.session_state.chat_history = response["chat_history"]

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(
                user_template.replace("{{MSG}}", message.content[84:]),
                unsafe_allow_html=True,
            )
        else:
            st.write(
                bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True
            )


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
    st.set_page_config(page_title="Medical Tutor", page_icon=":medical_symbol:")
    st.header("Medical Tutor :medical_symbol:")
    st.write(css, unsafe_allow_html=True)

    # create session state object
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    # receiving user's query
    query = st.text_input(
        """3 - Ask the tutor to help you learn from your document:
    \nExample: "Give me a question that could figure on my final exam." """
    )
    if query:
        process_query(query)

    with st.sidebar:
        st.markdown(
            """
        # Medical tutor helps you study for your medical exams using your own course material:
        """
        )
        st.subheader("1 - Upload your document and hit 'Process'")
        st.markdown(
            """
        Example:[ACC's Breast Cancer Document](https://www.cancer.org/content/dam/CRC/PDF/Public/8577.00.pdf)
        """
        )
        doc = st.file_uploader(
            "Your Document here",
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
                st.write("2 - File Processed!, You can start learning!")


if __name__ == "__main__":
    main()
