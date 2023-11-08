import os
import random
import langchain
import utils
import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain import HuggingFaceHub
from langchain.vectorstores import Chroma
from langchain.chains.question_answering import load_qa_chain
from langchain import PromptTemplate
import streamlit_survey as ss

YOUR_TOKEN = 'Replace with your huggingface token'
os.environ["HUGGINGFACEHUB_API_TOKEN"] = YOUR_TOKEN
HUGGINGFACEHUB_API_TOKEN = YOUR_TOKEN

repo_id = "HuggingFaceH4/zephyr-7b-beta" 




#decorator
def enable_chat_history(func):
    if os.environ.get("HUGGINGFACEHUB_API_TOKEN"):

        # to clear chat history after swtching chatbot
        current_page = func.__qualname__
        if "current_page" not in st.session_state:
            st.session_state["current_page"] = current_page
        if st.session_state["current_page"] != current_page:
            try:
                st.cache_resource.clear()
                del st.session_state["current_page"]
                del st.session_state["messages"]
            except:
                pass

        # to show chat history on ui
        if "messages" not in st.session_state:
            st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]
        for msg in st.session_state["messages"]:
            st.chat_message(msg["role"]).write(msg["content"])

    def execute(*args, **kwargs):
        func(*args, **kwargs)
    return execute

def display_msg(msg, author):
    """Method to display message on the UI

    Args:
        msg (str): message to display
        author (str): author of the message -user/assistant
    """
    st.session_state.messages.append({"role": author, "content": msg})
    st.chat_message(author).write(msg)

def configure_openai_api_key():
    openai_api_key = st.sidebar.text_input(
        label="OpenAI API Key",
        type="password",
        value=st.session_state['OPENAI_API_KEY'] if 'OPENAI_API_KEY' in st.session_state else '',
        placeholder="sk-..."
        )
    if openai_api_key:
        st.session_state['OPENAI_API_KEY'] = openai_api_key
        os.environ['OPENAI_API_KEY'] = openai_api_key
    else:
        st.error("Please add your OpenAI API key to continue.")
        st.info("Obtain your key from this link: https://platform.openai.com/account/api-keys")
        st.stop()
    return openai_api_key

def save_file(file):
    folder = 'tmp'
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    file_path = f'./{folder}/{file.name}'
    with open(file_path, 'wb') as f:
        f.write(file.getvalue())
    return file_path


@st.cache_data(show_spinner=False)
def retrieval_docs_irish(query):
    database_documents = "irish_documents_gov_ie"
    # Load documents
    # Embeddings
    model_name = "sentence-transformers/paraphrase-albert-small-v2"
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': False}
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )

    persist_directory = database_documents
    vectordb  = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    docs = vectordb.similarity_search(query, k=5)
            # Define retriever
  

    repo_id = "HuggingFaceH4/zephyr-7b-beta" 
    llm = HuggingFaceHub(repo_id=repo_id, model_kwargs={"temperature": 0.1, "max_new_tokens": 200})

    template = """
        Use the following pieces of text delimted by backticks (```) to answer the questions
        If you can not find the answer in the provided text, do not try to make up the answer but return your response as I am sorry, I do not know the answer
        In addition you are able to answer followup questions</s>\n
        ```{context}```
        Current conversation </s>\n
        {chat_history} </s>\n
        <|user|>: {human_input}</s>\n
        <|assistant|>\n """

    #llm = ChatOpenAI(model_name="gpt-3.5-turbo",  temperature=0, streaming=True)


    prompt_template = PromptTemplate(
        input_variables=["chat_history", "human_input", "context"], template=template
    )

    chain = load_qa_chain(
                    llm, chain_type="stuff", memory=st.session_state.conversation_retrieval_memory_irish, prompt= prompt_template
                )

    result = chain({"input_documents": docs, "human_input": query}, return_only_outputs=True)
    assistant_response = result['output_text']
    #assistant_response  = assistant_response
    assistant_response = f"""{assistant_response}""" 
    return   assistant_response , docs 


@st.cache_data(show_spinner=False)
def retrieval_docs_irish_hse(query):
    database_documents = "irish_documents_hse"
    # Load documents
    # Embeddings
    model_name = "sentence-transformers/paraphrase-albert-small-v2"
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': False}
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )

    persist_directory = database_documents
    vectordb  = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    docs = vectordb.similarity_search(query, k=5)
            # Define retriever
  

    repo_id = "HuggingFaceH4/zephyr-7b-beta" 
    llm = HuggingFaceHub(repo_id=repo_id, model_kwargs={"temperature": 0.1, "max_new_tokens": 200})

    template = """
        Use the following pieces of text delimted by backticks (```) to answer the questions
        If you can not find the answer in the provided text, do not try to make up the answer but return your response as I am sorry, I do not know the answer
        In addition you are able to answer followup questions</s>\n
        ```{context}```
        Current conversation </s>\n
        {chat_history} </s>\n
        <|user|>: {human_input}</s>\n
        <|assistant|>\n """

    #llm = ChatOpenAI(model_name="gpt-3.5-turbo",  temperature=0, streaming=True)


    prompt_template = PromptTemplate(
        input_variables=["chat_history", "human_input", "context"], template=template
    )

    chain = load_qa_chain(
                    llm, chain_type="stuff", memory=st.session_state.conversation_retrieval_memory_irish_hse, prompt= prompt_template
                )

    result = chain({"input_documents": docs, "human_input": query}, return_only_outputs=True)
    assistant_response = result['output_text']
    #assistant_response  = assistant_response
    assistant_response = f"""{assistant_response}""" 
    return   assistant_response , docs 


@st.cache_data(show_spinner=False)
def retrieval_docs_your_documents(query, uploaded_files):

    # Load documents
    docs = []
    for file in uploaded_files:
        file_path = save_file(file)
        loader = PyPDFLoader(file_path)
        docs.extend(loader.load())
    
    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=200
    )
    splits = text_splitter.split_documents(docs)

    # Create embeddings and store in vectordb
    embeddings = HuggingFaceEmbeddings(model_name="paraphrase-albert-small-v2")
    vectordb = DocArrayInMemorySearch.from_documents(splits, embeddings)
    docs_ra = vectordb.similarity_search(query, k=5)
   
    repo_id = "HuggingFaceH4/zephyr-7b-beta" 
    llm = HuggingFaceHub(repo_id=repo_id, model_kwargs={"temperature": 0.1, "max_new_tokens": 200})

    template = """
        Use the following pieces of text delimted by backticks (```) to answer the questions
        If you can not find the answer in the provided text, do not try to make up the answer but return your response as I am sorry, I do not know the answer
        In addition you are able to answer followup questions</s>\n
        ```{context}```
        Current conversation </s>\n
        {chat_history} </s>\n
        <|user|>: {human_input}</s>\n
        <|assistant|>\n """
    #llm = ChatOpenAI(model_name="gpt-3.5-turbo",  temperature=0, streaming=True)


    prompt_template = PromptTemplate(
        input_variables=["chat_history", "human_input", "context"], template=template
    )

    chain = load_qa_chain(
                    llm, chain_type="stuff", memory=st.session_state.conversation_retrieval_memory_your_documents, prompt= prompt_template
                )

    result = chain({"input_documents": docs_ra, "human_input": query}, return_only_outputs=True)
    assistant_response = result['output_text']
    #assistant_response  = assistant_response
    #docs = f"""{docs}""" 
    assistant_response = f"""{assistant_response}""" 
    return   assistant_response , docs_ra 