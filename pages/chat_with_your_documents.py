import langchain
import os
import utils
import streamlit as st
from langchain.memory import ConversationBufferMemory
import time
import streamlit_survey as ss
from streaming import StreamHandler
import os

## Put the Huggingface Token Here

YOUR_TOKEN = 'Replace with your huggingface token'
os.environ["HUGGINGFACEHUB_API_TOKEN"] = YOUR_TOKEN
HUGGINGFACEHUB_API_TOKEN = YOUR_TOKEN

repo_id = "HuggingFaceH4/zephyr-7b-beta"


st.set_page_config(page_title="ChatPDF", page_icon="📄")
st.header('Retrieval Augumented Generation with your own private documents')
st.write('Upload Documents and use a language model to chat with your documents by asking questions')

def pretty_print_docs(docs):
    return f"\n\n".join([f"Document {i+1}:\n\n" + d.page_content for i, d in enumerate(docs)])


if "messages" not in st.session_state:
    st.session_state.messages = []
    
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if "conversation_retrieval_memory_your_documents" not in st.session_state:
    st.session_state.conversation_retrieval_memory_your_documents = ConversationBufferMemory(memory_key="chat_history", input_key="human_input")




class CustomDataChatbot:

    def __init__(self):
        #utils.configure_openai_api_key()
        self.openai_model = repo_id

    def save_file(self, file):
        folder = 'tmp'
        if not os.path.exists(folder):
            os.makedirs(folder)
        
        file_path = f'./{folder}/{file.name}'
        with open(file_path, 'wb') as f:
            f.write(file.getvalue())
        return file_path

    @st.spinner('Analyzing documents..')
    def setup_qa_chain(self, uploaded_files, query):

        assistant_response , docs  = utils.retrieval_docs_your_documents(query, uploaded_files)


        return assistant_response , docs

    #@utils.enable_chat_history
    def main(self):

        # User Inputs
        uploaded_files = st.sidebar.file_uploader(label='Upload PDF files', type=['pdf'], accept_multiple_files=True)
        if not uploaded_files:
            st.error("Please upload PDF documents to continue!")
            st.stop()

        user_query = st.chat_input(placeholder="Ask me anything!")

        if uploaded_files and user_query:
           # response , docs = self.setup_qa_chain(uploaded_files, user_query)
            try:
                response, docs = self.setup_qa_chain(uploaded_files, user_query)
            except:
                response = 'Sorry, I could not answer the question, ask another question'


            #utils.display_msg(user_query, 'user')
            st.session_state.messages.append({"role": "user", "content": user_query})

            with st.chat_message("assistant"):
                st_cb = StreamHandler(st.empty())
                #try:
                
                #response = qa_chain.run(user_query)
                #except:
                   # response = 'Sorry, I could not answer the question, ask another question'
                st.markdown( user_query)
                message_placeholder = st.empty()
                full_response = ""
         
                #st_cb = StreamHandler(st.empty())
                #st_cb = StreamHandler(st.empty())
                for chunk in response.split():
                    full_response += chunk + " "
                    time.sleep(0.05)
                    message_placeholder.markdown(full_response + "▌")
                message_placeholder.markdown(full_response)

                st.session_state.messages.append({"role": "assistant", "content": response})
                survey = ss.StreamlitSurvey("Survey Example")
                Q1 = survey.radio("Rate Model Output", options=["NA", "👍", "👎"], horizontal=True)

                with st.expander("See Document Chunks used"):
                    docs_used = pretty_print_docs(docs)
                    st.markdown(docs_used)

if __name__ == "__main__":
    obj = CustomDataChatbot()
    obj.main()