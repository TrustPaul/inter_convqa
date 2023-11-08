import langchain
import os
import utils
import streamlit as st
from langchain.memory import ConversationBufferMemory
import time
import streamlit_survey as ss

YOUR_TOKEN = 'Replace with your huggingface token'
os.environ["HUGGINGFACEHUB_API_TOKEN"] = YOUR_TOKEN
HUGGINGFACEHUB_API_TOKEN = YOUR_TOKEN

repo_id = "HuggingFaceH4/zephyr-7b-beta" 


st.set_page_config(page_title="ChatPDF", page_icon="📄")
st.header('Retrieval Augumented Generation on Irish documents')
st.write('The model can answer questions on sample documents from hse website')

def pretty_print_docs(docs):
    return f"\n\n".join([f"Document {i+1}:\n\n" + d.page_content for i, d in enumerate(docs)])


if "messages" not in st.session_state:
    st.session_state.messages = []
    
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if "conversation_retrieval_memory_irish_hse" not in st.session_state:
    st.session_state.conversation_retrieval_memory_irish_hse = ConversationBufferMemory(memory_key="chat_history", input_key="human_input")

    

class CustomDataChatbot:

    def __init__(self):
        #utils.configure_openai_api_key()
        self.openai_model = repo_id

    @st.spinner('Analyzing documents..')
    def setup_qa_chain(self, query):
        assistant_response , docs = utils.retrieval_docs_irish_hse(query)

        return assistant_response , docs
    


    #@utils.enable_chat_history
    def main(self):
        user_query = st.chat_input(placeholder="Ask me about sample Irish Documents from HSE website")
        database_documents = "irish_documents_hse"
        if database_documents and user_query:
           # qa_chain , docs= self.setup_qa_chain(user_query)
            try:
                assistant_response , docs= self.setup_qa_chain(user_query)
            except:
                assistant_response = 'Sorry, I could not answer the question, ask another question'

        

            #utils.display_msg(user_query, 'user')
            st.session_state.messages.append({"role": "user", "content": user_query })
            with st.chat_message("assistant"):
                #try:
                #response = qa_chain.run(query=user_query)
            


                #assistant_response = f"{docs}"
                #except:
                   # response = 'Sorry, I could not answer the question, ask another question'
                st.markdown( user_query)
                message_placeholder = st.empty()
                full_response = ""
         
                #st_cb = StreamHandler(st.empty())
                
                for chunk in assistant_response.split():
                    full_response += chunk + " "
                    time.sleep(0.05)
                    message_placeholder.markdown(full_response + "▌")
                message_placeholder.markdown(full_response)

                survey = ss.StreamlitSurvey("Survey Example")
                Q1 = survey.radio("Rate Model Output", options=["NA", "👍", "👎"], horizontal=True)
                
                st.session_state.messages.append({"role": "assistant", "content": assistant_response})
            with st.expander("See Document Chunks used"):
                print(docs)
                docs_used = pretty_print_docs(docs)
                st.markdown(docs_used)
     
                

if __name__ == "__main__":
    obj = CustomDataChatbot()
    obj.main()