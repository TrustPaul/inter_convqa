import langchain
import utils
import streamlit as st
#from streaming import StreamHandler
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain import HuggingFaceHub
from streaming import StreamHandler
import time
import os
import streamlit_survey as ss
from langchain.prompts.prompt import PromptTemplate

## Put the Huggingface Token Here
YOUR_TOKEN = 'Replace with your huggingface token'
os.environ["HUGGINGFACEHUB_API_TOKEN"] = YOUR_TOKEN
HUGGINGFACEHUB_API_TOKEN = YOUR_TOKEN
repo_id = "HuggingFaceH4/zephyr-7b-beta" 



st.set_page_config(page_title="Basic Chatbot", page_icon="⭐")
st.header('Basic Chatbot')
st.write('Questions are answered based on the Language Model Parametric Chatbot')

if "messages" not in st.session_state:
    st.session_state.messages = []
    
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if "conversation_retrieval_memory_basic" not in st.session_state:
    st.conversation_retrieval_memory_basic = ConversationBufferMemory(k=5)   
    # Setup LLM and QA chain


class Basic:

    def __init__(self):
        #utils.configure_openai_api_key()
        self.openai_model = repo_id
    
    @st.cache_resource
    def setup_chain(_self):
        # Setup memory for contextual conversation 
   
        
        repo_id = "HuggingFaceH4/zephyr-7b-beta" 
        llm = HuggingFaceHub(
            repo_id=repo_id, model_kwargs={"temperature": 0.1, "max_new_tokens": 200}
        )
        template = """
            You are language models that answers user provided questions as faithfully as possible,
            In addition you are able to answer followup questions </s>\n
            Current conversation:
            {history}
            </s>\n
            <|user|>: {input}</s>\n
            <|assistant|>\n :"""
        PROMPT = PromptTemplate(input_variables=["history", "input"], template=template)
        chain = ConversationChain(prompt=PROMPT,llm=llm, memory= st.conversation_retrieval_memory_basic)
        return chain
    
    #@utils.enable_chat_history
    def main(self):
        chain = self.setup_chain()
        user_query = st.chat_input(placeholder="Ask me anything!")
        if user_query:
            #utils.display_msg(user_query, 'user')
            st.session_state.messages.append({"role": "user", "content": user_query })
            with st.chat_message("assistant"):
                #st_cb = StreamHandler(st.empty())
                #response = chain.predict(input=user_query)
                try:
                    response = chain.predict(input=user_query)
                except:
                    response = 'Sorry, I could not answer the question, ask another question'
                st.markdown( user_query)
                message_placeholder = st.empty()
                full_response = ""
         
                #st_cb = StreamHandler(st.empty())
                
                for chunk in response.split():
                    full_response += chunk + " "
                    time.sleep(0.05)
                    message_placeholder.markdown(full_response + "▌")
                message_placeholder.markdown(full_response)

                survey = ss.StreamlitSurvey("Survey Example")
                Q1 = survey.radio("Rate Model Output", options=["NA", "👍", "👎"], horizontal=True)
                st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    obj = Basic()
    obj.main()