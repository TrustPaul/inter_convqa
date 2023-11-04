import langchain
import streamlit as st
from langchain.memory import ConversationBufferMemory
from langchain.callbacks import StreamlitCallbackHandler
import streamlit_survey as ss
from langchain.tools import DuckDuckGoSearchRun
from langchain.agents import initialize_agent, Tool, AgentType, Tool
from langchain.schema import SystemMessage
from langchain.prompts import MessagesPlaceholder
import os
from langchain.chat_models import ChatOpenAI

##Replace your openAI api here
OPENAI_API_KEY = "Replace with your OpenAI API"
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

st.set_page_config(page_title="Search bot with CHATGPT", page_icon="🦜")
st.title("Tool and Retrieval Augumented Chatbot")

if "conversation_retrieval_memory" not in st.session_state:
    st.session_state.conversation_retrieval_memory =  ConversationBufferMemory(memory_key="memory", return_messages=True)

def openaiagent():
    #llm =  ChatOpenAI(temperature=0.9, model_name="gpt-4")
    #embeddings = OpenAIEmbeddings()
    openai_api_key = OPENAI_API_KEY
    llm = ChatOpenAI(model_name="gpt-4", openai_api_key=openai_api_key, streaming=True, temperature=0.1)
    #docs = vectorstore.similarity_search(prompt, k=5)#
    persist_directory = 'irish_database'
    #vectorstore  = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    search = DuckDuckGoSearchRun()

    tools = [

            Tool(
                name = "Search",
                func=search.run,
                description="A useful tool for searching the Internet to find information",

            )
        ]
    system_message = f"""
    You are an InformationGPT, a language model that will answer questions for user by searching both the internet 
    When asked the question, no matter how simple the question is, you must use the tools provided to confirm the factual details before generating the answer
      """
    system_message = SystemMessage(

                content=system_message

            )
    agent_kwargs = {
        "extra_prompt_messages": [MessagesPlaceholder(variable_name="memory")],
        "system_message": system_message
    }
    
    agent = initialize_agent(tools, 
                            llm,
                            agent=AgentType.OPENAI_FUNCTIONS, 
                            verbose=True,
                            handle_parsing_errors=True,
                            max_iterations=10,
                            memory = st.session_state.conversation_retrieval_memory,
                            system_message=system_message,
                            early_stopping_method="generate",
                            agent_kwargs=agent_kwargs,
                            )
    return agent

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []



# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
# React to user input


if prompt := st.chat_input("Ask me about questions that require searching the internet"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    #response = utils.agent(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container())
        executor = openaiagent()
        #executor =  utils.google_agent()
        #try:
        response = executor.run(prompt,  callbacks=[st_cb])
       # except:
            #response = 'Sorry, I could not answer the question, ask another question'

        st.markdown(response)
        #st.write(f"```\n{response}")
        #st.code(response)
            # "Copy" button


    # Add assistant response to chat history
    survey = ss.StreamlitSurvey("Survey Example")
    Q1 = survey.radio("Rate Model Output", options=["NA", "👍", "👎"], horizontal=True)
    st.session_state.messages.append({"role": "assistant", "content": response})