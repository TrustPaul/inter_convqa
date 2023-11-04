import langchain
import utils
import streamlit as st
import time
from langchain.tools import DuckDuckGoSearchRun
from langchain.agents import initialize_agent, Tool
from langchain import HuggingFaceHub
import os
from langchain.agents import initialize_agent
from langchain.memory import ConversationBufferWindowMemory
from langchain.callbacks import StreamlitCallbackHandler
import streamlit_survey as ss

## Put the Huggingface Token Here
YOUR_TOKEN = 'Replace with your huggingface token'
os.environ["HUGGINGFACEHUB_API_TOKEN"] = YOUR_TOKEN
HUGGINGFACEHUB_API_TOKEN = YOUR_TOKEN


repo_id = 'meta-llama/Llama-2-13b-hf'

st.set_page_config(page_title="ChatWeb", page_icon="🌐")
st.header('Tool Augumented Language Model Generation')
st.write('The model has access to the internet')

if "messages" not in st.session_state:
    st.session_state.messages = []
    
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if "conversation_retrieval_memory_agent" not in st.session_state:
    st.conversation_retrieval_memory_agent = ConversationBufferWindowMemory(memory_key="chat_history", k=5, return_messages=True, output_key="output") 


class ChatbotTools:

    def __init__(self):
        #utils.configure_openai_api_key()
        self.openai_model = "google/flan-t5-small"

    def setup_agent(self):
        # Define tool
        search = DuckDuckGoSearchRun()
        tools = [
            Tool(
                name="Google Search",
                func=search.run,
                description="useful for when you need to answer questions about current events or the current state of the world",
            )
        ]

        repo_id = "meta-llama/Llama-2-70b-chat-hf" 
        llm = HuggingFaceHub(
            repo_id=repo_id, model_kwargs={"temperature": 0.1, "max_new_tokens": 1024}
        )
        agent = initialize_agent(
            agent="chat-conversational-react-description",
            tools=tools,
            llm=llm,
            verbose=True,
            early_stopping_method="generate",
            max_iterations=2,
            memory= st.conversation_retrieval_memory_agent,
            handle_parsing_errors=True,
            #agent_kwargs={"output_parser": parser}
        )
        B_INST, E_INST = "[INST]", "[/INST]"
        B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
        sys_msg = B_SYS + """Assistant is a expert JSON builder designed to assist with a wide range of tasks.

        Assistant is able to respond to the User and use tools using JSON strings that contain "action" and "action_input" parameters.

        All of Assistant's communication is performed using this JSON format.

        Assistant can also use tools by responding to the user with tool use instructions in the same "action" and "action_input" JSON format. Tools available to Assistant are:

        - "Google Search": Useful for when you need to answer questions about current events. The input is the question to search relavant information.
        - To use the Google search tool, Assistant should write like so:
            ```json
            {{"action": "Google Search",
            "action_input": "What is AI?"}}
            ```

        Here are some previous conversations between the Assistant and User:

        User: Hey how are you today?
        Assistant: ```json
        {{"action": "Final Answer",
        "action_input": "I'm good thanks, how are you?"}}
        ```
        User: I'm great, who is the top scoler in English premier league?
        Assistant: ```json
        {{"action": "Google Search",
        "action_input": "top scoler in English premier league"}}
        ```
        User: Erling Haaland
        Assistant: ```json
        {{"action": "Final Answer",
        "action_input": "It looks like the answer is Erling Haaland!"}}
        ```
        User: Thanks could you tell me what team he plays for?
        Assistant: ```json
        {{"action": "Google Search",
        "action_input": "what team does Erling Haaland play in"}}
        ```
        User: Manchester City and the Norway national team
        Assistant: ```json
        {{"action": "Final Answer",
        "action_input": "It looks like the answer is Manchester City!"}}
        ```

        User: Hey how are you today?
        Assistant: ```json
        {{"action": "Final Answer",
        "action_input": "I'm good thanks, how are you?"}}


        Here is the latest conversation between Assistant and User.""" + E_SYS
        new_prompt = agent.agent.create_prompt(
            system_message=sys_msg,
            tools=tools
        )
        agent.agent.llm_chain.prompt = new_prompt
        instruction = B_INST + " Respond to the following in JSON with 'action' and 'action_input' values " + E_INST
        human_msg = instruction + "\nUser: {input}"

        agent.agent.llm_chain.prompt.messages[2].prompt.template = human_msg

        return agent

    #@utils.enable_chat_history
    def main(self):
        agent = self.setup_agent()
        user_query = st.chat_input(placeholder="Ask me anything!")
        if user_query:
            #utils.display_msg(user_query, 'user')
            st.session_state.messages.append({"role": "user", "content": user_query })
            with st.chat_message("assistant"):
                st_cb = StreamlitCallbackHandler(st.container())
               # st_callback = StreamlitCallbackHandler(st.container())
                try:
                  response = agent(user_query,callbacks=[st_cb])
                  response = response['output']
                except:
                    response = 'Sorry, I could not answer the question, ask another question'



                st.session_state.messages.append({"role": "assistant", "content": response})
               # st_cb = StreamHandler(st.empty())
                st.markdown( user_query)
                message_placeholder = st.empty()
                full_response = ""
         
                #st_cb = StreamHandler(st.empty())
                
                for chunk in response.split():
                    full_response += chunk + " "
                    time.sleep(0.05)
                    message_placeholder.markdown(full_response + "▌")
                message_placeholder.markdown(full_response)

                # Add assistant response to chat history
                survey = ss.StreamlitSurvey("Survey Example")
                Q1 = survey.radio("Rate Model Output", options=["NA", "👍", "👎"], horizontal=True)
               # st.write(response)

if __name__ == "__main__":
    obj = ChatbotTools()
    obj.main()