from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import BaseMessage
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from typing import List
from typing_extensions import TypedDict
import os
import streamlit as st
import sys
from docker_tools import (get_user_id, check_path_existence, upload_file_to_docker,
                          check_file_existence, download_file_from_docker)


# react_agent @see https://langchain-ai.github.io/langgraph/how-tos/create-react-agent
#
# Task -> Agent -> Tool_Calls -> Tool -> Tool_Result -> Done
#           ^                                |
#           |--------------------------------+
#
# Execution:
# 1.The agent node calls the LLM with the messages list (after applying the messages modifier).
# 2.If the resulting AIMessage contains tool_calls, the graph will then call the tools.
# 3.The tools node executes the tools (1 tool per tool_call) and adds the responses to the messages list as ToolMessage objects.
# 4.The agent node then calls the LLM again.
# 5.The process repeats until no more tool_calls are present in the response.
# 6.The agent then returns the full list of messages as a dictionary containing the key "messages".

# LLM settings
llm = ChatOllama(model="qwen2.5", temperature=0.25)
# llm = ChatOpenAI(api_key="",
#                  model="deepseek-chat",
#                  base_url='https://api.deepseek.com',
#                  temperature=0.25)

task_prompt = """
You are an AI assistant equipped with a set of tools to complete tasks.\n
Your goal is to execute tasks in the correct order, ensuring each step is completed accurately before moving to the next.\n
Follow these instructions precisely:\n
1.Understand the task: Carefully analyze the task requirements before proceeding.\n
2.Select the appropriate tool: Choose the most suitable tool from the provided list to accomplish the task.\n
3.Execute the task: Use the selected tool to perform the task step-by-step.\n
4.Verify the output: Check if the result meets the task's requirements. If not, retry or adjust your approach.\n
5.Proceed to the next task: Only move to the next task after successfully completing the current one.
"""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a large language model and a helpful assistant. Respond concisely."),
        ("placeholder", "{messages}"),
    ]
)

# increase recursion_limit to avoid agent stops prematurely
RECURSION_LIMIT=1000

class AgentState(TypedDict):
    messages: List[BaseMessage]

def format_for_model(state: AgentState):
    return prompt.invoke({"messages": state["messages"]})

def redirect_output_to_streamlit():
    class StdOutRedirector:
        def write(self, msg):
            msg = msg.replace("\n", "\n\n")
            st.write(msg)
    sys.stdout = StdOutRedirector()
    sys.stderr = StdOutRedirector()

def init_tools(use_local_tool = True):
    if use_local_tool:
        from local_tools import (create_file, make_directory, run_command, run_script)
    else:
        from docker_tools import (create_file, make_directory, run_command, run_script)
    return [create_file, make_directory, run_command, run_script]

def init_session_state():
    if 'history' not in st.session_state:
        st.session_state.history = []
    if 'current_text' not in st.session_state:
        st.session_state.current_text = "list current directory."
    # UI config
    st.sidebar.title("History")
    for idx, entry in enumerate(st.session_state.history):
        if st.sidebar.button(entry, key=f"history_{idx}"):
            st.session_state.current_text = entry
            st.rerun()
    if st.sidebar.button("Clear History"):
        st.session_state.history = []
        st.rerun()

def save_to_state(user_input: str):
    if user_input and user_input not in st.session_state.history:
        st.session_state.history.append(user_input)

def do_upload(upload_path, uploaded_file):
    user_id = get_user_id()
    temp_file = os.path.expanduser(f"~/{user_id}_{uploaded_file.name}")
    with open(temp_file, 'wb') as ul_file:
        ul_file.write(uploaded_file.read())
    upload_file_to_docker(temp_file, f"{upload_path}/{uploaded_file.name}")
    os.remove(temp_file)

def do_download(download_file):
    file_name = "".join(os.path.splitext(os.path.basename(download_file)))
    user_id = get_user_id()
    temp_file = os.path.expanduser(f"~/{user_id}_{file_name}")
    download_file_from_docker(download_file, temp_file)
    download_data = None
    with open(temp_file, 'rb') as dl_file:
        download_data = dl_file.read()
    os.remove(temp_file)
    if download_data:
        st.download_button(
            label="Download File",
            data=download_data,
            file_name=f"{file_name}",
            mime="application/octet-stream"
        )

# entry point
def run_agent(user_input: str, tools):
    agent = create_react_agent(llm, tools, state_modifier=format_for_model)
    inputs = {"messages": [("system", task_prompt), ("user", user_input)]}
    for s in agent.stream(inputs, stream_mode="values", config={"recursion_limit": RECURSION_LIMIT}):
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()
    save_to_state(user_input)

def main():
    redirect_output_to_streamlit()
    init_session_state()

    use_local_tool = True

    # UI
    st.title("Agentic Tool")
    user_input = st.text_area("Please input your task:", value=st.session_state.current_text, key="text_area")
    user_submit = st.button("Submit", key="input submit")

    if use_local_tool:
        # Logic
        if user_submit and user_input:
            run_agent(user_input, init_tools(use_local_tool))
    else:
        # UI
        upload_path = st.text_input("Please input upload path:", value="/")
        uploaded_file = st.file_uploader("Upload files")
        download_file = st.text_input("Please input download file full path:")
        download_submit = st.button("Submit", key="download submit")
        # Logic
        if user_submit and user_input:
            run_agent(user_input, init_tools(use_local_tool))
        elif download_submit and download_file:
            if check_file_existence(download_file):
                do_download(download_file)
            else:
                st.write("The download file not exist.")
        elif upload_path and uploaded_file:
            if check_path_existence(upload_path):
                do_upload(upload_path, uploaded_file)
            else:
                st.write("The upload path not exist.")


if __name__ == "__main__":
    main()
