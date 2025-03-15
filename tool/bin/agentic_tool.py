from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import BaseMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from typing import List
from typing_extensions import TypedDict
import os, subprocess, sys
import streamlit as st


# react_agent @see https://langchain-ai.github.io/langgraph/reference/prebuilt/#langgraph.prebuilt.chat_agent_executor.create_react_agent
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
class AgentState(TypedDict):
    messages: List[BaseMessage]

def format_for_model(state: AgentState):
    return prompt.invoke({"messages": state["messages"]})


# redirect all output to streamlit
class StdOutRedirector:
    def write(self, msg):
        msg = msg.replace("\n", "\n\n")
        st.markdown(msg)

sys.stdout = StdOutRedirector()
sys.stderr = StdOutRedirector()


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
5.Proceed to the next task: Only move to the next task after successfully completing the current one.\n\n
Tasks:\n
"""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a large language model and a helpful assistant. Respond concisely."),
        ("placeholder", "{messages}"),
    ]
)

# default timeout for executing command/script is 5 minutes
SHELL_TIMEOUT=300

# increase recursion_limit to avoid agent stops prematurely
RECURSION_LIMIT=100


# tools definitions
# @note remember to return either a result or a message to inform the LLM
@tool
def create_file(path: str, filename: str, content: str):
    """Create a new file in a path with the specified content.

    Args:
        path:     The directory where to create the file
        filename: The name of the file to create
        content:  The content to write to the file
    """
    try:
        expanded_path = os.path.expanduser(path + "/" + filename)
        with open(expanded_path, 'w') as file:
            file.write(content)
            return f"Created file {filename} in {path}"
    except Exception as e:
        return f"Failed to create file {filename} in {path}: {e}"

@tool
def make_directory(name: str, parent: str):
    """Create a new directory with the given name in the specified parent directory.

    Args:
        name:   The name of the new directory to create, e.g. testdir
        parent: The parent directory where the new directory should be created, e.g. /tmp
    """
    try:
        dir_path = parent + "/" +name
        expanded_dir_path = os.path.expanduser(dir_path)
        os.makedirs(expanded_dir_path, exist_ok=True)
        if os.path.exists(expanded_dir_path):
            return f"Directory {name} created/verified in {parent}"
        else:
            return f"Failed to create directory {name} in {parent}"
    except Exception as e:
        return f"Failed to create directory {name} in {parent}: {e}"

# super-powerful, capable of replacing numerous existing tools.
@tool
def run_command(command: str):
    """Run a command.

    Args:
        command: Command to run
    """
    try:
        result = subprocess.run(command,
                                check=True,
                                shell=True,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE,
                                timeout=SHELL_TIMEOUT)
        return result.stdout.decode()
    except subprocess.CalledProcessError as e:
        return f"Failed to run {command}: {e}"
    except subprocess.TimeoutExpired as e:
        return f"Run {command} timed out"

@tool
def run_script(script_program: str, script_file: str, script_args: str):
    """Run the script along with its specified arguments using the program.

    Args:
        script_program: Program to run the the script
        script_file:    Path to the script to run.  Supports relative paths and ~
        script_args:    Args for script to run
    """
    try:
        result = subprocess.run([script_program, script_file,script_args],
                                check=True,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE,
                                text=True,
                                timeout=SHELL_TIMEOUT)
        return result.stdout
    except subprocess.CalledProcessError as e:
        return f"Failed to run {script_file}: {e}"
    except subprocess.TimeoutExpired as e:
        return f"Run {script_file} timed out"


# streamlit settings
def prepare_history():
    if 'history' not in st.session_state:
        st.session_state.history = []
    if 'current_text' not in st.session_state:
        st.session_state.current_text = "list current directory."
    st.sidebar.title("History")
    for idx, entry in enumerate(st.session_state.history):
        if st.sidebar.button(entry, key=f"history_{idx}"):
            st.session_state.current_text = entry
            st.rerun()
    if st.sidebar.button("Clear History"):
        st.session_state.history = []
        st.rerun()

def save_to_history(user_input: str):
    if user_input and user_input not in st.session_state.history:
        st.session_state.history.append(user_input)
        st.session_state.current_text = user_input


# enter point
def run_agent(user_input: str):
    tools = [create_file, make_directory, run_command, run_script]
    agent = create_react_agent(llm, tools, state_modifier=format_for_model)
    inputs = {"messages": [("user", task_prompt + user_input)]}
    for s in agent.stream(inputs, stream_mode="values", config={"recursion_limit": RECURSION_LIMIT}):
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()

def main():
    prepare_history()
    st.title("Agentic Tool")
    user_input = st.text_area("Please input your task:", value=st.session_state.current_text, key="text_area")
    if st.button("Submit") and user_input:
        save_to_history(user_input)
        run_agent(user_input)


if __name__ == "__main__":
    main()
