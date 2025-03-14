from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import BaseMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from typing import List
from typing_extensions import TypedDict
import os, subprocess


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

@tool
def create_file(path: str, filename: str, content: str):
    "Create a new file in a path with the specified content."
    try:
        expanded_path = os.path.expanduser(path + "/" + filename)
        with open(expanded_path, 'w') as file:
            file.write(content)
            return f"Created file {filename} in {path}"
    except Exception as e:
        return str(e)

@tool
def make_directory(name: str, parent: str):
    "Create a new directory with the given name in the specified parent directory."
    try:
        dir_path = parent + "/" +name
        expanded_dir_path = os.path.expanduser(dir_path)
        os.makedirs(expanded_dir_path, exist_ok=True)
        return f"Directory name created/verified in parent"
    except Exception as e:
        return str(e)

# super-powerful, capable of replacing numerous existing tools.
@tool
def run_command(command: str):
    "Run a command."
    try:
        return subprocess.run(command, shell=True, capture_output=True, text=True).stdout
    except Exception as e:
        return str(e)

@tool
def run_script(script_program: str, script_file: str, script_args: str):
    "Run the script along with its specified arguments using the program."
    try:
        return subprocess.run([script_program, script_file,script_args], capture_output=True, text=True).stdout
    except Exception as e:
        return str(e)

task1= """
1.create a new directory '~/workspace/ai/testing'
2.create 3 text files in this directory, each file with a poetry with at least 100 words.
"""

task2 = """
1.create ~/workspace/ai/tool_script.py, it has following functions: it will print current time-stamp
2.run the script ~/workspace/ai/tool_script.py
"""

task3= """
1.Create a minesweeper game using html,css,js in directory '~/workspace/ai/minesweeper'
2.run the game with http server
"""

def main():
    tools = [create_file, make_directory, run_command, run_script]
    agent = create_react_agent(llm, tools, state_modifier=format_for_model)
    inputs = {"messages": [("user", task_prompt + task1)]}

    # increase recursion_limit to avoid agent stops prematurely
    for s in agent.stream(inputs, stream_mode="values", config={"recursion_limit": 100}):
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()


if __name__ == "__main__":
    main()
