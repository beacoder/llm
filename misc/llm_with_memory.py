from langchain_ollama import ChatOllama
from langgraph.graph import START, END, StateGraph, add_messages
from typing import Annotated
from typing_extensions import TypedDict
import streamlit as st
import sys, copy, pdb


llm = ChatOllama(model="qwen2.5", temperature=0.25)

class GraphState(TypedDict):
    # add_messages to make sure messages is appended instead of overwrite.
    messages: Annotated[list, add_messages]

def load_messages_from_st(state: GraphState):
    temp_list = copy.deepcopy(st.session_state.messages)
    temp_list.extend(state["messages"])
    state["messages"] = temp_list

def save_messages_to_st(state: GraphState):
    st.session_state.messages.extend(state["messages"])

def call_1st_llm(state: GraphState):
    load_messages_from_st(state)
    response = llm.invoke(state["messages"])
    return {"messages": response}

def call_2nd_llm(state: GraphState):
    input_message = {"role": "user", "content": "这本书有几个章节？"}
    state["messages"].append(input_message)
    load_messages_from_st(state)
    response = llm.invoke(state["messages"])
    return {"messages": response}

def last_node(state: GraphState):
    # this node is only for save messages to st
    save_messages_to_st(state)
    return {"messages": []}

def redirect_output_to_streamlit():
    class StdOutRedirector:
        def write(self, msg):
            msg = msg.replace("\n", "\n\n")
            st.write(msg)
    sys.stdout = StdOutRedirector()
    sys.stderr = StdOutRedirector()

def init_session_state():
    if 'history' not in st.session_state:
        st.session_state.history = []
    if 'current_text' not in st.session_state:
        st.session_state.current_text = "这本书主要讲述的是什么故事？"
    if 'messages' not in st.session_state:
        st.session_state.messages = []
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

def init_agent():
    workflow = StateGraph(GraphState)
    workflow.add_node("call_1st_llm", call_1st_llm)
    workflow.add_node("call_2nd_llm", call_2nd_llm)
    workflow.add_node("last_node", last_node)

    workflow.add_edge(START, "call_1st_llm")
    workflow.add_edge("call_1st_llm", "call_2nd_llm")
    workflow.add_edge("call_2nd_llm", "last_node")
    workflow.add_edge("last_node", END)

    return workflow.compile()

def main():
    redirect_output_to_streamlit()
    init_session_state()

    user_input = st.text_area("Please input your question:", value=st.session_state.current_text, key="text_area")
    user_submit = st.button("Submit", key="input submit")

    if user_submit and user_input:
        app = init_agent()
        input_message = {"role": "user", "content": user_input}
        for output in app.stream({"messages": [input_message]}):
            pass
        for message in st.session_state.messages:
            try:
                message.pretty_print()
            except:
                print(message)
        save_to_state(user_input)


if __name__ == "__main__":
    main()
