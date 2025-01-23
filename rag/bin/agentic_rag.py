from langchain.schema import Document
from langchain_core.vectorstores.base import VectorStoreRetriever
from langgraph.graph import END, StateGraph
from pprint import pprint
from rag_utils import *
from typing import List
from typing_extensions import TypedDict
import os, pdb


# simplified graph
#                                                                     / hallucination -> re-generate
#                                                relevant -> generate
# question -> retrieved docs -> grade documents /                     \ grade answer --> OK -> END
#                                               \                                    \-> No -> web search -> re-generate
#                                                irrelevant -> web search -> generate
class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        web_search: whether to add search
        documents: list of documents
        retriever: retriever for vectorstore
    """
    question : str
    generation : str
    web_search : str
    documents : List[str]
    retriever : VectorStoreRetriever

def retrieve(state):
    """
    Retrieve documents from vectorstore

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    print("---RETRIEVE---")
    question = state["question"]
    retriever = state["retriever"]

    # Retrieve
    documents = retriever.invoke(question)
    return {"documents": documents, "question": question}

def grade_documents(state):
    """
    Determine whether the retrieved documents are relevant to the question
    If any document is not relevant, we will set a flag to run web search

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Filter out irrelevant documents and updated web_search state
    """

    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["question"]
    documents = state["documents"]

    # Score each doc
    filtered_docs = []
    web_search = "No"
    for d in documents:
        score = retriever_grader.invoke({"question": question, "document": d.page_content})
        grade = score['score']
        # Document relevance
        if grade.lower() == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(d)
        # Document not relevant
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
            web_search = "Yes"
            continue
    return {"documents": filtered_docs, "question": question, "web_search": web_search}

def generate(state):
    """
    Generate answer using RAG on retrieved documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]

    # RAG generation
    generation = rag_chain.invoke({"context": documents, "question": question})
    return {"documents": documents, "question": question, "generation": generation}

def web_search(state):
    """
    Web search based on question

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    print("---WEB SEARCH---")
    question = state["question"]
    documents = state["documents"]

    # TODO: add more logic later on
    return {"documents": documents, "question": question}

def decide_to_generate(state):
    """
    Determines whether to generate an answer, or add web search

    Args:
        state (dict): The current graph state

    Returns:
        str: Binary decision for next node to call
    """
    print("---ASSESS GRADE DOCUMENTS---")
    question = state["question"]
    web_search = state["web_search"]
    filtered_docs = state["documents"]

    if web_search == "Yes":
        print("---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, INCLUDE WEB SEARCH---")
        return "websearch"
    else:
        print("---DECISION: GENERATE---")
        return "generate"

def grade_generation_v_documents_and_question(state):
    """
    Determines whether to retry generation

    Args:
        state (dict): The current graph state

    Returns:
        str: Binary decision for next node to call
    """
    print("---CHECK HALLUCINATION---")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]

    score = hallucination_grader.invoke({"documents": documents, "generation": generation})

    # currently, since llm can't generate a score, so default this to 'yes'
    # grade = score["score"]
    grade = "yes"

    # Check hallucination
    if grade == "yes":
        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        # Check question-answering
        print("---GRADE GENERATION VS QUESTION---")
        score = answer_grader.invoke({"question": question, "generation": generation})
        if grade == "yes":
            print("---DECISION: GENERATION ADDRESS QUESTION---")
            return "useful"
        else:
            print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
            return "not useful"
    else:
        print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
        return "not supported"

def main():
    workflow = StateGraph(GraphState)

    # Define the nodes
    workflow.add_node("websearch", web_search)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("generate", generate)

    # Build the graph
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {
            "websearch": "websearch",
            "generate": "generate",
        },
    )
    workflow.add_edge("websearch", "generate")
    workflow.add_conditional_edges(
        "generate",
        grade_generation_v_documents_and_question,
        {
            "not supported": "generate",
            "useful": END,
            "not useful": "websearch",
        },
    )

    # Compile
    app = workflow.compile()

    # Test
    questions = ["这本书主要讲的是什么故事?",
                 "西门庆有哪些朋友?",
                 "西门庆和几个女人有染，分别是谁?",
                 "西门庆和他的女人们的最后结局是啥?",
                 ]
    retriever = index_documents()

    for q in questions:
        input_state = {"question": q, "retriever": retriever}
        for output in app.stream(input_state):
            for key, value in output.items():
                pprint(f"Finished running: {key}:")
        print(f'question:\n{value["question"]}')
        print(f'answer:\n{value["generation"]}')


if __name__ == "__main__":
    main()
