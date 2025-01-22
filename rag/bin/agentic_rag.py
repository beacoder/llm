from doc_loader import load_text, load_org_in_dir
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain_chroma import Chroma
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_ollama.llms import OllamaLLM
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, StateGraph
from operator import itemgetter
from pprint import pprint
from typing import List
from typing_extensions import TypedDict
import os, pdb


load_dotenv()

QWEN = "qwen2.5"
llm = OllamaLLM(model=QWEN)
embeddings = OllamaEmbeddings(model=QWEN)

questions = [
    "这本书主要讲的是什么故事?",
    "西门庆有哪些朋友?",
    "西门庆和几个女人有染，分别是谁?",
    "西门庆和他的女人们的最后结局是啥?",
]

qa_template = """
You are an assistant for question-answering tasks.
Answer the question based on the context below.
If you can't answer the question, reply "I don't know".

Context: {context}
Question: {question}
"""

rerank_template = """
You are a grader assessing relevance of a retrieved document to a user question.
If the document contains keywords related to the user question, grade it as relevant.
It does not need to be a stringent test. The goal is to filter our erroneous retrievals. \n
Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question. \n
Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.

Here is the retrieved document: \n\n {document} \n\n
Here is the user question: {question} \n
"""

hallucination_check_template = """
You are a grader assessing whether an answer is grounded in / supported by a set of facts.
Give a binary score 'yes' or 'no' score to indicate whether an answer is grounded in / supported by a set of facts.
Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.

Here are the facts:
\n ------- \n
{documents}
\n ------- \n
Here is the answer: {generation}
"""

answer_check_template = """
You are a grader assessing whether an answer is useful to resolve a question.
Give a binary score 'yes' or 'no' score to indicate whether an answer is useful to resolve a question.
Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.

Here are the facts:
\n ------- \n
{generation}
\n ------- \n
Here is the question: {question}
"""

def index_documents():
    """Index documents and return a retriever."""

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, separators=["。"], add_start_index=True)
    doc_splits = text_splitter.split_documents(load_text("/home/huming/download/JinPingMei.txt"))
    if os.path.exists("./chroma_db"):
        vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    else:
        vectorstore = Chroma.from_documents(documents=doc_splits, embedding=embeddings, persist_directory="./chroma_db")
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 25})
    return retriever

retriever = index_documents()
retriever_grader = PromptTemplate.from_template(rerank_template) | llm | JsonOutputParser()
rag_chain = PromptTemplate.from_template(qa_template) | llm | StrOutputParser()
hallucination_grader = PromptTemplate.from_template(hallucination_check_template) | llm | StrOutputParser()
answer_grader =PromptTemplate.from_template(answer_check_template) | llm | JsonOutputParser()

def generate_answer(retriever, question):
    """Generate appropriate answer to the question."""
    docs = retriever.invoke(question)
    answer = rag_chain.invoke({"context": docs, "question": question})
    print(question)
    print()
    print(answer)
    print()
    return [question, docs, answer]

def rerank_documents(question, docs):
    """Reranking for retrieved docs."""
    doc_txt = docs[1].page_content
    print(f"rerank_documents\n")
    print(retriever_grader.invoke({"question": question, "document": doc_txt}))
    print()

def hallucination_check(docs, answer):
    """Check if the answer makes sense or not."""
    print(f"hallucination_check \n")
    print(hallucination_grader.invoke({"documents": docs, "generation": answer}))
    print()

def answer_check(question, answer):
    """Check if the answer is good enough."""
    print(f"answer_check \n")
    print(answer_grader.invoke({"question": question, "generation": answer}))
    print()

def run_test():
    for question in questions:
        question, docs, answer = generate_answer(retriever, question)
        rerank_documents(question, docs)
        hallucination_check(docs, answer)
        answer_check(question, answer)


# simplified graph
#                                                                    / hallucination -> re-generate
#                                               relevant -> generate
# quesion -> retrieved docs -> grade documents /                     \ grade answer -> OK -> END
#                                              \                                    \ No -> web search -> re-generate
#                                               irrelevant -> web search -> generate
class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        answer:   LLM generation
        documents: list of documents
    """
    question : str
    answer: str
    documents: List[str]

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

    # always generate lots of text, so comment it out for now
    # grade = score["score"]
    grade = "yes"

    # Check hallucination
    if grade == "yes":
        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
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
    inputs = {"question": "这本书主要讲的是什么故事?",
              # "question": "西门庆有哪些朋友?",
              # "question": "西门庆和几个女人有染，分别是谁?",
              # "question": "西门庆和他的女人们的最后结局是啥?",
              }

    for output in app.stream(inputs):
        for key, value in output.items():
            pprint(f"Finished running: {key}:")
    print(value["generation"])


if __name__ == "__main__":
    # run_test()
    main()
