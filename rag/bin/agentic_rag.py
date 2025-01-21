import os, pdb
from dotenv import load_dotenv
from langchain_ollama.llms import OllamaLLM
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import DocArrayInMemorySearch
from operator import itemgetter
from langchain_chroma import Chroma
from doc_loader import load_text, load_org_in_dir


load_dotenv()

QWEN = "qwen2.5"
llm = OllamaLLM(model=QWEN)
embeddings = OllamaEmbeddings(model=QWEN)

questions = [
    "how to save llm cost?",
    "这本书主要讲的是什么故事?",
    "西门庆有哪些朋友?",
    "西门庆有几个老婆?",
]

qa_template = """
You are an assistant for question-answering tasks.
Answer the question based on the context below.
If you can't answer the question, reply "I don't know".

Context: \n\n {context} \n\n
Question: {question} \n
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

def index_documents():
    """Index documents and return a retriever."""

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, separators=["。"], add_start_index=True)
    doc_splits = text_splitter.split_documents(load_text("/home/huming/workspace/ai/ragtest/input/JinPingMei(5).txt"))
    if os.path.exists("./chroma_db"):
        vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    else:
        vectorstore = Chroma.from_documents(documents=doc_splits, embedding=embeddings, persist_directory="./chroma_db")
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 25})
    return retriever

def rerank_documents(retriever, question):
    """Reranking for retriever result."""

    prompt = PromptTemplate.from_template(rerank_template)
    rerank_chain = prompt | llm | JsonOutputParser()
    docs = retriever.invoke(question)
    doc_txt = docs[1].page_content
    print(f"score for: {question}\n")
    print(rerank_chain.invoke({"question": question, "document": doc_txt}))
    print()

def generate_answer(retriever, question):
    """Generate appropriate answer to the question."""

    prompt = PromptTemplate.from_template(qa_template)
    rag_chain = prompt | llm | StrOutputParser()
    docs = retriever.invoke(question)
    print(rag_chain.invoke({"context": docs, "question": question}))
    print()

def main():
    retriever = index_documents()
    for question in questions:
        rerank_documents(retriever, question)
        generate_answer(retriever, question)


if __name__ == "__main__":
    main()
