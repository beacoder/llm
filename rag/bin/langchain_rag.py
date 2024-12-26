import os
import sys
from dotenv import load_dotenv
from langchain_ollama.llms import OllamaLLM
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import DocArrayInMemorySearch
from operator import itemgetter
from langchain_chroma import Chroma
import orgparse
from doc_loader import (
    load_text,
    load_org_in_dir
)

load_dotenv()

MODEL = "qwen2.5"
# MODEL = "mistral"

model = OllamaLLM(model=MODEL)
embeddings = OllamaEmbeddings(model=MODEL)
parser = StrOutputParser()

template = """
You are an assistant for question-answering tasks.
Answer the question based on the context below.
If you can't answer the question, reply "I don't know".

Context: {context}

Question: {question}
"""

prompt = PromptTemplate.from_template(template)
print(f"Prompt: {prompt}")

questions = [
    "这本书主要讲的是什么故事?",
    "西门庆有哪些朋友?",
    "西门庆有几个老婆?",
]

def check_model():
    chain = model | parser
    chain.invoke("Tell me a joke")

def check_prompt():
    chain = prompt | model | parser
    print(chain.invoke({"context": "My parents named me Santiago", "question": "What's your name'?"}))

def query(retriever):
    print(f"Querying ...")
    print()
    rag_chain = (
        {
            "context": itemgetter("question") | retriever,
            "question": itemgetter("question"),
        }
        | prompt | model | parser
    )
    for question in questions:
        print(f"Question: {question}")
        print(f"Answer: {rag_chain.invoke({'question': question})}")
        print()

def query_in_db(all_splits):
    if os.path.exists("./chroma_db"):
        vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    else:
        vectorstore = Chroma.from_documents(all_splits, embeddings)
        save_to_db = Chroma.from_documents(all_splits, embeddings, persist_directory="./chroma_db")
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})
    query(retriever)

def query_in_memory(all_splits):
    vectorstore = DocArrayInMemorySearch.from_documents(all_splits, embedding=embeddings)
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})
    query(retriever)

def handle_args():
    if len(sys.argv) == 2:
        global questions
        questions = [sys.argv[1]]

def main():
    handle_args()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, add_start_index=True)
    # all_chunks = splitter.split_documents(load_org_in_dir("/home/huming/workspace/org"))
    all_chunks = splitter.split_documents(load_text("/home/huming/workspace/ai/ragtest/input/JinPingMei.txt"))
    # query_in_db(all_chunks)
    query_in_memory(all_chunks)


if __name__ == "__main__":
    main()
