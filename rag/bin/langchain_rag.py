import os
import sys
from dotenv import load_dotenv
from langchain_ollama.llms import OllamaLLM
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import DocArrayInMemorySearch
from operator import itemgetter
from langchain_chroma import Chroma

load_dotenv()

MODEL = "qwen2.5"
# MODEL = "mistral"

model = OllamaLLM(model=MODEL)
embeddings = OllamaEmbeddings(model=MODEL)
parser = StrOutputParser()
all_splits = None

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

def index_doc():
    global all_splits
    loader = TextLoader("/home/huming/workspace/ai/ragtest/input/JinPingMei.txt")
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, add_start_index=True)
    all_splits = text_splitter.split_documents(docs)
    # print(all_splits)

def query(retriever):
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

def query_in_db():
    if os.path.exists("./chroma_db"):
        vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    else:
        vectorstore = Chroma.from_documents(all_splits, embeddings)
        save_to_db = Chroma.from_documents(all_splits, embeddings, persist_directory="./chroma_db")
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})
    query(retriever)

def query_in_memory():
    vectorstore = DocArrayInMemorySearch.from_documents(all_splits, embedding=embeddings)
    retriever = vectorstore.as_retriever()
    query(retriever)

def handle_args():
    if len(sys.argv) == 2:
        global questions
        questions = [sys.argv[1]]

def main():
    handle_args()
    index_doc()
    # query_in_db()
    query_in_memory()


if __name__ == "__main__":
    main()
