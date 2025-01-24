from doc_loader import load_text
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_ollama import ChatOllama
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os, pdb


load_dotenv()

embeddings = OllamaEmbeddings(model="nomic-embed-text")
llm = ChatOllama(model="qwen2.5", temperature=0)
# llm = ChatOpenAI(api_key=$deepseek_api_key,
#                  model="deepseek-chat",
#                  base_url='https://api.deepseek.com',
#                  temperature=0)

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

hallucination_template = """
You are a grader assessing whether an answer is grounded in / supported by a set of facts.
Give a binary score 'yes' or 'no' score to indicate whether an answer is grounded in / supported by a set of facts.
Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.

Here are the facts:
\n ------- \n
{documents}
\n ------- \n
Here is the answer: {generation}
"""

answer_template = """
You are a grader assessing whether an answer is useful to resolve a question.
Give a binary score 'yes' or 'no' score to indicate whether an answer is useful to resolve a question.
Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.

Here are the facts:
\n ------- \n
{generation}
\n ------- \n
Here is the question: {question}
"""

retriever_grader = PromptTemplate.from_template(rerank_template) | llm | JsonOutputParser()
rag_chain = PromptTemplate.from_template(qa_template) | llm | StrOutputParser()
hallucination_grader = PromptTemplate.from_template(hallucination_template) | llm | StrOutputParser()
answer_grader =PromptTemplate.from_template(answer_template) | llm | JsonOutputParser()

def index_documents():
    """
    Index documents and return a retriever.
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, separators=["。"])
    doc_splits = text_splitter.split_documents(load_text("/home/huming/download/JinPingMei.txt"))
    if os.path.exists("./chroma_db"):
        vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    else:
        vectorstore = Chroma.from_documents(documents=doc_splits, embedding=embeddings, persist_directory="./chroma_db")
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10})
    return retriever
