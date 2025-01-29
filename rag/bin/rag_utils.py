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
# llm = ChatOllama(model="qwen2.5", temperature=0.25)
llm = ChatOpenAI(api_key=$deepseek_api_key,
                 model="deepseek-chat",
                 base_url='https://api.deepseek.com',
                 temperature=0.25)

qa_template = """
You are an assistant for question-answering tasks.
Answer the following question using the provided context.
When answering, list the supporting facts and clearly indicate their source within the context.
If you can't answer the question, reply "I don't know".

Question: {question}
Context: {context}

**Question**:
<Provide the original question.>

**Answer**:
<Provide the answer to the question.>

**Supporting Facts**:
1. <Fact 1> (Source: <Exact location or reference in the context>)
2. <Fact 2> (Source: <Exact location or reference in the context>)
3. <Fact 3> (Source: <Exact location or reference in the context>)
...

**Example**:
Question: What is the capital of France?
Context: France is a country in Europe. Its capital is Paris, which is known for its art, fashion, and culture.

**Question**: What is the capital of France?
**Answer**: The capital of France is Paris.

**Supporting Facts**:
1. "Its capital is Paris" (Source: Second sentence of the context).
"""

rerank_template = """
You are a relevance evaluator. Your task is to rank the retrieved document based on their relevance to the question.
Assign a score from 0 to 10, where:
- **0-3**: Irrelevant or barely related to the query.
- **4-6**: Somewhat relevant but lacks key details or is only partially aligned with the query.
- **7-8**: Relevant and addresses the query well, but may have minor gaps or unnecessary details.
- **9-10**: Highly relevant, directly addresses the query, and provides comprehensive or precise information.

Provide the score as a JSON with a single key 'score' and provide the explanation as a JSON with a single key 'explanation'.

Here is the retrieved document: {document}
Here is the user question: {question}

**Examples**:

**Query:** "What are the benefits of regular exercise for mental health?"
**Document 1:** "Regular exercise improves cardiovascular health and reduces the risk of diabetes."
\'{{"score": 2, "explanation": "document focuses on physical health benefits, not mental health, making it irrelevant to the query.}}\'

**Document 2:** "Exercise can reduce symptoms of anxiety and depression by releasing endorphins."
\'{{"score": 9, "explanation": "The document directly addresses mental health benefits, aligning well with the query."}}\'

**Document 3:** "Mental health is influenced by diet, sleep, and exercise, but this article focuses on diet."
\'{{"score": 4, "explanation": "The document mentions exercise briefly but primarily focuses on diet, making it only somewhat relevant."}}\'

**Document 4:** "Regular exercise boosts mood, reduces stress, and improves cognitive function, all of which are critical for mental well-being."
\'{{"score": 10, "explanation": "The document comprehensively addresses the mental health benefits of exercise, directly matching the query."}}\'
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

rag_chain = PromptTemplate.from_template(qa_template) | llm | StrOutputParser()
retriever_grader = PromptTemplate.from_template(rerank_template) | llm | JsonOutputParser()
hallucination_grader = PromptTemplate.from_template(hallucination_template) | llm | JsonOutputParser()
answer_grader =PromptTemplate.from_template(answer_template) | llm | JsonOutputParser()

def index_documents(file_name):
    """
    Index documents and return a retriever.
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, separators=["。"])
    doc_splits = text_splitter.split_documents(load_text(file_name))
    vector_db_name = f"./chroma_db_{os.path.splitext(os.path.basename(file_name))[0]}"

    if os.path.exists(vector_db_name):
        vectorstore = Chroma(embedding_function=embeddings, persist_directory=vector_db_name)
    else:
        vectorstore = Chroma.from_documents(documents=doc_splits, embedding=embeddings, persist_directory=vector_db_name)
    # retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10}) # looking for "exact match" result
    retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={'k': 10, 'fetch_k': 50}) # looking for "relevant and diversity" result
    return retriever
