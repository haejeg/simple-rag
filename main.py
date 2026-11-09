import os
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from langchain_chroma.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

EMBEDDING_MODEL: str = 'llama3' # If changing this, make sure to also import the proper libraries
CHUNK_SIZE: int = 512
CHUNK_OVERLAP: int = 20
PERSIST_DIRECTORY = 'db'
LLM_NUM_PREDICT = 128
LLM_TEMPERATURE = 0.5

app = FastAPI(title='simple rag')

class QueryRequest(BaseModel):
    user_query: str

class QueryResponse(BaseModel):
    query: str
    response: str
    sources: str

def embeddings():
    embeddings = OllamaEmbeddings (
        model=EMBEDDING_MODEL,
    )
    return embeddings

def extract_text(path):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        is_separator_regex=False,
    )

    dataset = []

    with os.scandir(path) as es:
        for e in es:
            if e.is_file() and e.name.endswith('.md'):
                with open(e.path, 'r', encoding='utf-8') as file:
                    file_metadata = {'filename': file.name} 
                    dataset.extend(text_splitter.create_documents(texts=[file.read()], metadatas=[file_metadata])) 

    return dataset

def store_text(dataset):
    Chroma.from_documents(
        documents=[dataset[i] for i in range(len(dataset))],
        embedding=embeddings(),
        persist_directory=PERSIST_DIRECTORY,
    )

def generate_text(query):
    model = OllamaLLM(
        model = 'llama3',
        temperature = LLM_TEMPERATURE, 
        num_predict = LLM_NUM_PREDICT,
    )

    retriever = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings()).as_retriever()

    prompt = ChatPromptTemplate.from_template(""" Answer using ONLY the context below. If the context doesn't contain the answer,
        then mention that it doesn't. Cite the source file from the metadata as ONLY a file name ONLY in a section of its own.

        Context: {context}
    
        Question: {question}
    """)

    RAG = (
        {
        'context': retriever,
        'question': RunnablePassthrough()
        }
        | prompt
        | model
    )

    answer = RAG.invoke(query)
    return answer

store_text(extract_text('sample-data'))

print(generate_text('What makes a good ally?'))