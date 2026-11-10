import os
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from langchain_chroma.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

EMBEDDING_MODEL: str = 'hf.co/CompendiumLabs/bge-base-en-v1.5-gguf' # If changing this, delete the db folder and re-run store_text
GENERATIVE_MODEL: str = 'llama3'
CHUNK_SIZE: int = 512
CHUNK_OVERLAP: int = 100
PERSIST_DIRECTORY = 'db'
DATA_DIRECTORY = 'sample-data' # Folder of .md files that /ingest reads from
LLM_NUM_PREDICT = 512
LLM_TEMPERATURE = 0.5
SEARCH_K = 20 # Candidates fetched by vector search
TOP_N = 5 # Chunks kept after reranking

app = FastAPI(title='simple rag')

class QueryRequest(BaseModel):
    user_query: str

class QueryResponse(BaseModel):
    query: str
    response: str
    sources: List[str]

'''
    RAG Functions (LangChain)
'''

def embeddings():
    '''
        Embeds the text
        Parameters:
            - Embedding Model
    '''
    embeddings = OllamaEmbeddings (
        model=EMBEDDING_MODEL,
    )
    return embeddings

def extract_text(path):
    '''
        Given a path, chunks it and returns a dataset
        Parameters:
            - Chunk Size
            - Chunk Overlap
    '''
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
    '''
        Stores text from dataset into persistent directory (db in our case)
    '''
    Chroma.from_documents(
        documents=[dataset[i] for i in range(len(dataset))],
        embedding=embeddings(),
        persist_directory=PERSIST_DIRECTORY,
    )

# Loaded once at startup instead of on every query
reranker = CrossEncoderReranker(model=HuggingFaceCrossEncoder(model_name='BAAI/bge-reranker-base'), top_n=TOP_N)

def generate_text(query):
    '''
        Given a query, uses the stored text as a retriever and generates an answer.
        Parameters:
            - Model
            - Temperature
            - Num Predict
    '''
    model = OllamaLLM(
        model = GENERATIVE_MODEL,
        temperature = LLM_TEMPERATURE,
        num_predict = LLM_NUM_PREDICT,
    )

    base = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings())\
        .as_retriever(search_kwargs={'k': SEARCH_K})

    retriever = ContextualCompressionRetriever(base_compressor=reranker, base_retriever=base)

    prompt = ChatPromptTemplate.from_template(""" Answer using ONLY the context below. If the context doesn't contain the answer,
        then mention that it doesn't.

        Context: {context}

        Question: {question}
    """)

    docs = retriever.invoke(query)
    context = '\n\n'.join(d.page_content for d in docs)

    answer = (prompt | model).invoke({'context':context, 'question':query})

    sources = sorted({os.path.basename(d.metadata['filename']) for d in docs})

    return answer, sources


'''
    REST Interfaces
'''
@app.post('/query', response_model=QueryResponse)
def query(request: QueryRequest):
    '''
        Sends a Query to the AI
    '''
    answer, sources = generate_text(request.user_query)
    return QueryResponse(query=request.user_query, response=answer, sources=sources)

@app.delete('/db')
def remove_db():
    '''
        Deletes the stored chunks through Chroma instead of deleting the db folder,
        since the folder's files can still be open (and locked on Windows) while the server runs.

        TODO: Think of ways to manage the real db directory instead of the Chroma cache
    '''
    Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings()).delete_collection()
    return {'status': 'db cleared'}

@app.post('/ingest')
def ingest():
    '''
        Rebuilds the db from DATA_DIRECTORY: clears the old chunks first so nothing is duplicated,
        then chunks, embeds and stores every .md file in the folder. 

        TODO: Custom data directories
    '''
    Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings()).delete_collection()
    dataset = extract_text(DATA_DIRECTORY)
    store_text(dataset)
    return {'status': 'db rebuilt', 'chunks': len(dataset)}

if __name__ == '__main__':
    # Only runs with `python main.py`, not when the server imports this file.
    # store_text adds to the db every time, so only run it on a fresh/empty db.
    store_text(extract_text(DATA_DIRECTORY))

    answer, sources = generate_text('What makes a good ally?')
    print(answer)
    print(sources)