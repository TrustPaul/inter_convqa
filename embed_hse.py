
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pandas as pd
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import Docx2txtLoader
from langchain.document_loaders import TextLoader
from pathlib import Path
from langchain.vectorstores import Chroma
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import Chroma
#__import__('pysqlite3')
#import sys
#sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

documents = []
for file in os.listdir('HSE/'):
    if file.endswith('.pdf'):
        pdf_path = 'HSE/' + file
        loader = PyPDFLoader(pdf_path)
        documents.extend(loader.load())
    elif file.endswith('.docx') or file.endswith('.doc'):
        doc_path = './docs/' + file
        loader = Docx2txtLoader(doc_path)
        documents.extend(loader.load())
    elif file.endswith('.txt'):
        text_path = './docs/' + file
        loader = TextLoader(text_path)
        documents.extend(loader.load())



# Embeddings
model_name = "sentence-transformers/paraphrase-albert-small-v2"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}
embeddings = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)




def split_docs(documents,chunk_size=1000,chunk_overlap=200):
  text_splitter =  RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
  docs = text_splitter.split_documents(documents)
  return docs

docs = split_docs(documents)
print(len(docs))


persist_directory = "irish_documents_hse"

vectordb = Chroma.from_documents(
    documents=docs, embedding=embeddings, persist_directory=persist_directory
)

vectordb.persist()


query = 'What was contained in the Homeless Report 2022?'
vectordb = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
docs = vectordb.similarity_search(query)

print(docs)

