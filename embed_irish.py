
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pandas as pd
from pathlib import Path
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import Docx2txtLoader
from langchain.document_loaders import TextLoader
from langchain.text_splitter import TokenTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pandas as pd
from pathlib import Path
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import Docx2txtLoader
from langchain.document_loaders import TextLoader
from langchain.text_splitter import TokenTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.document_loaders import DataFrameLoader
#__import__('pysqlite3')
#import sys
#sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

df_created = pd.read_csv('gov_citzen_docs_version_two.csv')

print(df_created.shape)
data = df_created.loc[(df_created['ans_len'] >= 200) & (df_created['ans_len'] <= 100000)]
loader = DataFrameLoader(data, page_content_column="texts")

documents = loader.load()


# Embeddings
model_name = "sentence-transformers/paraphrase-albert-small-v2"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}
embeddings = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)




def split_docs(documents,chunk_size=2000,chunk_overlap=50):
  text_splitter =  RecursiveCharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap,
    length_function = len,
    separators=[ ""])
  docs = text_splitter.split_documents(documents)
  return docs

docs = split_docs(documents)
print(len(docs))


persist_directory = "irish_government_documents"

vectordb = Chroma.from_documents(
    documents=docs, embedding=embeddings, persist_directory=persist_directory
)

vectordb.persist()


query = 'What was contained in the Homeless Report 2022?'
vectordb = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
docs = vectordb.similarity_search(query)

print(docs)

