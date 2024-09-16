import os
#import streamlit as st
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
#from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import CSVLoader
#from langchain.chains import llm
from dotenv import load_dotenv

load_dotenv ()

OpenaiApiKey = os.getenv('OPENAI_API_KEY')



Openai = ChatOpenAI (
                    api_key= OpenaiApiKey,
                    model= 'gpt-3.5-turbo-16k-0613'
)




loader = CSVLoader(file_path= 'Estoque.csv')
documents = loader.load()

embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(documents, embeddings)


def retrieve_info(query):
    similiar_response = db.similarity_search(query, k=3)
    return [doc.page_content for doc in similiar_response]



