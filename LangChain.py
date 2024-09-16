import streamlit as st
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import CSVLoader
from langchain.chains import llm
from dotenv import load_dotenv

load_dotenv ()


loader = CSVLoader(file_path= 'Estoque.csv')
documents = loader.load()