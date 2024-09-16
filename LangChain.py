import os
#import streamlit as st
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
#from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import CSVLoader
#from langchain.chains import llm
from dotenv import load_dotenv
import pandas as pd

dataset = pd.read_csv('Estoque.csv')


print(dataset)


display(dataset)


'''


load_dotenv ()

os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')

llm = ChatOpenAI(model="gpt-3.5-turbo-16k-0613")

loader = CSVLoader(file_path= 'Estoque.csv')
documents = loader.load()



'''