import os
#import streamlit as st
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
#from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import CSVLoader
#rom langchain.chains import LLChains


from dotenv import load_dotenv

load_dotenv ()

os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')

llm = ChatOpenAI(model="gpt-3.5-turbo-16k-0613", temperature=0)

loader = CSVLoader(file_path= 'Estoque.csv')
documents = loader.load()

embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(documents, embeddings)

print(db)




"""

def retrieve_info(query):
    similar_response = db.similarity_search(query, k=3)
    return [doc.page_content for doc in similar_response]

template = '''
                Você é um analista de dados e esta apto para responder {message} 
           '''


prompt = PromptTemplate(
    input_variables=["message"],
    template= template
)

chain = LLChains(llm=llm, prompt=prompt)
def generate_response(message):
    mensagem = retrieve_info(message)
    response = chain.run(message = message, messagem = mensagem)
    return response


generate_response("quais produtos com a maior quantidade em estoque?")

"""