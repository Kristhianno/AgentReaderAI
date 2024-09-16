import os
from crewai import Agent, Task, Crew, Process
from langchain_groq import ChatGroq
from crewai_tools import FileReadTool
from dotenv import load_dotenv

load_dotenv()

GroqApiKey = os.getenv('GROQ_API_KEY')

