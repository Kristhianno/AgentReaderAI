import os
from crewai import Agent, Task, Crew, Process
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from crewai_tools import CSVSearchTool
#from crewai_tools import FileReadTool
from dotenv import load_dotenv

load_dotenv()

GroqApiKey = os.getenv('GROQ_API_KEY')

OpenaiApiKey = os.getenv('OPENAI_API_KEY')




Openai = ChatOpenAI (
                    api_key= OpenaiApiKey,
                    model= 'gpt-4o'
)





llama3 = ChatGroq(
            api_key= GroqApiKey,
            model="llama3-70b-8192"
        )



file_read_csv_tool = CSVSearchTool('Estoque.csv')

#file_read_txt_tool = FileReadTool(file_path= 'PraticasdeVendas.txt')







Diretor = Agent(
    role='Diretor Executivo',
    goal='Coordenar a demanda de vendas da equipe, retorne as mensagens em Português do Brasil',
    backstory="O Diretor é responsável e assertivo em suas decisões para otimizar faturamento e reduzir gastos.Retorne as mensagens em Português do Brasil",
    verbose= True,
    llm=Openai
)



Vendedor = Agent(
    role='Vendedor Comercial',
    goal='Pesquisar na web as melhores notícias para o analista possa identificar correlações entre elas e o documento.Retorne as mensagens em Português do Brasil',
    tools=[file_read_csv_tool],
    backstory="O vendedor é um profissional inteligente e comunicador,retorne as mensagens em Português do Brasil",
    verbose= True,
    llm=Openai
)




Analista = Agent(
    role='Analista de Dados',
    goal='Analisar os dados para tomada de decisão precisa do Diretor,retorne as mensagens em Português do Brasil',
    tools=[file_read_csv_tool],
    backstory="O Analista é um profissinal altamente requisitado para o time de negócio, retorne  as mensagens em Português do Brasil.",
    verbose= True,
    llm=Openai
)



coordenar_equipe = Task(
    description= " O Diretor deve coordenar a equipe, mantendo a comunicação e fornecendo suporte estratégico. Diretor deve ordenar primeiro que o Vendedor pesquise na web para que o Analista, analise o documento e possa trazer informações relevantes, retorne as mensagens em Português do Brasil" ,

    expected_output="Análise bem sucedida, temos a estratégia perfeita para a diretoria. Retorne as mensagens em Português do Brasil.",
    agent=Diretor,
    allow_delegation=True
)




Trazer_informacao = Task(
    description="O Vendedor deve trazer informações precisas para o Analista , retorne as mensagens em Português do Brasil.",
    expected_output="Todas as informações foram passadas para o Analista, caminho livre para trazer as melhores estratégias.",
    agent=Vendedor
)



Trazer_analises_precisas = Task(
    description="O Analista deve fornecer análises precisas do documento que venho do Vendedor obtendo insights valiosos. retorne as mensagens em Português do Brasil",
    expected_output="Análise conclúida com sucesso.",
    agent=Analista
)




# Inicialização da equipe
time_comercial = Crew(
    agents=[Diretor, Vendedor, Analista],
    tasks=[coordenar_equipe, Trazer_informacao, Trazer_analises_precisas],
    process=Process.hierarchical,
    manager_llm=Openai
)


# Exemplo de execução da equipe
resultado = Crew.kickoff(time_comercial)
print(resultado)