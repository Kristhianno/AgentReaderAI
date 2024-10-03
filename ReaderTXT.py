import os


from crewai import Agent, Task, Crew, Process
from langchain_groq import ChatGroq
from crewai_tools import FileReadTool, ScrapeWebsiteTool


from dotenv import load_dotenv
load_dotenv()


GroqApiKey = os.getenv('GROQ_API_KEY')

llama3 = ChatGroq( api_key= GroqApiKey, model="groq/llama3-70b-8192")




# Ferramentas que serão usadas pelos agentes
file_read_tool =FileReadTool(file_path= 'tecnicasVendas.txt')
WebScraping =ScrapeWebsiteTool(website_url= 'https://www.google.com')



Vendedor = Agent(
    role='Vendedor Comercial',

    goal='Pesquisar na internet as empresas do ramo de revestimento sintético e suas estratégias de marketing e vendas,' 
         'retorne 5 grandes empresas e pelo menos as 3 grandes estratégias de cada uma delas.' 
         'Retorne as mensagens em Português do Brasil',

    backstory="O vendedor é um profissional inteligente e comunicador,retorne as mensagens em Português do Brasil",

    tools= [WebScraping],
    
    verbose= True,
    
    llm=llama3
)


Trazer_informacao = Task(
    description="O Vendedor deve trazer informações precisas para o Analista , retorne as mensagens em Português do Brasil.",
    expected_output = "Todas as informações foram passadas para o Analista, caminho livre para trazer as melhores estratégias.",    
    agent=Vendedor
)



Analista = Agent(
    role='Analista de Dados',
    goal=  "Analisar os dados do arquivo e pontuar quais são as estratégias mais usadas no mundo corporativo" 
           "retorne as mensagens em Português do Brasil",
    tools = [file_read_tool],
    backstory="O Analista é um profissinal altamente requisitado para o time de negócio, muito qualificado com PHD no MIT , retorne  as mensagens em Português do Brasil.",
    verbose= True,
    llm=llama3
)


Trazer_analises_precisas = Task(
    description= "O Analista deve fornecer análises precisas de acordo com a pesquisa feita pelo vendedor, obtendo insights valiosos." 
                 "retorne as mensagens em Português do Brasil",
    
    expected_output="Análise conclúida com sucesso e encaminhada para o Diretor.",
    agent = Analista
)


Diretor = Agent(
    role='Diretor Comercial',
    goal='Decidir e executar a melhor estratégia para equipe de vendas,retorne as mensagens em Português do Brasil',
    backstory="O Diretor é um profissional altamente capacitado com PHD em Havard de Administração de Empresas" 
              "e muita experiência no segmento de vendas.Retorne as mensagens em Português do Brasil",
    verbose= True,    
    llm=llama3 
    )


coordenar_equipe = Task(
    description= " O Diretor deve coordenar a equipe, mantendo a comunicação e fornecendo suporte estratégico."
                 " Diretor deve ordenar primeiro que o Vendedor pesquise na web para que o Analista, analise o documento " 
                 " e possa trazer informações relevantes, retorne as mensagens em Português do Brasil" ,

    expected_output="Análise bem sucedida, temos a estratégia perfeita para a diretoria. Retorne as mensagens em Português do Brasil.",
    agent=Diretor
)


# Inicialização da equipe
time_comercial = Crew(
    agents=[Vendedor, Analista, Diretor],
    tasks=[Trazer_informacao, Trazer_analises_precisas, coordenar_equipe],
    process=Process.hierarchical,
    manager_llm=llama3
)


# Exemplo de execução da equipe
resultado = Crew.kickoff(time_comercial)
print(resultado)

