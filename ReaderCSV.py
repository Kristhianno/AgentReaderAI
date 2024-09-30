import os
from crewai import Agent, Task, Crew, Process
from langchain_groq import ChatGroq
#from langchain_openai import ChatOpenAI
#from crewai_tools import CSVSearchTool
from crewai_tools import FileReadTool
from dotenv import load_dotenv

load_dotenv()

GroqApiKey = os.getenv('GROQ_API_KEY')

'''
OpenaiApiKey = os.getenv('OPENAI_API_KEY')
Openai = ChatOpenAI (
                    api_key= OpenaiApiKey,
                    model= 'gpt-4o'
)

'''


llama3 = ChatGroq(
            api_key= GroqApiKey,
            model="groq/llama3-70b-8192"
        )



#file_read_csv_tool = CSVSearchTool()

file_read_txt_tool = FileReadTool(file_path= 'Representantesd.xlsx')









Diretor = Agent(
    role='Diretor Comercial',
    goal='Decidir e executar a melhor estratégia para equipe de vendas,retorne as mensagens em Português do Brasil',
    backstory="O Diretor é um profissional altamente capacitado com PHD em Havard de Administração de Empresas" 
              "e muita experiência no segmento de vendas.Retorne as mensagens em Português do Brasil",
    verbose= True,    
    llm=llama3 
    )



Analista = Agent(
    role='Analista de Dados',
    goal= "Analisar os dados do arquivo e pontuar quais são os top 10 produtos com mais estoque" 
            " disponível para venda,retorne as mensagens em Português do Brasil",
    tools=[file_read_txt_tool],
    backstory="O Analista é um profissinal altamente requisitado para o time de negócio, muito qualificado com PHD no MIT , retorne  as mensagens em Português do Brasil.",
    verbose= True,
    llm=llama3
)



Vendedor = Agent(
    role='Vendedor Comercial',
    goal='Pesquisar na web as melhores notícias para o analista possa identificar correlações entre elas e o documento.Retorne as mensagens em Português do Brasil',
    tools=[file_read_txt_tool],
    backstory="O vendedor é um profissional inteligente e comunicador,retorne as mensagens em Português do Brasil",
    verbose= True,
    llm=llama3
)



coordenar_equipe = Task(
    description=  " O Diretor deve coordenar a equipe, mantendo a comunicação e fornecendo suporte estratégico." 
                  "Diretor deve ordenar que o Analista faça uma pesquisa geral no documento e depois  "
                  " traga os top 10 produtos disponíveis em estoque para venda , retorne as mensagens em Português do Brasil" ,

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
    description="O Analista deve fornecer todos os dados encontrados no documento. retorne as mensagens em Português do Brasil",
    expected_output="Análise conclúida com sucesso.",
    agent = Analista
)



# Inicialização da equipe
time_comercial = Crew(
    agents=[Diretor,  Analista],
    tasks=[coordenar_equipe,  Trazer_analises_precisas],
    process=Process.hierarchical,
    manager_llm=llama3
)


# Exemplo de execução da equipe
resultado = Crew.kickoff(time_comercial)
print(resultado)