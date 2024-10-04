from crewai import Agent



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




Analista = Agent(
    role='Analista de Dados',
    goal=  "Analisar os dados do arquivo e pontuar quais são as estratégias mais usadas no mundo corporativo" 
           "retorne as mensagens em Português do Brasil",
    tools = [file_read_tool],
    backstory="O Analista é um profissinal altamente requisitado para o time de negócio, muito qualificado com PHD no MIT , retorne  as mensagens em Português do Brasil.",
    verbose= True,
    llm=llama3
)




Diretor = Agent(
    role='Diretor Comercial',
    goal='Decidir e executar a melhor estratégia para equipe de vendas,retorne as mensagens em Português do Brasil',
    backstory="O Diretor é um profissional altamente capacitado com PHD em Havard de Administração de Empresas" 
              "e muita experiência no segmento de vendas.Retorne as mensagens em Português do Brasil",
    verbose= True,    
    llm=llama3 
    )
