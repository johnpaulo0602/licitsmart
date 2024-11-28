# Importação de módulos necessários
import os
import yaml
from crewai import Agent, Task, Crew, Process
from crewai_tools import PDFSearchTool
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# Carregar as variáveis de ambiente do arquivo .env
load_dotenv()

# Configuração das chaves de API
os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")

# Diretório contendo os PDFs dos editais
pdf_folder = r"PDFs"

# Definições do projeto LicitSmart
solicitacoes = """
<solicitacoes>
1 - OBJETIVOS - Identificação dos Objetivos do Edital: Identifique os objetivos principais do edital, resumindo-os em um parágrafo claro e conciso, explicando o que se busca contratar ou realizar.
2 - EXIGÊNCIAS - Identificação de Exigências Técnicas e Documentais: Liste as exigências técnicas e documentais necessárias para participação.
3 - OPORTUNIDADES - Identificação de Oportunidades no Edital: Identifique vantagens competitivas para o fornecedor, como critérios de desempate ou margem de desconto.
4 - PRAZOS - Extração de Prazos e Datas-Chave: Liste prazos importantes como envio de propostas, lances e questionamentos.
5 - LIMITAÇÕES - Identificação de Possíveis Limitações: Explique barreiras como prazos curtos ou requisitos técnicos complexos.
6 - RECOMENDAÇÕES - Recomendações para Participação: Sugira estratégias para melhorar as chances de sucesso na licitação.
7 - ANÁLISE CRÍTICA - Identifique inconsistências ou ambiguidades no edital.
</solicitacoes>
"""

controles = """
<controle>
1. Entonação: Formal técnico.
2. Foco de Tópico: Exclusivo no edital.
3. Língua: Português do Brasil.
4. Controle de Sentimento: Neutro e analítico.
5. Nível Originalidade: 10. Parafrasear totalmente.
6. Nível de Abstração: 2. Manter concreto e específico.
7. Tempo Verbal: Use o presente ou futuro conforme apropriado.
</controle>
"""

restricoes = """
<restricoes>
1. Não traduza termos técnicos amplamente aceitos.
2. Mantenha referências a artigos e leis no idioma original.
3. Não envolva o retorno do YAML com blocos de código como ```yaml.
4. Não insira comentários fora do conteúdo delimitado por YAML.
</restricoes>
"""

template = """
<template>
EDITAL:
- ARQUIVO: "nome_do_edital.pdf"
- OBJETIVOS: "Resumo dos objetivos do edital."
- EXIGÊNCIAS: "Lista das exigências técnicas e documentais."
- OPORTUNIDADES: "Identificação de possíveis vantagens no edital."
- PRAZOS: "Prazos e datas importantes."
- LIMITAÇÕES: "Descrição das limitações encontradas."
- RECOMENDAÇÕES: "Sugestões para aumentar as chances de sucesso."
- ANÁLISE CRÍTICA: "Análise crítica do edital."
</template>
"""

# Perfil da empresa
company_profile = """
<company_profile>
- NOME: "Central IT Tecnologia da Informação Ltda."
- LICENCAS: ["Licença Operacional de TI", "Alvará de Funcionamento"]
- CERTIFICACOES: ["ISO 9001", "ISO 27001", "CMMI Nível 3"]
- LOCALIZACAO: ["Brasília", "DF"]
- AREAS_DE_ESPECIALIZACAO: ["Desenvolvimento de Software", "Consultoria em TI", "Gestão de Serviços de TI"]
- EXPERIENCIA: ["Projetos Governamentais", "Sistemas Corporativos", "Soluções em Nuvem"]
- PROJETOS_REALIZADOS:
  - {"Nome do Projeto": "Portal de Serviços Públicos", "Cliente": "Ministério da Economia", "Valor": "4,000,000 BRL", "Ano": 2021}
  - {"Nome do Projeto": "Sistema de Gestão Educacional", "Cliente": "Secretaria de Educação", "Valor": "2,500,000 BRL", "Ano": 2020}
- REFERENCIAS_DE_CLIENTES: ["Depoimento de Cliente A", "Depoimento de Cliente B"]
- TECNOLOGIAS_UTILIZADAS: ["Java", "Python", "AWS", "Azure", "Microservices"]
- PREMIOS_E_RECONHECIMENTOS: ["Prêmio de Inovação em TI 2020", "Melhor Empresa de Software 2019"]
- CAPACIDADE_OPERACIONAL: "Capacidade para executar múltiplos projetos de grande porte simultaneamente"
- HISTORICO_DE_LICITACOES: {"Participacoes": 20, "Vitorias": 15}
- COMPLIANCE_E_REGULAMENTACAO: "Conformidade total com as normas legais e regulatórias, sem sanções registradas"
- COBERTURA_DE_SEGURO: ["Seguro de Responsabilidade Civil Profissional", "Seguro de Equipamentos"]
- ESTABILIDADE_FINANCEIRA: {"Receita Anual": "60,000,000 BRL", "Margem de Lucro": "12%"}
- IDIOMAS_SUPORTADOS: ["Português", "Inglês", "Espanhol"]
- LOCALIZACOES_ADICIONAIS: ["São Paulo, SP", "Rio de Janeiro, RJ"]
</company_profile>
"""

# Funções para criar agentes e tarefas
def create_agent_leitor(llm, tool):
    return Agent(
        role='Leitor de Editais',
        goal="Ler PDFs de editais e extrair informações específicas conforme definido nas solicitações em <solicitacoes>. "
             "Gerar um YAML de acordo com o modelo especificado em <template>. {solicitacoes} {template}.",
        backstory="Você é um especialista em leitura e análise de editais. Sua missão é identificar informações cruciais, "
                  "como objetivos, exigências, prazos e oportunidades, garantindo uma análise completa e organizada. "
                  "Seu papel é fundamental para auxiliar na tomada de decisão estratégica em licitações. "
                  "Ao responder às solicitações delimitadas por <solicitacoes></solicitacoes>, "
                  "você deve considerar os controles definidos em <controle></controle> e as restrições em <restricoes></restricoes>. "
                  "Siga rigorosamente as diretrizes estabelecidas em {solicitacoes}, {template}, {restricoes} e {controles}.",
        tools=[tool],
        verbose=True,
        memory=False,
        llm=llm
    )

def create_agent_revisor(llm):
    return Agent(
        role='Revisor de YAML de Editais',
        goal="Ler os dados extraídos pelo Agente Leitor e verificar se o YAML foi produzido de acordo com o template proposto em <template>, "
             "com todas as informações solicitadas em <solicitacoes>. Revisar e ajustar o YAML, se necessário, para garantir precisão. "
             "Como resultado, você deve retornar um YAML revisado no mesmo formato do template proposto. {solicitacoes} {template}.",
        backstory="Você é um especialista em revisão e análise de informações estruturadas em YAML. "
                  "Sua função é garantir que os dados extraídos pelos agentes estejam completos, precisos e formatados "
                  "conforme o modelo definido. Você também deve assegurar que todas as solicitações em <solicitacoes> sejam atendidas, "
                  "seguindo os controles de qualidade estabelecidos em <controle> e respeitando as restrições de <restricoes>. "
                  "Sua atenção aos detalhes assegura que as informações finais sejam consistentes e confiáveis para tomada de decisão.",
        verbose=True,
        memory=False,
        llm=llm
    )

def create_agent_analista(llm):
    return Agent(
        role='Analista de Compatibilidade',
        goal="Analisar o edital e o perfil da empresa fornecido em {company_profile}, comparando as exigências do edital com as capacidades e qualificações da empresa, "
             "e gerar um relatório detalhado que inclua uma porcentagem de chances que a empresa tem de ganhar aquele edital com base nas informações fornecidas.",
        backstory="Você é um especialista em análise de compatibilidade entre editais e perfis empresariais. "
                  "Sua missão é avaliar quão bem a empresa se encaixa nas exigências do edital, considerando licenças, certificações, localização, experiência e capacidade financeira. "
                  "Você deve fornecer um relatório detalhado com insights e recomendações, além de uma estimativa percentual das chances de sucesso.",
        verbose=True,
        memory=False,
        llm=llm
    )

def leitor_task(agent_leitor):
    return Task(
        description="Leia o PDF do edital e responda em YAML às solicitações definidas em <solicitacoes>, "
                    "usando o modelo definido em <template>. Certifique-se de que as informações sejam completas e organizadas. "
                    "{solicitacoes} {template}",
        expected_output="YAML com as informações extraídas do edital, atendendo todas as solicitações definidas em <solicitacoes> "
                        "e no formato do template em <template>. {solicitacoes} {template}.",
        agent=agent_leitor
    )

def revisor_task(agent_revisor):
    return Task(
        description="Revise o YAML produzido pelo Agente Leitor para garantir que ele esteja de acordo com o template definido em <template>, "
                    "e que contenha todas as informações solicitadas em <solicitacoes>. Ajuste erros ou omissões identificadas. "
                    "{solicitacoes} {template}.",
        expected_output="YAML revisado e ajustado, seguindo o formato do template definido em <template> e cobrindo todas as solicitações "
                        "definidas em <solicitacoes>. {solicitacoes} {template}.",
        agent=agent_revisor
    )

def analista_task(agent_analista):
    return Task(
        description="Utilize o edital em YAML fornecido pelo agente anterior e o perfil da empresa em {company_profile} para realizar a análise. "
                    "Compare as exigências e requisitos do edital com as capacidades e qualificações da empresa, "
                    "e elabore um relatório detalhado. O relatório deve incluir uma estimativa percentual das chances que a empresa tem de ganhar o edital, "
                    "justificando os pontos fortes e fracos, e sugerindo possíveis ações para aumentar as chances de sucesso.",
        expected_output="Relatório detalhado com análise comparativa entre o edital e o perfil da empresa, incluindo a porcentagem de chance de sucesso.",
        agent=agent_analista
    )

# Processar os PDFs
pdf_files = [f for f in os.listdir(pdf_folder) if f.endswith('.pdf')]
all_editais = []

for pdf_file_name in pdf_files:
    # Configurando os agentes e ferramentas
    gpt = ChatOpenAI(model="gpt-4-turbo")
    pdf_path = os.path.join(pdf_folder, pdf_file_name)
    pdf_tool = PDFSearchTool(pdf_path)

    # cria os agentes
    agent_leitor = create_agent_leitor(gpt, pdf_tool)
    agent_revisor = create_agent_revisor(gpt)
    agent_analista = create_agent_analista(gpt)

    # criar as tarefas
    task_leitor = leitor_task(agent_leitor)
    task_revisor = revisor_task(agent_revisor)
    task_analista = analista_task(agent_analista)

    # Configurar a eqipe
    crew = Crew(
        agents=[agent_leitor, agent_revisor, agent_analista],
        tasks=[task_leitor, task_revisor, task_analista],
        process=Process.sequential
    )

    # Executar o Crew
    inputs = {
        'solicitacoes': solicitacoes,
        'template': template,
        'restricoes': restricoes,
        'controles': controles,
        'company_profile': company_profile
    }
    results = crew.kickoff(inputs=inputs)

    print(results)
    print("Chaves disponíveis em 'results':", results.keys())
    print("Salvar resultados em arquivo\n\n\n")

    output_file = f"{os.path.splitext(pdf_file_name)[0]}_relatorio.md"
    with open(output_file, 'w', encoding='utf-8') as file:
        file.write(results)

    print(f"Resultados salvos em {output_file}")
