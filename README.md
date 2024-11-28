# LicitSmart - Análise Automatizada de Editais

Este projeto realiza a análise automatizada de editais em PDF, extraindo informações relevantes e comparando-as com o perfil de uma empresa para gerar um relatório detalhado com a porcentagem de chances de ganhar a licitação.

## Pré-requisitos

- **Python 3.10** ou superior instalado em seu sistema.
- Conta na **OpenAI** com uma chave de API válida.

## Passo a Passo para Configuração

### 1. Clonar o Repositório

Clone o repositório do projeto para o seu ambiente local:

```bash
git clone https://github.com/johnpaulo0602/licitsmart.git
```

### 2. Navegar para o Diretório do Projeto

```bash
cd licitsmart
```

### 3. Criar um Ambiente Virtual

Crie um ambiente virtual para isolar as dependências do projeto:

```bash
python -m venv venv
```

Ative o ambiente virtual:

- **No Windows:**

  ```bash
  venv\Scripts\activate
  ```

- **No Linux/MacOS:**

  ```bash
  source venv/bin/activate
  ```

### 4. Instalar as Dependências

Instale as bibliotecas necessárias usando o arquivo `requirements.txt`:

```bash
pip install -r requirements.txt
```

### 5. Configurar a Chave de API da OpenAI

Crie um arquivo `.env` na raiz do projeto e adicione sua chave de API da OpenAI:

```env
OPENAI_API_KEY=your-openai-api-key
```

Substitua `your-openai-api-key` pela sua chave de API real.

### 6. Preparar os Arquivos PDF

Coloque os arquivos PDF dos editais que deseja analisar na pasta `PDFs` na raiz do projeto. Se a pasta não existir, crie-a:

```bash
mkdir PDFs
```

### 7. Executar o Script

Execute o script principal:

```bash
python script.py
```

Os resultados serão salvos em arquivos `.md` correspondentes a cada edital analisado.

## Arquivos Importantes

- `script.py`: Script principal que realiza a análise dos editais.
- `requirements.txt`: Lista das dependências necessárias para o projeto.
- `.env`: Arquivo que contém a variável de ambiente com a chave de API da OpenAI.
- `PDFs/`: Diretório que deve conter os arquivos PDF dos editais.

## Conteúdo do `requirements.txt`

O arquivo `requirements.txt` deve conter as seguintes bibliotecas:

```txt
crewai
crewai-tools
python-dotenv
langchain
openai
PyYAML
```

## Observações

- Certifique-se de que sua chave de API da OpenAI está ativa e tem permissões para usar o modelo GPT-4.
- Verifique se todos os arquivos necessários estão nos diretórios corretos.