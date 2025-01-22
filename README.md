# RAG-Chatbot

Local RAG system that works with PDF files. Ollama is used to serve the embedding model (nomic-embed-text) and the LLM model (suggested Llama3.1). 
The embedding are stored in a ChromaDB local database.

Two versions are provided:
- __Basic RAG System__: showcase a simple RAG implementation;
- __RAG-Agent Chatbot__: final implementation of a RAG-based LLM-Agent Chatbot.

## Tools:
- [LangChain](https://python.langchain.com/docs/introduction/)
- [LangGraph](https://langchain-ai.github.io/langgraph/tutorials/introduction/)
- [Ollama](https://ollama.com/)
- [ChromaDB](https://www.trychroma.com/)

## Models:
- [Llama-3.1-8B](https://huggingface.co/meta-llama/Llama-3.1-8B)
- [nomic-embed-text](https://ollama.com/library/nomic-embed-text)

## Setup
Clone the repository and install the required libraries.
```
git clone https://github.com/federicochiarello/RAG-Chatbot
python -m pip install -r requirements.txt
```

Install Ollama from this [link](https://ollama.com/). Then install the embedding model and the desired language model: 
```
ollama pull nomic-embed-text
ollama pull [llama-model]
```
Tested language models are: `llama3.2:1b`, `llama3.1`. 

Save the __pdf__ files under `data/pdf/` and the __csv__ files under `data/csv/`.



# Basic RAG System

Commands to use the simple RAG system.

Make sure there is a `.env` file in the root directory of the project.
Sample `.env` file:
```
PDF_PATH="path/to/pdfs"
CSV_PATH="path/to/csvs"
CHROMA_PATH="path/to/chroma.sqlite3"
COLLECTION_NAME="chatbot"
LLAMA_MODEL="llama3.2:1b"
```

## Commands
Move to `src/basic_rag_system/` directory
```
cd src/basic_rag_system/
```

### Create (or update) the database:
```
python populate_database.py
```

### Ask question through terminal:
```
python query_data.py "your question"
```

### Interact with the Chatbot UI:
```
streamlit run chatbot_ui.py
```


# RAG-Agent Chatbot

Commands to use the RAG-based LLM-Agent Chatbot implementation.

Make sure there is a `.env` file in the root directory of the project.
Sample `.env` file:
```
CHROMA_PATH="path/to/chroma.sqlite3"
COLLECTION_NAME="chatbot"
LANGSMITH_API_KEY="your_langsmith_key"  #Optional
```

## Commands
Move to `src/agent/` directory
```
cd src/agent/
```

### Interact with the Chatbot UI:
```
streamlit run chatbot_ui.py
```

