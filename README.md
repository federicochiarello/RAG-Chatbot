# RAG-Chatbot

Local RAG system that works with both PDF and CSV files. It uses Ollama to serve the embedding model (nomic-embed-text) and the LLM model (Llama3.1). 
The embedding are stored in a ChromaDB local database.

## Tools:
- [LangChain](https://www.langchain.com/)
- [Ollama](https://ollama.com/)
- [ChromaDB](https://www.trychroma.com/)

## Setup

```
# Clone the repository
git clone https://github.com/federicochiarello/RAG-Chatbot

# Install required libraries
python -m pip install -r requirements.txt
```

Save the pdf files under 'data/pdf/'.
Save the csv files under 'data/csv/'.

## Commands

```
# move to src directory
cd src
```

### Create (or update) the database:
```
python populate_database.py
```

### Ask question through terminal:
```
python query_data.py "your question"
```

### Interact with a Chatbot UI:
```
stremlit run chatbot_ui.py
```
