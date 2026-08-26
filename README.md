# Learning RAG with LangChain

This project is a hands-on guide to building a **Retrieval-Augmented Generation (RAG)** application. You will learn how documents become embeddings, how those embeddings are stored and searched, and how retrieved information is passed to a language model.

## What is RAG?

RAG combines two steps:

1. **Retrieval:** Find the most relevant pieces of information from your documents.
2. **Generation:** Give those pieces to an LLM so it can generate an answer grounded in your data.

Instead of asking an LLM to answer only from its training data, RAG lets it answer questions about your own files.

## Learning roadmap

### Step 1: Understand embeddings

An embedding is a list of numbers representing the meaning of text. Texts with similar meanings have vectors that are close together.

Experiment with sentences such as:

- `Python is a programming language.`
- `I enjoy writing Python code.`
- `The weather is sunny today.`

Compare their similarity scores and observe which sentences are closest.

### Step 2: Prepare documents

Create a `documents` folder and add a few `.txt` or `.pdf` files. Start with simple text files so you can focus on the concepts.

### Step 3: Split documents into chunks

Large documents are split into smaller overlapping pieces. Chunk size and overlap affect retrieval quality. Try different values and compare the results.

### Step 4: Create a vector store

Convert every chunk into an embedding and store the embeddings in ChromaDB. The original text is stored with each vector so it can be returned during a search.

### Step 5: Retrieve relevant chunks

When a user asks a question:

1. Embed the question.
2. Search for nearby document vectors.
3. Return the most relevant chunks.

### Step 6: Generate an answer

Build a prompt containing the retrieved chunks and the user’s question. Ask the LLM to answer only using the supplied context.

### Step 7: Evaluate your RAG system

Test questions whose answers are present in the documents and questions whose answers are absent. Check whether the system retrieves the right text and avoids inventing information.

## Setup

This project already contains a local virtual environment in `.venv`. Activate it in PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install the packages for the RAG project:

```powershell
pip install langchain langchain-community langchain-text-splitters langchain-chroma sentence-transformers pypdf
```

This project uses Ollama for local LLM responses. Install Ollama separately, then download a model:

```powershell
ollama pull llama3.2
```

Run the current starter program with:

```powershell
python main.py
```

## Suggested project structure

```text
LangChainProject/
├── documents/          # Your source files
├── data/               # Saved vector database
├── 01_embeddings.py    # Learn and compare embeddings
├── 02_ingest.py        # Load, split, embed, and store documents
├── 03_retrieve.py      # Search the vector store
├── 04_rag.py           # Ask questions using retrieved context
├── main.py             # Application entry point
└── README.md
```

## Core RAG flow

```text
Documents → Load → Split into chunks → Embed → Store vectors

Question → Embed → Similarity search → Retrieved context → LLM answer
```

## Experiments to try

- Change the chunk size and overlap.
- Retrieve 1, 3, and 5 chunks and compare answers.
- Compare different embedding models.
- Add metadata such as filename and page number.
- Ask questions that require information from multiple chunks.
- Deliberately add irrelevant documents and measure retrieval quality.
- Add citations showing which file and page supported the answer.

## Important terms

- **Embedding:** Numeric representation of meaning.
- **Vector store:** Database optimized for searching embeddings.
- **Chunk:** A small piece of a document.
- **Similarity search:** Finding vectors closest to a query vector.
- **Retriever:** Component that returns relevant chunks.
- **Context:** Retrieved text given to the LLM.
- **Hallucination:** An answer that is not supported by the available information.

## Completion goal

By the end, you should be able to explain every part of this pipeline and build a chatbot that answers questions from your own documents with source references.
