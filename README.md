# Concert Tour Assistant (In-memory RAG)

This project is a Python-based Concert Tour Assistant designed to manage and retrieve concert tour documents. It uses Retrieval-Augmented Generation (RAG) to store, summarize, and query concert-tour-related data, all in memory for the duration of the session.

## Features

1. **Document Ingestion**: 
   - Supports PDF and TXT files.
   - Extracts text, checks for relevance (based on keywords), summarizes the content, and stores it in an in-memory index.

2. **Querying**: 
   - Users can ask questions related to ingested concert tour documents.
   - The assistant uses a combination of embeddings and extractive QA to provide answers.

3. **Models Used**: 
   - Sentence Transformers for text embeddings.
   - HuggingFace's DistilBART for text summarization.
   - DistilBERT for extractive question-answering.

## Requirements

Before running this project, you need to install the following dependencies. Use the `requirements.txt` below to install all necessary packages.

### Installation

1. Clone or download this repository.
2. Navigate to the project directory.
3. Create a Python virtual environment (optional but recommended):
   
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
