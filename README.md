# Retrieval-Augmented Generation (RAG) System for Source Code Question Answering

## Overview

This project implements a multi-layer Retrieval-Augmented Generation (RAG) architecture designed for source code and software engineering question answering. The system combines topic classification, semantic embedding, vector retrieval, and large language model generation to provide accurate, context-aware responses while minimizing hallucinations.

The architecture consists of four sequential processing layers, from query understanding to final response generation.

---

# System Architecture

## Layer 1: Intent and Topic Classification

### Function

When a user submits a query, a classification model immediately analyzes the input and assigns one or more relevant topic tags.

### Purpose

Pre-identifying the topic helps narrow the search space, reduce irrelevant information retrieval, minimize hallucinations, and improve retrieval efficiency.

---

## Layer 2: Semantic Embedding

### Technology

**BAAI/bge-base-en-v1.5**

### Function

The user's query is transformed into dense vector embeddings that capture its semantic meaning.

### Purpose

Unlike traditional keyword matching approaches, BGE embeddings enable the system to understand the underlying intent and meaning of the query, improving retrieval accuracy.

---

## Layer 3: Vector Search and Retrieval

### Technology

**Qdrant Vector Database**

### Function

The system performs Hybrid Search, combining:

* Semantic vector similarity search
* Tag-based filtering using the topics identified in Layer 1

### Purpose

This layer retrieves the most relevant document chunks and contextual information to support answer generation.

---

## Layer 4: Response Generation

### Technology

**Qwen/Qwen2.5-Coder-7B-Instruct**

### Function

The language model receives:

* The original user query
* Retrieved contextual information from Layer 3

The model then analyzes, summarizes, and reformulates the information into a coherent response.

### Purpose

Qwen2.5-Coder-7B-Instruct serves as the reasoning and generation engine, producing natural-language answers that remain grounded in the retrieved knowledge base while supporting complex technical and programming-related tasks.

---

# Workflow

1. User submits a question.
2. Topic Classification identifies relevant tags.
3. Query is converted into semantic embeddings.
4. Qdrant performs hybrid retrieval using embeddings and topic filters.
5. Relevant document chunks are returned.
6. Qwen2.5-Coder-7B-Instruct generates the final answer based on the retrieved context.

---

# Required Models

Before running the system, download the following pretrained models:

| Model                          | Purpose              |
| ------------------------------ | -------------------- |
| Qwen/Qwen2.5-Coder-7B-Instruct | Response Generation  |
| BAAI/bge-base-en-v1.5          | Semantic Embedding   |
| microsoft/codebert-base        | Topic Classification |

---

# Installation

## 1. Clone the Repository

```bash
git clone <repository_url>
cd <project_directory>
```

## 2. Install Dependencies

Install all required Python packages:

```bash
pip install -r requirements.txt
```

## 3. Download Required Models

Download and configure the following models:

* Qwen/Qwen2.5-Coder-7B-Instruct
* BAAI/bge-base-en-v1.5
* microsoft/codebert-base

Ensure the model paths are correctly configured in the project settings.

---

# Running the Application

Start the system by executing:

```bash
python main.py
```

After startup, the application will be ready to receive user queries and generate context-aware responses.

---

# Key Features

* Multi-layer RAG architecture
* Topic-aware retrieval
* Semantic search using BGE embeddings
* Hybrid retrieval with Qdrant
* Context-grounded response generation
* Reduced hallucination rate
* Optimized for software engineering and source code related tasks

---

# Technologies Used

* Python
* PyTorch
* CodeBERT
* BGE Embeddings
* Qdrant Vector Database
* Qwen2.5-Coder-7B-Instruct
* Retrieval-Augmented Generation (RAG)

---

# License

This project is intended for research and educational purposes.
Please review the licenses of all third-party models and libraries before commercial use.
