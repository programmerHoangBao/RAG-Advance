# RAG Advance
# Table of Contents

* [RAG Advance](#rag-advance)

  * [1. Overview](#1-overview)
  * [2. Dataset](#2-dataset)
  * [3. Architecture](#3-architecture)

    * [3.1. Semantic-Classification](#31-semantic-classification)
    * [3.2. EMBEDDING Data](#32-embedding-data)
    * [3.3. Vector Database](#33-vector-database)
    * [3.4. Generation Answer with LLM](#34-generation-answer-with-llm)
  * [4. Outcome](#4-outcome)
  * [5. Install Guide](#5-install-guide)

    * [5.1. Requirements](#51-requirements)
    * [5.2. Step-by-step guide](#52-step-by-step-guide)

## 1. Overview
Research and propose a 4-Layer Chatbot Architecture: Integrate a deep topic classification layer before executing the RAG process in order to guide the search space.

Develop and evaluate an enhanced RAG model with classification: Build a complete advanced RAG architecture that incorporates a multi-label data classification model, then conduct evaluations against the traditional RAG architecture (without classification) in terms of answer quality and system performance. Compare with the traditional RAG architecture to determine the specific areas where the advanced RAG architecture achieves improvements.

## 2. Dataset

Each sample contains the following attributes:

| Column        | Data Type | Description                                      |
| ------------- | --------- | ------------------------------------------------ |
| `question_id` | `int64`   | Unique identifier of the Stack Overflow question |
| `answer_id`   | `int64`   | Unique identifier of the corresponding answer    |
| `title`       | `str`     | Title of the Stack Overflow question             |
| `question`    | `str`     | Main textual content of the question             |
| `answer`      | `str`     | Content of the corresponding answer              |
| `tags`        | `object`  | List of one or more programming-related tags     |
| `q_score`     | `float64` | Score of the question                            |
| `a_score`     | `float64` | Score of the answer                              |
| `q_license`   | `str`     | License associated with the question             |
| `a_license`   | `str`     | License associated with the answer               |

All more **700,000 samples contain non-null values** for every column.

## 3. Architecture

![Architecture-RAG](/Architecture-RAG.png)

### 3.1. Semantic-Classification
- The model we are building is a deep learning architecture designed for the problem of multi-label classification on source code text or programming-related text.
- The architecture is constructed as a combination of:
  CodeBERT → Bi-LSTM → Multi-Head Self-Attention → Feed-Forward Network → Multi-Head Classification
- In this setup, CodeBERT is used as a pre-trained language model to generate contextual representations for the input sequence. Then, the Bidirectional LSTM (Bi-LSTM) learns sequential dependencies in both directions. Multi-Head Self-Attention further identifies important positions within the sequence. Finally, the extracted features are passed through multiple independent classification heads to simultaneously predict multiple labels.
- This model is suitable for tasks where a single data sample can belong to multiple classes at the same time. For example, a programming question may be assigned labels such as: 'python', 'java', 'c++',...
- Therefore, instead of selecting only one class, the model outputs independent probabilities for each label.
- Link to project development: [Semantic Classification of Stack Overflow Data](https://github.com/programmerHoangBao/Semantic-Classification-of-Stack-Overflow-Data.git)

### 3.2. EMBEDDING Data

After the Stack Overflow data is preprocessed, the question content is converted into semantic vector representations (embeddings) using the **SentenceTransformer** model. The embeddings represent textual content in a vector space, enabling the system to retrieve documents with semantic similarity to the user's query.

During the retrieval process, the system uses the instruction *“Represent this sentence for searching relevant passages:”* before the query to guide the embedding model toward semantic search. The resulting query vector is then used to retrieve relevant Stack Overflow questions and answers.

To reduce computational costs when the same query needs to be encoded multiple times, the system implements an **LRU Cache** for embeddings. The cache stores up to 100 recent query vectors and removes the least recently used entries when the capacity limit is exceeded.

### 3.3. Vector Database

The generated embeddings are stored and retrieved using **Qdrant Vector Database**. The system uses the `stackoverflow_questions` collection to manage vectors together with their associated metadata, including the question content (`question`), answer content (`answer`), and tags (`tags`).

When a user submits a query, its vector is used to perform semantic search in Qdrant. By default, the system retrieves the **Top-K = 5** most relevant results and applies a **score threshold = 0.80** to filter out documents with low similarity scores.

In addition to semantic similarity search, the system supports filtering based on `tags`. If the filtered search does not return sufficient results, the system performs a fallback search without the tag filter to increase the likelihood of retrieving relevant information.

The retrieved results contain the question, answer, and similarity score. These results are then combined into a context that is provided to the language model during the answer generation stage.

### 3.4. Generation Answer with LLM

After the retrieval process is completed, the relevant Stack Overflow documents are combined into a **context** that serves as the basis for answer generation. The system uses a language model loaded through `AutoModelForCausalLM` and `AutoTokenizer` from the Hugging Face Transformers library.

Before retrieval, the system performs **query rewriting** using the LLM to transform the current question into a standalone search query. This process uses up to two recent conversation turns to resolve pronouns, omitted subjects, and context-dependent references.

During the generation stage, the LLM receives the retrieved Stack Overflow posts as context together with the rewritten query. The system prompt instructs the model to provide direct, accurate, and concise answers based on the retrieved documents. The generation process uses `max_new_tokens = 1024`, `temperature = 0.3`, and `top_p = 0.9`, providing a balance between response flexibility and generation control.

## 4. Outcome
## 5. Install Guide
### 5.1. Requirements
### 5.2. Step-by-step guide
