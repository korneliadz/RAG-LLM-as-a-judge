# Advanced RAG — University Regulations Assistant

A Retrieval-Augmented Generation (RAG) system for querying university regulations using natural language. The project retrieves relevant document fragments, generates answers grounded in the retrieved context, and evaluates answer quality using an LLM-as-a-judge.

## Overview

University regulations are long and often difficult to search manually. This project explores how a RAG pipeline can make institutional documents easier to query while providing traceability between generated answers and their sources.

The system currently works with documents from **Lodz University of Technology (Politechnika Łódzka)**, including:

- *Regulamin Studiów PŁ* — study regulations
- *Prawo o szkolnictwie wyższym i nauce* — Polish Law on Higher Education and Science

The assistant can retrieve relevant passages, use multiple tools to gather additional context, and return an answer with source references. An evaluation component checks whether the answer is correct and supported by the retrieved evidence.

## Key Features

- **Semantic retrieval** using vector embeddings.
- **Document-grounded generation** with an instruction to answer using retrieved evidence.
- **Tool calling** for vector search, metadata search, document lookup, chunk retrieval, and neighboring-chunk retrieval.
- **Source traceability** through document metadata and retrieved chunk IDs.
- **LLM-as-a-judge evaluation** of correctness, faithfulness, source quality, completeness, and hallucination.
- **Execution tracing** to inspect tool calls, search queries, retrieved sources, and evaluation results.
- **Local inference** using Ollama.

## Architecture

```text
User question
     |
     v
Ollama — Qwen2.5 7B
     |
     v
Tool selection
     |
     +---- Semantic vector search
     +---- Metadata search
     +---- Document / chunk lookup
     +---- Neighboring chunk retrieval
     |
     v
Weaviate vector database
     |
     v
Retrieved document fragments
     |
     v
Generated answer
     |
     v
LLM-as-a-judge
     |
     v
Answer + execution trace + evaluation
```

## Tech Stack

| Component | Technology |
|---|---|
| Programming language | Python |
| Language model | Qwen2.5 7B via Ollama |
| Embedding model | `intfloat/multilingual-e5-base` |
| Vector database | Weaviate |
| Document processing | pypdf |
| Semantic retrieval | Vector similarity search |
| Evaluation | LLM-as-a-judge |
| Local services | Docker Compose |

## RAG Pipeline

1. **Document processing** — PDF documents are converted into text and divided into chunks.
2. **Metadata extraction** — chunks retain document information, including source, document ID, document type, date, issuing authority, and available article or paragraph references.
3. **Embedding generation** — text chunks are encoded using `multilingual-e5-base`.
4. **Indexing** — chunk text, metadata, and vectors are stored in Weaviate.
5. **Retrieval** — the language model can call tools to search for relevant fragments or retrieve additional context.
6. **Answer generation** — the model generates an answer using the retrieved information.
7. **Evaluation** — a separate judge evaluates the answer against the retrieved source fragments.

## Evaluation

The project includes an LLM-as-a-judge component that evaluates generated answers using the following criteria:

| Metric | Purpose |
|---|---|
| Correctness | Does the answer correctly address the question? |
| Faithfulness | Are the claims supported by the retrieved evidence? |
| Source quality | Are the retrieved sources relevant and appropriate? |
| Answer completeness | Does the answer cover the important parts of the question? |
| Hallucination detection | Does the answer contain unsupported claims? |
| Groundedness | Is the answer sufficiently grounded in the provided sources? |

The system also records an execution trace containing tool calls, arguments, retrieval queries, retrieved chunk IDs, and source documents. This makes it possible to inspect not only the final answer but also how the system arrived at it.

## Example Results

The following examples illustrate the system's behavior on two different questions: a question about a specific university regulation and an unrelated or nonsensical question.

### Example 1 — Question about missing ECTS credits

**Question**

> ile moge miec ects braku

**Generated answer**

> [Paste the complete generated answer here.]

**Retrieved evidence**

Paste the relevant source fragments returned by the retriever. Include the document title, paragraph or article number when available, and the text needed to verify the answer.

> [Paste retrieved fragment 1 here.]

> [Paste retrieved fragment 2 here.]

**Judge evaluation**

Paste the JSON returned by the judge.

```json
{
  "correctness": 0,
  "faithfulness": 0,
  "source_quality": 1,
  "answer_completeness": 0,
  "hallucination_detected": true,
  "grounded_in_sources": false
}
```

*The values above illustrate the recorded evaluation for this particular run. The generated answer claimed a limit of 15 ECTS and cited §21(6), but the retrieved fragments did not substantiate those claims.*

**Observation**

This example demonstrates why retrieval alone is not sufficient: a system may retrieve relevant documents and still generate claims that are not supported by the passages it receives. The judge flags the unsupported answer, while the execution trace helps identify the evidence available to the model.

### Example 2 — Unrelated or nonsensical question

**Question**

> [Paste the second question here.]

**Generated answer**

> [Paste the generated answer here.]

**Retrieved evidence**

> [Paste the relevant retrieved fragments here, or state that no relevant fragments were retrieved.]

**Judge evaluation**

```json
{
  "correctness": "...",
  "faithfulness": "...",
  "source_quality": "...",
  "answer_completeness": "...",
  "hallucination_detected": "...",
  "grounded_in_sources": "..."
}
```

Replace the placeholder values with the actual judge output.

**Observation**

[Describe whether the system appropriately declined to answer, retrieved irrelevant information, or generated unsupported claims.]

### Comparison

| Aspect | ECTS question | Unrelated question |
|---|---|---|
| Query type | University-regulation question | [Describe query type] |
| Retrieval behavior | Multiple tool calls | [Describe observed behavior] |
| Relevant evidence found | [Yes / partial / no] | [Yes / partial / no] |
| Answer supported by evidence | No, in this recorded run | [Result] |
| Judge detected hallucination | Yes | [Result] |

These examples demonstrate how retrieval traces and judge evaluations can be used together to analyze both answer quality and failure modes.

## Project Structure

```text
Advanced-RAG/
├── compose.yml
├── requirements.txt
└── src/
    ├── ask.py
    ├── rag.py
    ├── tools.py
    ├── judge.py
    ├── build_index.py
    ├── restore_index.py
    ├── add_documents.py
    ├── all_chunks.pkl
    └── documents/
        ├── Regulamin studiów w PŁ (zmiany od 1.10.2026).pdf
        └── Prawo o szkolnictwie wyzszym i nauce.pdf
```

## Setup and Usage

### Requirements

- Python 3.12
- Docker and Docker Compose
- Ollama
- The `qwen2.5:7b` model downloaded in Ollama

### 1. Create a virtual environment

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

### 2. Start Weaviate

```bash
docker compose up -d
```

### 3. Download the language model

```bash
ollama pull qwen2.5:7b
```

Make sure the Ollama service is running.

### 4. Run the assistant

```bash
cd src
python ask.py
```

Enter a question in the terminal. Type `exit` to stop the application.

> **Note:** The vector database must contain the indexed documents before querying. Indexing and restoration scripts are available in `src/`. Do not run indexing scripts unless you intend to create or update the index.

## Limitations

- Answer quality depends on the relevance and completeness of retrieved chunks.
- The model may generate unsupported claims even when relevant documents are retrieved.
- LLM-as-a-judge is an additional model-based evaluation step and is not a guarantee of factual