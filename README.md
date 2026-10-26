# Advanced RAG — University Regulations Assistant

A Retrieval-Augmented Generation (RAG) system for asking questions about university regulations in natural language. An LLM agent decides which retrieval tools to call, answers using only the retrieved fragments with article/paragraph citations, and a second LLM call ("LLM-as-a-judge") evaluates whether the answer is supported by the evidence.

Everything runs locally: Ollama for the language model, Weaviate in Docker for the vector database.

# Quick Demo



## Overview

University regulations are long, full of cross-references and hard to search manually. This project explores how a RAG pipeline can make such documents easier to query while keeping every answer traceable to its source.

The system currently works with two documents:

- *Regulamin Studiów Politechniki Łódzkiej*: study regulations of Lodz University of Technology (consolidated text with amendments effective 1 October 2026)
- *Prawo o szkolnictwie wyższym i nauce*: the Polish Law on Higher Education and Science

Questions can be asked in Polish, including student slang (e.g. "kolos" = test, "oblać" = fail).

## Key Features

- **Hybrid retrieval**: BM25 keyword search combined with vector similarity (Weaviate `hybrid`, `alpha=0.5`).
- **Agentic tool calling**: the model chooses between vector search, metadata search (fetch a paragraph/article by number), chunk lookup, neighboring-chunk retrieval and document lookup.
- **Structure-aware chunking**: documents are split along their legal structure (`§` for the regulations, `Art.` for the act), with long sections split further with overlap.
- **Cleaning of the consolidated text**: footnotes and superseded wording are removed so only the version in force is indexed (`clean_regulamin.py`).
- **Source traceability**: every chunk keeps document title, ID, type, date, issuing authority and its paragraph/article number.
- **LLM-as-a-judge** scoring correctness, faithfulness, source quality, completeness and hallucination.
- **Execution trace** of every tool call, search query and retrieved source.
- **Bounded agent loop**: at most 5 tool rounds, then the model must answer from what it has.

## Architecture

```text
User question
     |
     v
Qwen3.5 35B (Ollama)  <-------------------+
     |                                    |
     |  tool calls (max 5 rounds)         |
     v                                    |
+-----------------------------------+     |
| search_vector       (hybrid)      |     |
| search_by_metadata  (§ / Art. no.)|-----+  retrieved fragments
| get_chunk / get_neighboring_chunks|
| get_document                      |
+-----------------------------------+
     |
     v
Weaviate (BM25 + vectors from multilingual-e5-base)

Final answer with citations
     |
     v
LLM-as-a-judge (same model, temperature 0)
     |
     v
Answer + execution trace + evaluation JSON
```

## Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.12 |
| Language model | Qwen3.5 35B via Ollama |
| Embedding model | `intfloat/multilingual-e5-base` (sentence-transformers) |
| Vector database | Weaviate 1.39 (Docker Compose) |
| Retrieval | Hybrid search (BM25 + vector) |
| PDF processing | pypdf + regex-based cleaning and chunking |
| Evaluation | LLM-as-a-judge (JSON output) |

Developed and tested on a MacBook Pro with M4 Pro and 48 GB unified memory.

## RAG Pipeline

1. **Text extraction**: PDFs are converted to text with pypdf and whitespace is normalised.
2. **Cleaning**: in the study regulations, footnotes and wording valid only until 30 September 2026 are removed, so each provision appears once, in its current form.
3. **Chunking**: text is split at real `§` / `Art.` headings (cross-references such as "w § 16" are ignored). Sections longer than 1500 characters are split at sentence boundaries with 150 characters of overlap.
4. **Metadata**: each chunk stores source, document ID, type, date, issuing authority, paragraph/article number and its position in the document (`chunk_index`).
5. **Embedding and indexing**: chunks are encoded with `multilingual-e5-base` and stored in Weaviate together with their metadata.
6. **Agentic retrieval**: the model searches, follows cross-references ("o którym mowa w § 21") with `search_by_metadata`, and extends incomplete fragments with `get_neighboring_chunks`.
7. **Answer generation**: the answer must be based only on retrieved fragments and cite the paragraph/article each claim comes from.
8. **Evaluation**: the judge receives the question, the answer, the retrieved fragments and the trace, and returns scores as JSON.

## Evaluation

| Metric | Question it answers |
|---|---|
| Correctness | Is the answer correct according to the retrieved sources? |
| Faithfulness | Is every claim supported by the retrieved fragments? |
| Source quality | Are the retrieved sources relevant and appropriate? |
| Answer completeness | Does the answer cover what the sources say about the question? |
| Hallucination detected | Does the answer contain claims not present in the fragments? |
| Grounded in sources | Is the answer, overall, grounded in the provided evidence? |

The judge is instructed to ignore its own general knowledge: a claim that is true in the real world but absent from the retrieved fragments counts as unsupported. This measures **faithfulness to the retrieved evidence**, not retrieval recall. If the retriever misses the relevant passage, an honest "not found" answer can still score well (see *Limitations*).

## Example Results

### Example 1: Regulation question (missing ECTS credits)

**Question**

> ile moge miec ects braku *(how many ECTS credits can I be missing?)*

**Generated answer**

> Zgodnie z **Regulaminem Studiów PŁ (§ 21 ust. 6)**, limity dopuszczalnych braków punktowych przy rozliczaniu postępów w nauce na wszystkich formach studiów wynoszą:
> - **po semestrze zimowym** – 15 punktów ECTS
> - **po roku studiów** – 15 punktów ECTS

**Trace**: `search_vector -> search_vector -> search_vector`

**Judge evaluation**

```json
{
  "correctness": 3,
  "faithfulness": 3,
  "source_quality": 3,
  "answer_completeness": 3,
  "hallucination_detected": false,
  "grounded_in_sources": true
}
```

**Observation**

The answer is supported by the retrieved fragment of § 21 ust. 6. However, it quotes the wording valid **until 30 September 2026**. The consolidated PDF contains the old and the new wording side by side, so the judge could not detect the problem either, because the outdated wording is also "in the sources". This run motivated the cleaning step (`clean_regulamin.py`). After cleaning, § 21 ust. 6 contains only the current rule: a single limit of 15 ECTS per semester.

Zgodnie z **Regulaminem Studiów PŁ (§ 21)**, dopuszczalny limit braków punktowych wynosi **15 punktów ECTS**. Ten limit obowiązuje przy rozliczaniu postępów w nauce na wszystkich formach studiów i kierunkach studiów.

Prorektor może, w porozumieniu z Samorządem Studenckim, dokonać zmiany tych limitów, ale zmiana musi być ogłoszona nie później niż na jeden miesiąc przed rozpoczęciem roku akademickiego, od którego ma obowiązywać. W przypadkach szczególnych Prodziekan może również indywidualnie zmienić dla studenta ten limit.

### Example 2: Broad question (student benefits)

**Question**

> jakie swiadczenia przysluguja studentom *(what benefits are students entitled to?)*

**Trace**: `search_vector -> search_by_metadata (Art. 86) -> search_vector`

The model listed the benefits from Art. 86 ust. 1 and the student dormitories and canteens from Art. 11 of the act, fetching Art. 86 directly by number.

**Judge evaluation**

```json
{
  "correctness": 2,
  "faithfulness": 1,
  "source_quality": 3,
  "answer_completeness": 2,
  "hallucination_detected": true,
  "grounded_in_sources": false
}
```

**Observation**

The content was correct, but one claim (students form the majority of the scholarship committee) was attributed to **§ 33 of the regulations** instead of **Art. 86 ust. 3 of the act**. The judge caught the misattributed citation. It is a good example of why citations need to be checked, not just the answer text. The system prompt was then changed to require citing the exact fragment each claim comes from.

### Example 3: Out-of-scope question

**Question**

> przepis na naleśniki *(a pancake recipe)*

**Generated answer**: the system declined, stating that the documents do not contain the information.

W dokumentach nie ma wystarczających informacji o przepisie na naleśniki. Dostępne dokumenty dotyczą prawa o szkolnictwie wyższym i nauce (ustawa Sejm RP z 2018 r.) i zawierają jedynie przepisy prawne dotyczące m.in. wynagrodzeń pracowników uczelni, nagród jubileuszowych oraz tworzenia uczelni publicznych.

**Judge evaluation**

```json
{
  "correctness": 3,
  "faithfulness": 3,
  "source_quality": 0,
  "answer_completeness": 2,
  "hallucination_detected": false,
  "grounded_in_sources": true
}
```

**Observation**

The refusal was correct, but the evaluation is inconsistent: `grounded_in_sources: true` with no sources at all, and `source_quality: 0` / `answer_completeness: 2` penalise a correct refusal. The metrics are designed for content answers and need a separate path for refusals (see *Roadmap*).

### Comparison

| Aspect | ECTS question | Benefits question | Out-of-scope question |
|---|---|---|---|
| Tool calls | 3 × vector search | search + fetch by article | — |
| Relevant evidence found | Yes | Yes | Not applicable |
| Answer supported by evidence | Yes (outdated wording) | Mostly, one wrong citation | Correct refusal |
| Judge verdict | Correct | Flagged hallucination | Correct, but inconsistent fields |

## Development Notes

The most interesting part of the project was debugging it. Main problems found and how they were solved:

| Problem | Cause | Fix |
|---|---|---|
| Only one of the two documents was searchable | Indentation bug: `all_chunks.extend()` ran after the document loop, so only the last document was indexed | Moved into the loop |
| 125 of 263 chunks labelled "§ 1", 121 chunks containing only footnotes | Splitting at every `§`, including references like "Dodany przez § 1 pkt 13" | Split only at real headings (`§ N` followed by `1.` or a capital letter) |
| Answers quoting provisions no longer in force | The consolidated PDF keeps old and new wording side by side (67 places) | `clean_regulamin.py` removes superseded wording and footnotes |
| Labels such as `§ 23118` | Footnote numbers glued to paragraph numbers in the PDF text | Splitting glued numbers during cleaning |
| 15–19 tool calls per question, slow answers | Pure vector search often missed the relevant paragraph, so the model kept searching | Hybrid search (BM25 + vectors): typically 1–3 calls |
| Answers cut off mid-sentence | Context window filled with retrieved chunks | Larger `num_ctx` and a limit of 5 tool rounds |
| Endless tool loops and empty answers | `temperature=0` (greedy decoding) is not recommended for Qwen in thinking mode | Default sampling for the agent; temperature 0 kept only for the judge |
| Model never used tools other than search | Tool descriptions said *what* a tool does, not *when* to use it | Docstrings and prompt describe when to follow cross-references |
| Judge marking correct answers as unsupported | Judge received a truncated subset of fragments | Fragment selection and larger context |

## Project Structure

```text
Advanced-RAG/
├── compose.yml              # Weaviate service
├── requirements.txt
├── ask.py                   # interactive assistant: agent loop, trace, judge call
├── tools.py                 # retrieval tools exposed to the model
├── judge.py                 # LLM-as-a-judge
├── rag.py                   # PDF extraction, chunking, embeddings, Weaviate indexing
├── clean_regulamin.py       # removes footnotes and superseded wording
├── build_index.py           # full (re)build of the index from PDFs
├── restore_index.py         # restores Weaviate from all_chunks.pkl (no re-embedding)
├── add_documents.py         # appends new documents to the index
├── all_chunks.pkl           # cached chunks with embeddings
└── source-documents/        # needed only for build_index.py / add_documents.py
    ├── Regulamin studiów w PŁ (zmiany od 1.10.2026).pdf
    └── Prawo o szkolnictwie wyzszym i nauce.pdf
```

`ask.py` does not read the PDFs. It only queries Weaviate. The PDFs are needed only to build the index.

## Setup and Usage

### Requirements

- Python 3.12
- Docker and Docker Compose
- Ollama with the `qwen3.5:35b` model (roughly 24 GB of memory for the model alone)

### 1. Virtual environment

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
ollama pull qwen3.5:35b
```

### 4. Build or restore the index

```bash
python build_index.py      # first run, or after changing PDFs / chunking code
# or
python restore_index.py    # Weaviate was reset, all_chunks.pkl is up to date
```

`build_index.py` deletes and recreates the `StudyChunks` collection.

### 5. Ask questions

```bash
python ask.py
```

Type a question in the terminal; `exit` quits. Each answer is followed by the execution trace and the judge's evaluation.

## Hardware

Developed and run entirely locally on:

| | |
|---|---|
| Machine | MacBook Pro, Apple M4 Pro |
| Memory | 48 GB unified memory |
| Language model | `qwen3.5:35b` in Ollama (GPU via Metal) |
| Vector database | Weaviate in Docker Desktop |
| Embeddings | `multilingual-e5-base` via sentence-transformers |

Unified memory is shared by the language model (about 24 GB), its context cache, Docker and the embedding model, so 48 GB leaves comfortable headroom with a 32k-token context. No cloud API is used, so documents and questions never leave the machine.

Most of the latency comes from the agent loop, not from search: every tool round is a full pass of a 35B model over a growing context. Early versions made 15–19 tool calls per question. After switching to hybrid search, most questions need 1–3, which made answers several times faster. `ollama ps` shows whether the model runs fully on the GPU. If part of it falls back to the CPU, responses slow down sharply.

## Reproducing the Project

- **Documents.** The PDFs are public. Use the same versions: the study regulations as a consolidated text with amendments effective 1 October 2026 (Senate resolution 73/2026), and the Law on Higher Education and Science as the consolidated text (Dz. U. 2024 poz. 1571, as amended). File names must match the paths in `build_index.py` exactly, including Polish characters and parentheses. A different PDF version produces different chunks and different answers.
- **Order of steps.** `docker compose up -d` → `build_index.py` (or `restore_index.py` if `all_chunks.pkl` is present) → `ask.py`. With an empty index, every search returns nothing and the model declines to answer.
- **Working directory.** Scripts use relative paths: run them from the project root, where `source-documents/` is located.
- **Cleaning runs only during indexing.** `clean_regulamin.py` is called automatically inside `chunking()` for documents of type `regulamin`, so you never call it by hand. Its rules are tailored to this consolidated text. For a different regulation, check the resulting chunks (e.g. that paragraph labels are unique).
- **Model and memory.** `qwen3.5:35b` needs roughly 24 GB of memory for the model alone. Smaller models run on weaker hardware, but tool calling is usually less reliable. Use a recent Ollama version with tool-calling support for Qwen3.5.
- **First run** downloads `multilingual-e5-base` from Hugging Face (about 1 GB), so an internet connection is needed.
- **Ports** 8080 and 50051 must be free for Weaviate.
- **`all_chunks.pkl`** lets you restore the database in seconds, but it is valid only for the current PDFs and chunking code. After changing either, run `build_index.py`. Embeddings at indexing and query time must come from the same model. Only load pickle files you trust, because unpickling can execute code.
- **`build_index.py` deletes and recreates** the `StudyChunks` collection.
- **Results vary between runs.** The agent uses the model's default sampling (greedy decoding causes loops with Qwen in thinking mode). The judge runs at temperature 0 and is more stable.

## Limitations

- **The judge measures faithfulness, not retrieval recall.** It only sees the retrieved fragments, so it cannot tell when the relevant passage exists in the database but was not retrieved.
- **The same model generates and judges answers**, so they may share blind spots.
- **Sensitive to typos.** A heavily misspelled question ("ild mogd midc ects braku") led the model to search for the wrong concept.
- **Cleaning rules are document-specific.** `clean_regulamin.py` is tailored to the format of this consolidated text, and the act is only split, not cleaned.
- **Hardcoded configuration.** Document paths and metadata are defined in `build_index.py`, and model names in the code.
- **No automated test set yet**, so quality is assessed on individual questions.
- Answers are not legal advice; the official documents remain the authoritative source.

## Roadmap

Planned next steps:

- [ ] **Dynamic document loading without hardcoded paths**: scan `source-documents/` automatically and read metadata from a sidecar file (e.g. `documents.yaml`) or from the PDF itself, so adding a document requires no code changes.
- [ ] **Configuration file / `.env`** for model names, `num_ctx`, Weaviate host and paths (`python-dotenv` is already a dependency).
- [ ] **Incremental indexing**: hash each PDF and re-index only new or changed files.
- [ ] **Evaluation set with reference answers** (question, expected answer, expected paragraph) to measure retrieval recall and answer accuracy, not just faithfulness.
- [ ] **Separate refusal evaluation** in the judge, and computing deterministic fields (e.g. `grounded_in_sources` when no fragments were retrieved) in code instead of by the LLM.
- [ ] **Separate judge model** to reduce shared bias with the generator.
- [ ] **E5 `query:` / `passage:` prefixes**, which the embedding model was trained with.
- [ ] **Version-aware indexing**: store `valid_from` / `valid_to` for provisions instead of deleting old wording, to answer questions about a given date.
- [ ] **Cleaning for the act** (ISAP page headers and footers).
- [ ] Simple web interface (e.g. Streamlit).
