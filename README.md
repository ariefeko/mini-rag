# Mini-RAG: Document Q&A with LangChain

A minimal, educational Retrieval-Augmented Generation (RAG) pipeline built with
LangChain. Ask questions about a document in plain language and get answers
grounded in its actual content — not the LLM's general knowledge.

This project is intentionally small: one document, one in-memory vector
store, no server, no database. It exists to make the core RAG concepts
concrete before moving on to a production setup (persistent vector stores,
multiple documents, API-based retrieval, etc.).

---

## What this project demonstrates

RAG combines two ideas:

- **Retrieval** — given a question, find the most relevant pieces of a
  document instead of feeding the whole thing to the LLM.
- **Generation** — hand those relevant pieces to an LLM as context, and let
  it answer using that context rather than guessing from memory.

LangChain provides the building blocks (loaders, splitters, embeddings,
vector stores, chains) so you don't have to wire each step by hand.

---

## Tutorial flow

Here's exactly what happens when you run `mini_rag.py`, step by step:

```
┌─────────────────────────────────────────────────────────────────┐
│  1. LOAD                                                         │
│     kebijakan_cuti.txt is read from disk and wrapped into a      │
│     LangChain Document object.                                  │
└───────────────────────────┬───────────────────────────────────────┘
                            │
┌───────────────────────────▼───────────────────────────────────────┐
│  2. SPLIT                                                        │
│     The document is broken into small overlapping chunks         │
│     (~300 characters each) using RecursiveCharacterTextSplitter. │
│     Overlap keeps sentences from being cut off mid-thought.       │
└───────────────────────────┬───────────────────────────────────────┘
                            │
┌───────────────────────────▼───────────────────────────────────────┐
│  3. EMBED                                                        │
│     Each chunk is converted into a vector (a list of numbers      │
│     representing its meaning) using nomic-embed-text, running     │
│     locally through Ollama — no API cost, no internet needed      │
│     for this step.                                                │
└───────────────────────────┬───────────────────────────────────────┘
                            │
┌───────────────────────────▼───────────────────────────────────────┐
│  4. STORE                                                         │
│     All chunk vectors are stored in Chroma, an in-memory vector   │
│     database. It's rebuilt from scratch every time the script     │
│     runs — nothing is persisted to disk.                          │
└───────────────────────────┬───────────────────────────────────────┘
                            │
                    (you type a question)
                            │
┌───────────────────────────▼───────────────────────────────────────┐
│  5. RETRIEVE                                                      │
│     Your question is embedded the same way, then Chroma finds     │
│     the 2 chunks whose vectors are most similar to it.             │
└───────────────────────────┬───────────────────────────────────────┘
                            │
┌───────────────────────────▼───────────────────────────────────────┐
│  6. GENERATE                                                      │
│     The 2 retrieved chunks are inserted into a prompt template     │
│     as "context," and sent to Groq's Llama 3.3 70B model, which    │
│     answers based only on that context.                            │
└─────────────────────────────────────────────────────────────────┘
```

The whole thing is expressed in code as a single LCEL (LangChain Expression
Language) chain:

```python
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
```

Read it left to right: retrieve context → build the prompt → call the LLM →
parse the plain-text answer out of the response.

---

## Project structure

```
mini-rag/
├── mini_rag.py          # Main script — the pipeline described above
├── kebijakan_cuti.txt    # Example knowledge base (Indonesian leave policy doc)
├── requirements.txt      # Python dependencies
├── .env.example          # Template for your API key
└── .gitignore            # Keeps .env and local artifacts out of version control
```

---

## Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com) installed and running locally
- A free [Groq API key](https://console.groq.com)

---

## Setup

**1. Pull the local embedding model**

```bash
ollama pull nomic-embed-text
```

Make sure Ollama is running (`ollama serve`, or it may already be running as
a background service after installation).

**2. Create a virtual environment and install dependencies**

```bash
python3 -m venv venv
source venv/bin/activate   # on Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**3. Configure your Groq API key**

```bash
cp .env.example .env
```

Then open `.env` and set:

```
GROQ_API_KEY=your_actual_api_key_here
```

---

## Running it

```bash
python mini_rag.py
```

You'll see the pipeline build step by step (`[1/5]` through `[5/5]`), then a
prompt:

```
Pertanyaan:
```

Try asking (in Indonesian, since the example document is in Indonesian):

- `berapa hari cuti tahunan?`
- `apakah cuti sakit potong jatah cuti tahunan?`
- `berapa lama cuti melahirkan?`

Type `exit` to quit.

---

## Adding your own knowledge

Since the vector store is rebuilt on every run, adding knowledge is as
simple as editing the text file and re-running the script:

1. Open `kebijakan_cuti.txt` (or point the script at your own `.txt` file).
2. Add or edit content.
3. Save, then run `python mini_rag.py` again.

No re-indexing step, no cache to clear — everything is recomputed from
scratch each time, which is exactly why this setup only makes sense for
small, single-file examples like this one.

To load **multiple files**, adjust `build_rag_pipeline` to accept a list of
paths and loop over them, combining all resulting `Document` objects before
splitting.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `Connection refused` during the embedding step | Ollama isn't running | Run `ollama serve` in a separate terminal |
| `GROQ_API_KEY belum ditemukan` | `.env` missing or in the wrong folder | Confirm `.env` sits next to `mini_rag.py` and contains `GROQ_API_KEY=...` |
| `ModuleNotFoundError` | A dependency didn't install | Re-run `pip install -r requirements.txt`, or install the missing package individually |
| Answers ignore the document / hallucinate | Chunk size too large/small for the content, or `k` (retrieved chunks) too low | Try adjusting `chunk_size` in `RecursiveCharacterTextSplitter` or `k` in `search_kwargs` |

---

## Why this isn't production-ready (on purpose)

| | This project | A production RAG system |
|---|---|---|
| Vector store | Chroma, in-memory, rebuilt every run | Persistent (e.g. Qdrant), incremental updates |
| Knowledge source | 1 local text file | Many documents, possibly from a live codebase or CMS |
| Adding knowledge | Edit file, re-run script | Validated seed pipeline + idempotent loader script |
| Scale | Fine for a few KB of text | Needs chunk-level metadata, deduplication, access control |

This project is meant to build intuition for the RAG pattern itself. Once it
makes sense, the next step is applying the same five stages — load, split,
embed, store, retrieve+generate — to a system that needs to persist and
scale.