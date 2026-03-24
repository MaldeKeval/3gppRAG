# Local 3GPP NR RAG (CLI)

Retrieval-augmented QA over the Hugging Face dataset [raoulbia/3gpp-5g-nr-qa](https://huggingface.co/datasets/raoulbia/3gpp-5g-nr-qa) using **Ollama** for embeddings and chat, and **Chroma** for local vector storage.

## Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com/) installed and running (default URL `http://127.0.0.1:11434`).

### Pull models

Embedding and chat models are separate. Pull both before indexing or chatting:

```bash
ollama pull nomic-embed-text
ollama pull qwen2.5:7b
```

To use a different chat model (for example if you use a Qwen 3.5 build under another tag), set `RAG_CHAT_MODEL` after pulling that model:

```bash
# Windows PowerShell
$env:RAG_CHAT_MODEL = "your-model:tag"
```

Defaults are defined in [`rag/config.py`](rag/config.py) and can be overridden with environment variables (see below).

## Setup

```bash
cd path/to/RAG
python -m pip install -r requirements.txt
```

## Usage

**Build the index** (downloads the dataset from Hugging Face and embeds each row via Ollama). First run can take a long time for the full ~27k training rows.

```bash
python -m rag --rebuild-index
```

Smoke test with a small slice:

```bash
python -m rag --rebuild-index --limit 100
```

**Interactive REPL** (requires an existing index under `data/chroma/` unless you just rebuilt with a question):

```bash
python -m rag
```

**Single question**:

```bash
python -m rag -q "What are the default values for timers t310 and t311?"
```

**Rebuild and exit** (no REPL):

```bash
python -m rag --rebuild-index
```

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `RAG_OLLAMA_HOST` | `http://127.0.0.1:11434` | Ollama API base URL |
| `RAG_EMBED_MODEL` | `nomic-embed-text` | Must match the model used to build the index |
| `RAG_CHAT_MODEL` | `qwen2.5:7b` | Generation model |
| `RAG_CHROMA_PATH` | `<project>/data/chroma` | Persistent Chroma directory |
| `RAG_COLLECTION_NAME` | `3gpp_5g_nr_qa` | Chroma collection name |
| `RAG_TOP_K` | `5` | Retrieved chunks per query |
| `RAG_MAX_CONTEXT_CHARS` | `12000` | Max size of concatenated context |
| `RAG_DATASET` | `raoulbia/3gpp-5g-nr-qa` | Hugging Face dataset id |
| `RAG_DATASET_SPLIT` | `train` | Dataset split to index |

If you change `RAG_EMBED_MODEL`, rebuild the index with `--rebuild-index`.

## License and data

The dataset is released under **CC-BY-NC-4.0** (non-commercial). See the [dataset card](https://huggingface.co/datasets/raoulbia/3gpp-5g-nr-qa). If you use it in research or redistributed work, cite the dataset and [TSpec-LLM](https://arxiv.org/abs/2406.01768) as described there.

Answers are LLM-generated and retrieved from indexed text; verify critical details against official 3GPP specifications.

## Disclamer

This project is vibe coded run at your own risk, I am not liable