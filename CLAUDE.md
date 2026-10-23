# CLAUDE.md

## Project

**ai-hybrid-rag-agent** - a hybrid retrieval-augmented generation (RAG) agent over a document corpus.

The target pipeline: load documents (currently PDF) -> extract and chunk text -> index with **both** a dense (embedding/vector) index and a sparse (keyword/BM25) index -> fuse the two result sets -> retrieve -> generate an answer with an LLM that cites its sources.

**Current state: early.** PDF text extraction and structure-aware chunking work. The chunks are stored in PostgreSQL with **both** a dense index (pgvector) and a sparse index (Postgres full-text search), and the two are fused with reciprocal rank fusion. A FastAPI chat endpoint sits on top: it retrieves, calls an LLM (Groq) with the passages it retrieved, and streams the answer back as server-sent events. Conversation history is kept in PostgreSQL, so a follow-up question is answered with the earlier turns in view. Answers to the questions that are asked often are kept in a **distributed memory** (Redis, run in Docker), so a question that has already been answered comes straight back without reaching the model.

The code follows the four stages of the pipeline, one package each:

| Stage | Package | What it does |
|---|---|---|
| Indexing | `src/indexing/` | documents -> chunks -> embeddings, stored in both indexes |
| Retrieval | `src/retrieval/` | the dense and sparse indexes, fused with RRF |
| Augmentation | `src/augmentation/` | the model's context: the retrieved passages, the conversation, and a summary of the part too long to replay |
| Generation | `src/generation/` | the call to the LLM, and the citations the answer used |

`src/api/` is the transport only - the FastAPI wiring, the request/response shapes, and the CLI. It holds no business logic. See **Layout**.

`src/memory/` is the distributed memory. It is not a stage of the pipeline - it sits beside it, and the chat turn asks it before the pipeline runs.

Treat roadmap items as open questions to settle with the user, not as decisions already made.

## Strict Instructions

**No orphan methods** - Don't write any classes or methods that never being called from anywhere.
**Single Responsibility** - Maintain single responsibility principle when creating classes.
**No deprecated code** - Make sure the functions are not deprecated

## Commands

```bash
uv sync                                  # install/sync dependencies
docker compose up -d redis               # start the distributed memory (Redis)
docker compose down                      # stop it
uv run python src/indexing/pdf.py        # run the PDF extractor
uv run python src/indexing/chunker.py    # run the chunker (prints chunks as JSON)
uv run python src/indexing/run_index.py --force     # chunk -> embed -> store in PostgreSQL
uv run python src/retrieval/run_search.py "query" --mode hybrid   # hybrid (dense + keyword) search
uv run python src/retrieval/run_search.py "query" --mode dense     # dense only
uv run python src/retrieval/run_search.py "query" --mode sparse    # keyword only
uv run python src/api/run_api.py                  # serve the chat API (SSE)
uv run python src/api/run_api.py --reload         # serve it with auto-reload
uv add <package>                         # add a dependency (updates pyproject.toml + uv.lock)
uv add --dev <package>                   # add a dev dependency
uv pip uninstall <package>               # drop a dependency
uv run python -m pytest                  # run the test suite
```

- Use `uv add` / `uv pip uninstall` - never `pip install` directly. `uv.lock` is committed and must stay in sync with `pyproject.toml`.
- The console script `ai-hybrid-rag-agent` maps to `ai_hybrid_rag_agent:main`, which currently just prints a hello. Keep it working, or repoint it when a real CLI lands.
- The two CLIs (`run_index.py`, `run_search.py`) put `src/` on `sys.path` themselves so they work when run as a script from the repo root, and import their modules by full package path.
- Python version is pinned to **3.14** via `.python-version`.

## Layout
```
src/ai_hybrid_rag_agent/   # the installed package (CLI entry point)
src/indexing/              # document preparation - turns files into chunks
  pdf.py                   # raw PDF text extraction
  chunker.py               # structure-aware chunking -> metadata-carrying chunks
  indexer.py               # chunk -> embed -> write (the pipeline stage)
  run_index.py             # CLI: index the source PDF
src/retrieval/             # everything that searches the index
  config.py                # env-driven settings (DB connection, embedding model)
  embedder.py              # local Hugging Face sentence-transformer -> vectors
  database.py              # connection handling + schema/index lifecycle
  vector_store.py          # dense retrieval (pgvector, cosine distance)
  text_search.py           # sparse retrieval (Postgres tsvector/tsquery)
  retriever.py             # hybrid search, fuses both indexes with RRF
  faq_store.py             # the FAQ index: the questions, searchable by meaning
  index_state.py           # the version of the corpus, and the fingerprint of each document
  run_search.py            # CLI: query the index
src/augmentation/         # the model's context for one turn
  config.py              # env-driven settings (how much context to gather)
  history.py             # the conversation: the replay window, the search query
  context.py             # builds the messages: passages, summary, history, question
  session_store.py       # conversation history and summaries, in PostgreSQL
  augmenter.py           # retrieve -> choose the window -> summarize
  chat_service.py        # the turn: augment -> generate -> remember
src/generation/           # the call to the model, and the citations
  config.py              # env-driven settings (Groq key, model, temperature)
  llm.py                 # the LlmClient interface and the Groq client
  responder.py           # the prompt goes in, the text and citations come out
  citations.py           # resolves the model's [1] markers to the passages
  summarizer.py          # summarizes a run of conversation turns
src/memory/                # the distributed memory - answers asked often, in Redis
  config.py              # env-driven settings (where Redis is, how long it keeps an answer)
  cache.py               # the AnswerCache interface, the Redis cache, the cache key
  faq.py                 # what makes a question frequently asked, and the statistics behind it
src/api/                 # the transport only - no business logic
  config.py              # env-driven settings (where the API listens)
  schemas.py             # request/response shapes
  app.py                 # the FastAPI app and the POST /chat endpoint
  run_api.py             # CLI: serve the API with uvicorn
tests/                     # pytest suite
datasource/ # source documents, gitignored
docker-compose.yml         # the Redis server, run in Docker
.env                       # environment variables (gitignored - see gotcha 1)
.env.example               # placeholder copy of the env vars
pyproject.toml             # project metadata and dependencies
```

Seven packages under `src/`, split by the stage of the pipeline they belong to:

- **`src/indexing/`** prepares documents - it reads a file and produces chunks. It does not know how a chunk is searched.
- **`src/retrieval/`** owns the indexes and the search. It knows nothing about PDFs.
- **`src/augmentation/`** builds the model's context. It reads the indexes through `retrieval` and knows nothing about HTTP.
- **`src/generation/`** talks to the model and resolves the citations. It knows nothing about the index or about HTTP.
- **`src/memory/`** holds the distributed memory: the answers, the statistics of the questions, and the port the FAQ search answers. It knows nothing about the pipeline - it stores an answer and gives it back.
- **`src/api/`** is the transport. It reads the other packages and holds no business logic of its own.

The dependency runs one way only - `indexing` -> `retrieval` -> `augmentation` -> `generation` -> `api` (the indexer writes through `VectorStore`; the chunker knows nothing about vectors; the augmenter reads the indexes but never the chunker; the api layer only wires the others together). `retrieval` must never import from `indexing`, and `api` must never import from `indexing` either, or the packages become a cycle.

`memory` is a leaf beside the chain, not in it. It imports nothing from the project, and it may be imported: `augmentation` asks it, `api` builds it, and `retrieval` implements the one port it owns - the search for the question a new one means the same as. The port itself lives in `memory/faq.py`, next to the statistics it belongs with, so `retrieval` never has to import `augmentation` back.

The one place two stages meet is `augmentation` -> `generation`: the augmenter needs a model to summarize the conversation with, so it takes an `LlmClient`. That interface lives in `generation/llm.py`, so `generation` never has to import `augmentation` back.

All seven are real packages: they have `__init__.py`, they are listed in `[tool.uv.build-backend] module-name`, and they are installed. Import them by their full path - `from retrieval.database import Database`, `from indexing.chunker import PDFChunker` - never flat (`from database import Database`), which will not resolve outside the repo root.

## Chunking

`src/indexing/chunker.py` holds `PDFChunker`, which turns a PDF into `Chunk(chunk_text, metadata)` pairs sized for both BM25 and dense retrieval.

```python
from indexing.chunker import PDFChunker   # run as a script from the repo root
chunks = PDFChunker().chunk()              # falls back to $SOURCE_PDF
```

The approach is **heading-driven, not size-driven**: the document is a tree of headings, and each heading owns the body beneath it. The pipeline is extract text -> classify each physical line -> re-join the lines the PDF layout engine wrapped -> walk the blocks carrying project/role context forward -> split anything still over the embedding ceiling.

Metadata emitted per chunk: `section`, `source`, `page`, `end_page`, `section_index`, `chunk_index`, `section_chunk_index`, plus `project`, `client`, `period`, `role` and `label` where the document provides them. Example:

```json
{"chunk_text": "Designed and implemented GenAI-powered chat assistants ...",
 "metadata": {"section": "PROFESSIONAL EXPERIENCE", "source": "resume.pdf",
              "page": 2, "end_page": 2, "section_index": 5, "chunk_index": 1,
              "section_chunk_index": 0,
              "project": "TruStage", "client": "USA", "period": "Jul 2025- Till Date",
              "role": "Cloud Engineer with AI Capabilities", "label": "Responsibilities"}}
```

- `section` is the ALL-CAPS heading the chunk sits under.
- `page` / `end_page` are the pages the chunk's text came from. They differ only when a single block runs across a page break.
- `chunk_index` is unique across the whole document - it is the primary key of a stored chunk. `section_chunk_index` counts within the section.
- `project` / `client` / `period` come from project headers, in two shapes: `Project: TruStage, USA   Jul 2025- Till Date` and `District Marketplace: (Danske Bank, India)   Aug 2022 to Jan 2024`.
- `role` comes from `Role: ...` lines, and from a bare title line that follows a project header.
- `label` is the scoping label in force (`Responsibilities`, `Environment`). Only the labels in `_SCOPING_LABELS` scope content, so contact lines like `Email: ...` do not become labels - but they are still kept as chunk text.
- A scoping label always opens its own block, so it is never absorbed into the bullet above it (a bullet with no trailing period otherwise looks like an unfinished sentence).
- `section_rules` is the escape hatch for extra fields: a `dict[section_title, dict[field, compiled regex]]` whose group 1 becomes the metadata value.

The chunker is document-shape agnostic - the section, project, role and label patterns are module-level regexes at the top of `chunker.py` and are meant to be edited when a new document shape appears.

**Console encoding:** on Windows the default `cp1252` console cannot print the bullet characters this PDF contains. Run chunking scripts with `PYTHONIOENCODING=utf-8`, or write output to a UTF-8 file.

## Storage and retrieval

Both indexes live in **one table**, `chunks`, so the dense and the sparse side always describe exactly the same set of chunks. One row holds the chunk text, its metadata as `jsonb`, a `tsvector` for keyword search, and a `vector(384)` for similarity search.

| Class | File | Responsibility |
|---|---|---|
| `Database` | `retrieval/database.py` | connections, the schema, the extension, the indexes |
| `EmbeddingModel` | `retrieval/embedder.py` | text -> vectors, via a local sentence-transformer |
| `VectorStore` | `retrieval/vector_store.py` | dense read/write - pgvector, cosine distance |
| `TextSearchStore` | `retrieval/text_search.py` | sparse read/write - `tsvector` / `tsquery` |
| `HybridRetriever` | `retrieval/retriever.py` | runs both indexes, fuses with RRF |
| `Indexer` | `indexing/indexer.py` | the chunk -> embed -> write stage |

**Dense.** Vectors are L2-normalised before they are written, so cosine distance and inner product agree, and the HNSW index is built with `vector_cosine_ops`. Search ranks by `1 - (embedding <=> query)`, so the score is a similarity where higher is better. A search that filters puts the session into the iterative mode of the index, so the filter cannot throw the answer away - see gotcha 29.

**Sparse.** This is the standard Postgres full-text approach: a `tsvector` column computed by the server from the chunk text, a GIN index over it, and a `websearch_to_tsquery` built from the user's query. Ranking is `ts_rank_cd` (cover density), which rewards a term that appears in a tightly packed region rather than merely appearing often. Note that `websearch_to_tsquery` **ANDs** the terms - "kubernetes docker" only matches rows containing both. Use `OR` explicitly for a broader search.

**Fusion.** Reciprocal rank fusion, `RRF(d) = sum of 1 / (k + rank(d))` with `k = 60`. It reads only the *position* of a chunk in each list, never the score, which is what makes it work across two incomparable scales (cosine similarity vs. `ts_rank_cd`). A chunk that ranks well in both indexes beats one that ranks well in only one.

Both `VectorStore` and `TextSearchStore` accept a `filters` dict with `source`, `section`, `project` and `page`.

**Upserts.** A chunk is keyed on `(source, chunk_index)`, enforced by a *partial* unique index - PostgreSQL treats NULLs as distinct in a plain unique constraint, which would let duplicate rows in. Because the index is partial, `ON CONFLICT` must repeat its predicate:

```sql
ON CONFLICT (source, chunk_index)
WHERE source IS NOT NULL AND chunk_index IS NOT NULL
DO UPDATE SET ...
```

**The embedding column's dimension is fixed at creation time.** Changing `EMBEDDING_MODEL` means the column must be recreated - `ALTER TABLE ... ADD COLUMN IF NOT EXISTS` will not change an existing `vector(384)` into a `vector(768)`. Drop and re-create the table when the model changes.

## Augmentation, generation, the memory and the chat API

The last three stages of the pipeline are separate packages, and the API is the transport that wires them together.

### The augmentation step - `src/augmentation/`

This is what the model is given for one turn. `Augmenter.augment()` does three things in order:

1. **Retrieves.** The question is first made self-contained by `build_retrieval_query` (`augmentation/history.py`), which folds the earlier *user* questions into the search query with the current question last. The retrieval step does not see the conversation, and "what was my role in it?" means nothing to a keyword search on its own - without this a follow-up retrieves the wrong chunks even though the model has the context. The answers the assistant gave are deliberately left out of the query.
2. **Chooses the replay window.** `select_window` keeps the newest turns, up to a token budget. The budget is spent on *whole* turns - a question and the answer it produced - because replaying an answer without its question gives the model a reply to something it never saw. The newest turn is always kept, even when it does not fit.
3. **Summarizes the rest.** Whatever did not fit is summarized by the model (`generation/summarizer.py`) and the summary is put in the context in place of the original turns.

`build_messages` (`augmentation/context.py`) assembles the final message list: one system message holding the instructions, the numbered passages and the summary, then the replayed turns, then the question. The passages and the summary go into the *system* message so they are always in force, rather than being a user turn the model might treat as a question.

**A summary is not regenerated on every turn.** Once a summary has been written it is stored in `conversation_summaries`, against the id of the last message it covers, and only the turns *after* that id are considered for summarizing next time. Re-summarizing a summary would only lose detail.

### The generation step - `src/generation/`

`Responder.respond()` sends the prompt and streams the answer back token by token. When the answer is complete, the citation markers the model used - `[1]`, `[2]` - are resolved back to the passages they point at (`generation/citations.py`), so the response can say where each claim came from. A marker outside the range of the passages that were retrieved is **dropped**: the model must not be able to invent a source. The fullwidth brackets are understood too, because models often emit those.

`LlmClient` (`generation/llm.py`) is the interface every call to the model goes through, so the whole pipeline can be tested without a network call and without an API key. The summarizer uses the same interface, because summarizing a conversation is a generation task too.

### The distributed memory - `src/memory/`

The answers that are asked often are kept in Redis, which runs in Docker (`docker compose up -d redis`). The chat turn asks the memory **before** it retrieves or calls the model, so a question that has already been answered comes back straight away - with the same sources and the same citations it had the first time, and without a token spent on the model.

The memory is asked **twice, in two ways**.

**The exact question.** An answer is stored under a key built from the question and from the query the retrieval step would use (`build_cache_key`). The question is normalized, so a difference in case or spacing does not make a new key. The query carries the earlier questions of the conversation, so **a question that stands on its own always lands on the same key, and a follow-up lands on a key of its own** - a follow-up is answered from the passages and the history, and serving it a stale standalone answer would be wrong.

**The meaning of the question.** An exact key only catches a question that is worded the same way. So the questions that have been asked are also indexed by meaning, in a second table - `faq_questions` in PostgreSQL, with a `vector(384)` column and an HNSW index, using the same embedder as the chunks (`retrieval/faq_store.py`). A new question is embedded and the nearest stored question is looked up; if it is close enough, the answer it already has is served. The search is the same one the retriever runs, with one difference that matters: **the score is compared against a threshold, not ranked.** A rank says only that one question is nearer than another; a threshold says the two questions really are the same one, which is what has to be true before an answer is served without asking the model.

The threshold is `FAQ_SIMILARITY_THRESHOLD`, default `0.90`. Measured with the current embedder, against "What projects did Rose work on?":

| Question | Similarity | Matched? |
|---|---|---|
| `What projects did Rose work on?` | 1.00 | yes |
| `which projects was rose work on` | 0.96 | yes (and the exact key catches it too) |
| `Which projects was Rose involved in?` | 0.93 | yes |
| `What work projects has Rose been involved in?` | 0.92 | yes |
| `List the projects Rose worked on` | 0.78 | no - a rephrasing that falls below the line |
| `How many years of experience does Rose have?` | 0.69 | no - a different question about the same person |
| `What was my role in the TruStage project?` | 0.30 | no |
| `What is the capital of France?` | -0.02 | no |

The default is deliberately high. The failure modes are not symmetric: a false hit returns a **confidently wrong answer** to a factual question, which is much worse than a slow answer. Lowering the threshold to `0.85` catches more rewordings at the cost of the margin above the "different question" scores; raising it to `0.95` keeps only near-identical wordings.

**Only a question that stands on its own is looked up, and only a standalone question goes in the index.** A follow-up is answered from the conversation it was asked in, and a question in other words carries none of that context - so the answer to the stored question would not be the answer to this one. The test for "stands on its own" is that the retrieval query is exactly the question.

**What makes a question an FAQ.** Every question the memory is asked about gets a record: how many times it has been asked, how many times the memory served it, how much model time that saved, and when it was last asked (`memory/faq.py`). A question counts as a **frequently asked question** when all three of these hold:

- **asked often enough** - `FAQ_PROMOTE_ASKS`, default 3. One question asked twice is a coincidence, not a pattern.
- **served from the memory at least once** - `FAQ_PROMOTE_HITS`, default 1. The point of remembering a question is that the same question comes back; a question that never repeats was never an FAQ.
- **still fresh** - asked within `FAQ_FRESH_SECONDS`, default a day. A question nobody has asked in a week is not one anybody is asking now.

The rule decides two things. **How long the answer is kept**: a fresh answer for an hour (`CACHE_TTL_SECONDS`), an FAQ for a day (`FAQ_TTL_SECONDS`) - and the time to live is only ever extended, never cut short, so a question that has gone quiet keeps the time it had left. **What the memory reports**: `GET /memory/faqs` lists the questions that meet the rule, with their asks, hits, hit rate, and the model time they have saved. The registry keeps at most `FAQ_MAX_ENTRIES` questions, and the least asked ones are dropped first - the eviction is least-frequently-used, and it is about what the memory keeps *track of*; the answers themselves expire on their own time to live.

The counters live in Redis rather than in the process, so they are shared by every worker and survive a restart.

**The memory fails open.** A Redis that cannot be reached is a miss, not an error: the lookup returns nothing, the write is dropped, and the chat turn carries on as if the memory were off. The API logs a warning at startup when the ping fails. The same goes for the FAQ index: if the table cannot be built, the exact memory still works and the questions are simply not matched by meaning.

### The version of the corpus - `src/retrieval/index_state.py`

Every answer in the memory was built from a particular version of the corpus. When the resume is edited and indexed again, the answers built from the corpus as it was are wrong - and a wrong answer served from the memory is worse than a slow one, because it looks like an answer.

The version of the corpus is a counter, and it is part of every cache key: `build_cache_key(question, query, generation)` mixes it into the digest. After a rebuild the counter is different, so **every key the memory holds is a key nobody asks for again**. The memory misses, the model is asked, and the answer is stored under the new version. The old answers are not deleted - they expire on their own time to live.

The counter lives in one row of `index_state`, and the indexer moves it on (`IndexState.bump()`) as part of the write. **The write is one transaction**: the old rows of the document are removed, the new rows are written, the fingerprint of the document is recorded and the version moves on - or none of it happened. There is never a moment at which the rows of the corpus and the version that describes them disagree, so an answer can never be stored against a corpus that was never there.

**The version moves on last**, after the rows are written. An answer given while the write is in flight belongs to the corpus as it was, which is still the one the rows of the moment describe.

**A rebuild that changes nothing is skipped.** The fingerprint of every indexed document is kept in `indexed_documents` - a SHA-256 over the text and the metadata of all its chunks. A document whose fingerprint is unchanged is not written again, and the version does not move on: the memory keeps the answers it has, and nothing is paid for nothing. The name of the embedding model is part of the fingerprint, so a change of model writes the document again instead of skipping it - the same text under another model is another document as far as the index is concerned.

**The FAQ index is versioned too.** A question in the index points at the key its answer is stored under, and that key belongs to the version the answer was built from. So `faq_questions` carries a `generation` column, and the search only looks at the questions of the current version. A question asked against an older corpus is not a match, the model is asked, and the row is rewritten with the new version - the index heals itself.

**Reading the state fails open; writing it does not.** A corpus whose version cannot be read is reported as version `0`, and the memory works exactly as it did before - it just cannot tell that the corpus has changed. A version that could not be moved on, or a fingerprint that could not be recorded, is raised instead: a corpus is never written without the version that describes it, and a document is never written without the fingerprint that will stop it being written again for nothing.

### The chat API - `src/api/`

Transport only. It owns the FastAPI wiring, the request/response shapes and the CLI, and no business logic.

`POST /chat`:

```bash
curl -N -X POST http://127.0.0.1:8000/chat   -H "Content-Type: application/json"   -d '{"message": "What projects did Rose work on?", "conversation_id": "s1"}'
```

The request carries the user message and a `conversation_id`. The response is a `text/event-stream` with five kinds of event:

| Event | Carries |
|---|---|
| `sources` | The passages the answer will be based on. Sent first, so the UI can show them while the answer is still arriving. |
| `token` | One piece of the answer. |
| `citations` | The passages the answer actually cited, resolved from the markers the model used. |
| `done` | The end of the stream, with the full answer. |
| `error` | A failure. The stream ends after it. |

The `sources` and `citations` events carry two flags about where the answer came from:

| Flag | Value | Meaning |
|---|---|---|
| `cache_hit` | `true` / `false` | Whether the answer came from the memory or from the model. |
| `cache_match` | `null` / `"exact"` / `"semantic"` | *How* the memory recognised the question. `null` when the model answered. |

`"exact"` is the same question, matched on the key. `"semantic"` is the same question in different words, matched on meaning against the FAQ index. The distinction matters when a caller wants to know whether a new phrasing was recognised - a semantic hit is the feature working, and it is the one worth measuring.

`GET /memory/faqs` lists the questions that meet the FAQ rule, with the statistics behind each one:

```bash
curl http://127.0.0.1:8000/memory/faqs
```

```json
{"faqs": [{"question": "What projects did Rose work on?", "asks": 4, "hits": 2,
           "misses": 2, "hit_rate": 0.5, "saved_ms": 1180, "cost_ms": 590,
           "last_asked_at": 1761182400.0}]}
```

`GET /health` is a liveness check that does not touch the database, and `GET /docs` serves the interactive API docs.

**The conversation id is the only thing holding a session together.** The endpoint keeps no state in the process - the client sends the same `conversation_id` for every message in a conversation, and the model picks up where it left off. A different id starts a new conversation. The history lives in PostgreSQL, in the `conversations`, `messages` and `conversation_summaries` tables, so it survives a restart of the API and works unchanged behind more than one worker.

**The answer is grounded in the passages.** The retrieved chunks are numbered in the system message and the model is told to cite them with bracketed markers. The sources are stored alongside the answer, so a later turn can see what the earlier answers were built from.

Two things are deliberately kept off the event loop. The retriever is synchronous and does blocking work - a database round trip and an embedding forward pass - so it runs in a worker thread through `anyio.to_thread.run_sync`; running it inline would stall every other request. And the embedding model is loaded once in the FastAPI lifespan, so the first real request does not pay several seconds for the load.

## Tests

`tests/` holds a pytest suite that audits the chunker against the real resume. It uses the chunker's own `build_blocks()` output as the reference, so it cannot drift from the implementation.

- `tests/test_coverage.py` - no PDF text is lost (every word lands in a chunk text or a chunk's metadata), no word is invented, every text block produces a chunk, every heading / project header / role header is captured, every page produces chunks.
- `tests/test_pages.py` - label counts match the raw PDF, page numbers match the source block, a label never swallows the bullet above it, project metadata does not leak between projects.
- `tests/test_edge_cases.py` - missing PDF, empty `SOURCE_PDF`, determinism, the size ceiling, contact details surviving, and no bullet markers leaking into chunk text.
- `tests/test_storage.py` - the storage layer against a live PostgreSQL: upsert idempotency, dense ranking, filters, keyword stemming, matched terms, and the RRF score formula. **These tests skip themselves** when PostgreSQL is unreachable, or when the live `embedding` column is not `vector(384)`.
- `tests/test_api.py` - the chat endpoint against stand-ins for the retriever, the LLM, the session store and the memory: the SSE stream, the sources, the citations, the conversation memory, the follow-up query, and the cache hit that skips the model. **No database, no model, no API key and no Redis needed.**
- `tests/test_augmentation.py` - the augmentation and generation steps against stand-ins: the replay window, the retrieval query, the context building, the summarization and the citation resolution. **No database, no model and no API key needed.**
- `tests/test_memory.py` - the distributed memory against a stand-in for the Redis client: the cache key, the stored form of an answer, the time to live, and the behaviour when Redis is down. **No Redis server needed.**
- `tests/test_faq.py` - the FAQ rule and the semantic index: what makes a question frequently asked, the tiered time to live, the counters, the eviction, and the matching against a live PostgreSQL. **These tests skip themselves** when PostgreSQL is unreachable, or when the embedding model will not load.
- `tests/test_index_state.py` - the version of the corpus: the key it is part of, the fingerprint that skips a rebuild that changes nothing, and the state and the indexer against a live PostgreSQL, in a database of their own. **These tests skip themselves** when PostgreSQL is unreachable.
- `tests/test_ann_filter.py` - the filtered search over the approximate index: a search that filters on a column the HNSW index does not cover still returns the rows it was asked for, and the FAQ search still finds a question of the current version of the corpus. **These tests skip themselves** when PostgreSQL is unreachable.
- `tests/test_sessions.py` - the conversation history against a live PostgreSQL: message ordering, the history window, source storage, the summaries and session isolation. **These tests skip themselves** when PostgreSQL is unreachable.

Run with `uv run python -m pytest`. The chunker tests skip themselves if `datasource/` is empty, since `datasource/` is gitignored and a fresh clone has no source PDF.

## Environment variables

Read from `.env` via `python-dotenv` (`load_dotenv()` at module import in scripts).

| Variable | Purpose |
|---|---|
| `SOURCE_PDF` | Path to the PDF to index, relative to the repo root. Currently `datasource/Rose Gladson AG_.NETAzureOpsFullStack.pdf`. |
| `DATABASE_URL` | Full `postgresql://user:pass@host:port/db` URL. Overrides the `POSTGRES_*` variables. |
| `POSTGRES_HOST` / `POSTGRES_PORT` | Connection target. Default `127.0.0.1` / `5432`. |
| `POSTGRES_USER` / `POSTGRES_PASSWORD` | Credentials. Default user `postgres`. |
| `POSTGRES_DB` | Database name. Default `rag_agent`. Created on first run if absent. |
| `EMBEDDING_MODEL` | Hugging Face model id. Default `sentence-transformers/all-MiniLM-L6-v2` (384-dim). |
| `EMBEDDING_DEVICE` | `cpu` or `cuda`. Default `cpu`. |
| `EMBEDDING_BATCH_SIZE` | Chunks per forward pass. Default `32`. |
| `EMBEDDING_NORMALIZE` | L2-normalise the vectors. Default `true` - required for the cosine index to behave. |
| `EMBEDDING_MAX_SEQ_LENGTH` | Token limit. Default `512`. |
| `HF_TOKEN` | Hugging Face token, for gated or private models. |
| `GROQ_API_KEY` | The Groq API key. The chat API will not start without it. |
| `LLM_MODEL` | Which Groq model to answer with. Default `llama-3.3-70b-versatile`. |
| `LLM_TEMPERATURE` | Sampling temperature. Default `0.2`. |
| `LLM_MAX_TOKENS` | The cap on the length of an answer. Default `1024`. |
| `AUGMENT_TOP_K` | How many chunks the hybrid search returns per question. Default `5`. |
| `AUGMENT_HISTORY_TURNS` | How many of the most recent turns are replayed verbatim. Default `5`. |
| `AUGMENT_QUERY_HISTORY_MESSAGES` | How many earlier questions are folded into the search query. Default `2`. |
| `AUGMENT_MAX_HISTORY_TOKENS` | The token budget for the replayed history. Past it, the older turns are summarized. Default `2000`. |
| `AUGMENT_SUMMARY_MAX_TOKENS` | The cap on the length of the summary. Default `400`. |
| `API_HOST` / `API_PORT` | Where the chat API listens. Default `127.0.0.1` / `8000`. |
| `REDIS_HOST` / `REDIS_PORT` | Where the distributed memory is. Default `127.0.0.1` / `6379`. |
| `REDIS_PASSWORD` | The password for the memory. Default `ragredis`, the same default the Docker Compose file uses. |
| `REDIS_DB` | The Redis database number. Default `0`. |
| `REDIS_URL` | The full connection URL. Overrides the four variables above. |
| `CACHE_TTL_SECONDS` | How long an answer is kept in the memory. Default `3600` - one hour. |
| `FAQ_ENABLED` | Whether the FAQ layer is on at all. Default `true`. When it is off, the exact memory still works and the questions are simply not matched by meaning. |
| `FAQ_SIMILARITY_THRESHOLD` | The cosine similarity two questions have to reach before they are treated as the same question. Default `0.90` - deliberately high, because a false hit returns a confidently wrong answer. |
| `FAQ_PROMOTE_ASKS` | How many times a question has to be asked before it counts as frequently asked. Default `3`. |
| `FAQ_PROMOTE_HITS` | How many times the memory has to have served it. Default `1`. |
| `FAQ_FRESH_SECONDS` | How long a question stays fresh. Default `86400` - a day. |
| `FAQ_TTL_SECONDS` | How long the answer to a frequently asked question is kept. Default `86400` - a day. |
| `FAQ_MAX_ENTRIES` | How many questions the registry keeps track of. Default `500`. The least asked are dropped first. |

The free Hugging Face **Hub download** of a model does not need credits. The hosted **inference API** does - it returns HTTP 402 without a paid plan - which is why the model runs locally here.

## Conventions
- **uv** for all dependency and environment management.
- Use `uv run` for every command.
- **Type hints** on public function signatures (see `main() -> None`).
- **`load_dotenv()`** at the top of any module that reads config from the environment.
- **Secrets stay out of git** - see gotcha 1 before adding anything to `.env`.
- Prefer pathlib over `os.path` for new path handling, and `os.getenv` for config that must come from the environment.
- Keep indexing stages independently runnable: each stage should read its input and write its output, so a stage can be re-run without redoing the ones before it.

## Gotchas

1. **`.env` is now untracked.** It was tracked by git (`git ls-files` listed it), which meant anything added to it got committed and pushed to the public remote. It has been added to `.gitignore` and untracked (`git rm --cached .env`); `.env.example` holds the placeholder values.
2. **`src/indexing/pdf.py:11` prints the wrong thing** - `f"Number of pages: {reader.pages}"` prints the list repr, not a count. It should be `len(reader.pages)`.
3. **`src/indexing/pdf.py:17` has an import-time side effect** - `print(extract_text_from_pdf())` runs on import. Guard with `if __name__ == "__main__":` so importing the module does not trigger extraction.
4. **`page.extract_text()` can return `None`** for pages with no extractable text, which makes `text += page.extract_text()` raise `TypeError`. Use `page.extract_text() or ""`.
5. **The packages must stay installed and absolute.** All seven under `src/` have an `__init__.py` and are all listed in `[tool.uv.build-backend] module-name`, so `uv build` ships them. Import them by full path (`from retrieval.database import Database`). A flat import (`from database import Database`) only resolves if the module's own directory happens to be on `sys.path` - it breaks the moment the code is installed or imported from anywhere else.
6. **`README.md` is empty.** It is the `readme` referenced by `pyproject.toml` and needs real content.
7. **`datasource/` is gitignored** - a fresh clone has no documents. Anyone else working on this repo must supply their own files, and `SOURCE_PDF` must be updated accordingly.
8. **The PDF layout engine corrupts text.** `pypdf` hard-wraps body text mid-sentence and injects stray spaces inside words (`Aj ax`, `larg est`, `implementa tion`, `Ju n 20 19`). The chunker re-joins wrapped lines, but the intra-word spaces are still in the extracted text and will land in chunks and embeddings. Fix at the source if the noise hurts retrieval - a custom `extraction_mode` on `PdfReader` may extract cleaner text.
9. **pgvector is installed from a third-party build.** The upstream project ships no Windows binaries; the extension here came from `andreiramani/pgvector_pgsql_windows` (tag `0.8.6_18`, for PG18 x64) - not maintained by the pgvector project or EDB. Rebuild from source (`nmake /F Makefile.win`) if that matters. Installing it means copying `lib/vector.dll`, `share/extension/vector*` and the headers into `C:\Program Files\PostgreSQL\18` **and restarting the `postgresql-x64-18` service** - the server only picks the DLL up on restart.
10. **psycopg has no adapter for the `vector` type.** Vectors are written as the string literal `[1,2,3]` and cast in SQL with `::vector`. Passing a Python list raises `TypeError: expected bytes, tuple found`.
11. **PostgreSQL will not accept a bind parameter as a type modifier.** `vector(%s)` is a syntax error - "type modifiers must be simple constants or identifiers". The dimension has to be interpolated into the DDL string.
12. **`websearch_to_tsquery` ANDs its terms.** "kubernetes docker" matches only rows containing *both*, which looks like a broken keyword search when a corpus has them in different chunks. Use `OR` for a broader query.
13. **`tsvector` stores stemmed lexemes.** "kubernetes" is stored and returned as `kubernet`. Assert on the stem, not the surface form.
14. **The API will not start without `GROQ_API_KEY`.** `GroqLlmClient` checks the key when it is built, not on the first request, so a misconfigured deployment fails loudly at startup instead of on the first question.
15. **A conversation id is the only thing that keeps a session together.** The endpoint holds no state in memory - the client must send the same `conversation_id` for every message in a conversation, or the model starts each question from scratch.
16. **The project name only appears in the chunk metadata, never in the chunk text.** "TruStage" is a `metadata->>'project'` value; the text of that chunk says "Trustage applications" or nothing at all. So a question naming the project does not match it in the keyword index, and the dense index has nothing to match on either. This is why a follow-up like "what was my role in the TruStage project?" can come back empty even though the chunk is right there. The fix is to search the metadata column too, or to put the project name into the embedded text.
17. **A backslash escape in a heredoc payload can collapse into the character itself.**
When writing a file through `uv run python - << 'PYEOF'`, an escaped sequence in the payload
can end up as the character it stands for - an escaped NUL became a NUL byte in the file, and an
escaped newline became a real newline and broke the string literal. Write the file with the Write
tool where possible, or build the escape in the payload with `chr(92)`.
18. **The `api` package is transport only.** Business logic that used to live there (the prompt building, the session store, the chat service, the LLM client) now lives in `augmentation/` and `generation/`. If a rule starts appearing in `src/api/`, it belongs in one of those two.
19. **The FAQ match is a threshold, not a rank.** The chunks are fused with RRF, but the FAQ index is not - a rank says only that one question is nearer than another, and that is not enough to serve an answer without asking the model. A similarity of `0.90` says the two questions really are the same one. The two mechanisms do different jobs and must not be confused.
20. **Only a question that stands on its own is looked up, and only a standalone question goes into the index.** A follow-up is answered from the conversation it was asked in, so a question in other words carries none of that context - the answer to the stored question would not be the answer to this one. The test for "stands on its own" is that the retrieval query is exactly the question.
21. **The FAQ index has its own `vector(384)` column, and it is fixed at creation time too.** `faq_questions.embedding` is a separate column from `chunks.embedding`; changing `EMBEDDING_MODEL` means recreating both. The table is created by the API at startup (`ensure_schema`), and a failure there is logged, not raised - the exact memory still works without it.
22. **The time to live is only ever extended.** The refresh uses `expire(key, ttl, gt=True)`, so a question that has stopped being asked keeps the time it had left instead of having it cut short. Cutting it short would mean a question asked once an hour could never be a FAQ at all.
23. **`memory` is a leaf, and `retrieval` implements the one port it owns.** `memory` imports nothing from the project. `augmentation` and `api` may import it; `retrieval` may import it for the FAQ port, and the port lives in `memory/faq.py` next to the statistics it belongs with, so `retrieval` never has to import `augmentation` back.
24. **The memory fails open.** A Redis that is down, or a network that is down, is a miss - never an
error. The lookup returns nothing, the write is dropped, and the chat turn carries on as if the memory
were off. The API logs a warning at startup when the ping fails.
25. **The cache key carries the retrieval query, not just the question.** A question that stands on its
own always lands on the same key, so it is answered from the memory. A follow-up lands on a key of its
own, because the earlier questions are folded into the query it is answered under. Serving a follow-up
a stale standalone answer would be wrong.
26. **The cache key carries the version of the corpus.** The key is `sha256(question | query | generation)`,
so a corpus that has been rebuilt invalidates the memory without anything being deleted: the old answers
are still in Redis, but they are under keys nobody asks for again. An answer from the corpus as it was
is never served - which is the point, because a confidently wrong answer is worse than a slow one.
27. **The FAQ index has to be versioned as well.** A question in the index points at the key its answer
is stored under, so a row of an older version points at an answer of an older corpus. `faq_questions`
carries a `generation` column and the search only looks at the current version - otherwise the semantic
match would hand back exactly the answer the version was meant to invalidate.
28. **A rebuild that changes nothing is skipped.** The fingerprint of a document is a SHA-256 over the
text and the metadata of all its chunks, kept in `indexed_documents`. An unchanged document is not
written again and the version does not move on, so the memory keeps the answers it has. Without this,
every re-index would empty the memory for nothing.

29. **A filter is applied after the approximate scan, not before it.** pgvector's HNSW scan visits a bounded
number of tuples and the WHERE filter throws away the ones that do not match, so a search that filters on a
column the index does not cover can come back with fewer rows than it was asked for - including none at all.
The search session is therefore put into the iterative mode (`SET hnsw.iterative_scan = strict_order`), which
keeps the scan going until there are enough rows that match, and keeps them in the order the distance puts
them in. `VectorStore.search` and `PostgresFaqMatcher.match` are the two searches that filter, and both
call `prepare_ann_search` before the query. A filter another index can answer - the source, for one -
hides the bug entirely: the planner never reaches for the HNSW index at all, and the answer is exact.
30. **The corpus is written in one transaction, and the version moves on last.** The rows, the fingerprint
and the version are written together, or not at all - so there is never a moment at which the rows of the
corpus and the version that describes them disagree. A write that cannot move the version on rolls the
rows back with it, and nothing of the corpus is left behind.

## Open decisions

Settle these with the user before building on them - each one shapes the indexing and retrieval code:
- **Embedding model**: `all-MiniLM-L6-v2` (384-dim, fast) is the current default. A stronger model (`bge-small-en-v1.5`, `all-mpnet-base-v2`) would retrieve better but is slower. Changing it means recreating the `embedding` column - the dimension is fixed at table-creation time.
- **Reranking**: whether a cross-encoder reranker runs over the fused candidates, and which one.
- **Agent framing**: what "agentic" means here - a single retrieve-then-generate turn, or an agent that can issue multiple retrieval rounds and self-correct.
- **Evaluation**: the question set and metrics used to judge retrieval quality. Without this there is no signal that hybrid beats either index alone.

Settled: the vector store is **pgvector**, the sparse index is **Postgres full-text search**, the fusion strategy is **reciprocal rank fusion**, the LLM provider is **Groq**, and the conversation history is kept in **PostgreSQL**.
