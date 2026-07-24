# AeroDocs

AeroDocs is a safety-conscious aviation engineering documentation assistant.
It is designed to support evidence retrieval, citation-grounded question
answering, synthetic defect workflows, and human-reviewed work-order drafting
using public FAA and EASA material.

Milestone 3 adds fully local semantic indexing, ranked evidence retrieval, and
inspectable retrieval evaluation to the secure PDF foundation. Search returns
source passages, not generated answers, maintenance decisions, or approval of
maintenance or airworthiness.

## Requirements

- Python 3.11
- [uv](https://docs.astral.sh/uv/)

## Local development

Install the locked dependencies:

```shell
uv sync
```

Copy `.env.example` to `.env` if local configuration overrides are needed.
Settings use the `AERODOCS_` environment-variable prefix. Never place secrets in
either file.

Start the API:

```shell
uv run uvicorn app.main:app --reload
```

Verify the service:

```shell
curl http://127.0.0.1:8000/health
```

The default response is:

```json
{"name":"AeroDocs","version":"0.1.0","status":"healthy"}
```

## PDF ingestion

Only public or otherwise authorized PDF documents may be ingested. Upload a
document with:

```shell
curl -X POST http://127.0.0.1:8000/documents \
  -H "accept: application/json" \
  -F "file=@public-manual.pdf;type=application/pdf"
```

The response contains a deterministic document identifier and normalized
metadata. Use that identifier to retrieve stored representations:

```shell
curl http://127.0.0.1:8000/documents/doc_<sha256>
curl "http://127.0.0.1:8000/documents/doc_<sha256>/pages?offset=0&limit=100"
curl "http://127.0.0.1:8000/documents/doc_<sha256>/chunks?offset=0&limit=100"
```

Accepted PDFs are stored under the configured
`AERODOCS_DOCUMENT_STORAGE_ROOT` (default `data/runtime`) using atomic
publication. Each completed document contains the source PDF, metadata, pages,
chunks, and a versioned integrity manifest with artifact hashes and sizes.
Reads and duplicate checks validate that manifest. Runtime files remain local
and are ignored by Git.

Uploads must be unencrypted, use `application/pdf`, begin with a PDF signature,
and contain at least one page. The PDF-file limit defaults to 25 MiB; a separate
27 MiB request-body limit allows multipart overhead and is enforced before
multipart parsing. Extraction defaults to at most 1,000 pages, 500,000
characters per page, and 10,000,000 characters total. Limits are configurable
with the `AERODOCS_` settings shown in `.env.example`; content is rejected, not
truncated, when a limit is exceeded.

Duplicate source bytes with the same processing profile return the existing
document. A profile consists of the processing, extraction, and chunking format
versions plus chunk size and overlap. The same bytes with a different profile
return HTTP 409 and identify the stored profile. Automatic reprocessing and
version migration are intentionally unsupported in Milestone 2.

OCR is not implemented; scanned or image-only pages can have empty text and a
warning. PDF metadata and detected headings are best-effort context, not
authoritative engineering data. Local no-follow and containment checks reduce
symlink risks, but filesystem checks and operations retain an unavoidable local
TOCTOU window. The runtime root must therefore be writable only by the AeroDocs
process or a trusted operator.

## Local semantic retrieval

The first index build loads the configured Sentence Transformer model. If it is
not cached, the library downloads it from its model host; pin
`AERODOCS_EMBEDDING_MODEL_REVISION` for reproducible deployments. Automated
tests inject a fake provider and never download a model, use a GPU, or access
the network.

PyTorch CPU thread pools are process-global. AeroDocs configures them lazily
before constructing the model and initializes every inference worker thread.
`AERODOCS_EMBEDDING_TORCH_NUM_THREADS` defaults to `1`, which prioritizes
stability for local macOS/Uvicorn worker-thread inference over maximum
throughput. All application instances in one process must use the same value;
restart the process to change it. Inference is serialized per embedding
provider. A native segmentation fault cannot be recovered by Python's typed
error handling, so this initialization is preventative rather than a recovery
mechanism.

Run this manual macOS AnyIO worker-thread smoke test after installing or caching
the configured model (the first run may download it):

```shell
uv run python - <<'PY'
import anyio

from app.core.config import Settings
from app.retrieval.embeddings import SentenceTransformerEmbeddingProvider

provider = SentenceTransformerEmbeddingProvider(Settings())

async def main() -> None:
    vector = await anyio.to_thread.run_sync(
        provider.embed_query,
        "corrosion inspection",
    )
    print(vector.shape)

anyio.run(main)
PY
```

Build or refresh the global index:

```shell
curl -X POST http://127.0.0.1:8000/indexes/rebuild
```

An unchanged, integrity-valid active corpus returns HTTP 200 with
`rebuilt=false`. If the same logical profile already has a valid inactive
physical version, AeroDocs fully validates and reactivates the
lexicographically smallest physical version ID without embedding again; the
response returns its original stored metadata and HTTP 200. Initial, changed,
explicitly reconfigured, or corrupt-version recovery builds publish a new
immutable physical version and return HTTP 201. Logical profile and corpus IDs
remain deterministic, while physical version IDs are unique so a corrupt
version is never overwritten. Search and metadata reads reject a corrupt active
index; only an explicit rebuild attempts recovery from independently validated
source documents or a separately validated compatible version. Inspect the
active version:

```shell
curl http://127.0.0.1:8000/indexes/active
```

Search all indexed public or authorized documents:

```shell
curl -X POST http://127.0.0.1:8000/search \
  -H "content-type: application/json" \
  -d '{"query":"How should corrosion be inspected?","top_k":5}'
```

Filter without rebuilding by passing `"document_ids":["doc_<sha256>"]`. With
the default normalized `IndexFlatIP`, scores are cosine similarities from -1 to
1 and higher is more similar. If normalization is disabled, scores are
`1 / (1 + squared L2 distance)` from 0 to 1. Equal scores use chunk ID as the
deterministic tie-break.

Index builds are bounded by configured chunk, character, and batch limits.
FAISS flat indexes retain every vector in memory, while the Transformer model
and one embedding batch require additional memory. The selected model may
truncate chunks beyond its token limit; index metadata records that limit.
Artifacts use hashes, manifests, immutable version directories, an atomic
active pointer, and an OS-managed build lock. Runtime index files remain under
`data/runtime/` and are ignored by Git.

Run a local, non-LLM evaluation:

```shell
uv run python -m app.evaluation.runner \
  --cases evaluation-cases.jsonl \
  --output data/runtime/indexes/evaluations/latest.json
```

Cases identify relevant chunk IDs or document/page ranges. Per-case Recall@k is
the fraction of distinct declared chunk and canonical document/page targets
matched within the first k results; duplicate declarations and duplicate
results do not increase the numerator or denominator. Aggregate Recall@k is the
arithmetic mean of those per-case fractions. Page targets use interval overlap,
and MRR uses the first result matching any target. Search failures receive zero
recall and reciprocal rank with an explicit failure category. Reports also
contain no-result rate, observational latency, and per-case rankings. Small
evaluations demonstrate mechanics only; they do not establish broad aviation
validity.

Sentence Transformers and the default MiniLM model use Apache-2.0 licensing;
FAISS uses MIT licensing. Review the selected model card, license, limitations,
and architecture-specific FAISS compatibility before distribution or
deployment. FAISS files should be rebuilt when moving between architectures.

AeroDocs semantic search does not replace approved manufacturer maintenance
documentation, applicable regulations, or qualified engineering judgment. It
does not yet perform LLM answer generation, RAG, reranking, lexical search, or
work-order drafting.

## Verification

```shell
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run mypy .
```

Only public, synthetic, or explicitly authorized data may be used. Do not
commit `.env` files, credentials, private manuals, runtime documents, or
generated data. AeroDocs does not replace approved manufacturer maintenance
documentation or qualified engineering judgment.

## Document data

Intentionally tracked public FAA and EASA documents may eventually be placed
under `data/public/`. Private, temporary, or generated material must be placed
under `data/private/`, `data/temporary/`, or `data/generated/`; these locations
are ignored by Git. Private or proprietary aviation manuals must never be
committed.
