# AeroDocs

AeroDocs is a safety-conscious aviation engineering documentation assistant.
It is designed to support evidence retrieval, citation-grounded question
answering, synthetic defect workflows, and human-reviewed work-order drafting
using public FAA and EASA material.

Milestone 5 adds a secure, local stdio Model Context Protocol adapter to the
PDF, semantic-retrieval, and citation-grounded answer foundation. It exposes
only bounded document metadata, ranked evidence, and existing grounded answers
without adding write or maintenance-approval capabilities.

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
The default `sentence-transformers/all-MiniLM-L6-v2` embedding model is pinned
to immutable revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`.
Changing that revision makes existing indexes incompatible and requires a
rebuild; indexes with a legacy null revision are not silently upgraded. This
embedding/index identity is separate from the pinned Qwen generation-provider
identity.
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
does not perform reranking, lexical search, or work-order drafting.

## Grounded local answers

The initial development provider uses the Apache-2.0
[`Qwen/Qwen2.5-1.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)
model at immutable revision
`989aa7980e4cf806f80c7fef2b1adb7bc71aa306`. The model and tokenizer always use
that same revision. It remains replaceable, can be inaccurate, and is not an
aviation authority. The first answer request downloads it if it is not cached.
Health checks, ingestion, search, index metadata reads, and automated tests
neither load nor download it. Review the model card and license before
distribution.

After building an index, request a grounded answer:

```shell
curl -X POST http://127.0.0.1:8000/answers \
  -H "content-type: application/json" \
  -d '{"question":"How should corrosion be inspected?","top_k":5}'
```

The only optional retrieval controls are `top_k` and `document_ids`; the server
owns the evidence threshold. The response contains structured claims,
server-numbered citations, exact quotes, document title, filename, page range,
section, chunk identifier, retrieval score, model identity, index identity,
grounding results, and an audit identifier. A claim contains explicit support
links, so distinct quotations used by two claims remain visible even when they
share one provenance citation. The model may cite only opaque evidence IDs
supplied for that request. AeroDocs validates every identifier, exact quote,
polarity, modality, safety qualifier, comparator, numeric value, supported unit
conversion, and lexical-overlap proxy before rendering `[1]`-style citations.
A failure in any claim causes whole-answer abstention rather than a partial
answer.

Model output uses a strict discriminated JSON union: supported answer,
insufficient-evidence abstention, or conflict. Duplicate keys, omitted required
fields, fields from another variant, prose, Markdown fences, comments, multiple
values, unknown fields, and non-object roots are rejected without repair or a
second generation attempt.

Retrieved documents are untrusted data. High-confidence instruction-like text
causes the complete evidence item to be quarantined before generation, fixed
policy instructions are kept separate from canonical JSON evidence, and
model-provided provenance is never trusted. Screening covers bounded direct,
indirect, role-spoofing, schema-changing, citation-bypassing, Unicode-obfuscated,
and base64-like instruction patterns without treating ordinary maintenance
imperatives as prompt injection. Conservative checks also abstain on narrow,
scope-aligned evidence conflicts and requests to approve maintenance, determine
airworthiness, or bypass required procedures. These defenses reduce risk but
do not provide complete prompt-injection or general semantic-contradiction
detection.

Generation is bounded by question, evidence, context, exact chat-template
prompt tokens, model context window, generated tokens, claims, support links,
quotes, conflict identifiers, and a monotonic deadline. Configurable structured
limits may be lowered per deployment but cannot exceed the fixed parser
ceilings: 8 claims, 800 characters per claim, 8 supports per claim, 500
characters per quotation, and 8 conflict evidence IDs. Values are rejected,
never clamped or truncated. CPU inference is serialized per provider and
shares the process-global PyTorch thread setting documented above. The total
deadline begins before prompt preparation and includes waiting for the provider
lock. It is checked between decoding steps; Python cannot forcefully interrupt a native
forward pass already in progress, so a response can exceed the deadline by one
forward pass and repeated slow native calls can consume worker capacity.
Subprocess isolation is outside Milestone 4.

Audit records are written atomically under
`data/runtime/generation/audit/` by default. They contain hashes and selected
identifiers plus categorical success, abstention, retrieval, model, timeout,
resource-limit, parsing, and grounding outcomes—not questions, evidence text,
raw prompts, raw generated output, exception messages, stack traces, cache
locations, or filesystem paths. These local JSON records are audit-friendly
but are not tamper-evident and retain the usual trusted-local-filesystem TOCTOU
limitation. Required audit-write failure is fail-closed.

Run deterministic generation evaluation with manually inspectable JSONL cases:

```shell
uv run python -m app.generation.evaluation \
  --cases generation-evaluation-cases.jsonl \
  --output data/runtime/generation/evaluations/latest.json
```

Cases manually label answer, policy abstention, evidence abstention, or
conflict. The evaluator reports execution success, operational and malformed
failure rates, per-class abstention precision/recall/accuracy, conflict
precision/recall, valid-citation rate, claim citation coverage, evidence-ID
integrity, exact-quote validity, and lexical support/unsupported proxies.
Operational failures never count as successful abstentions. Lexical support can
produce false positives and false negatives; it is not entailment or an
aviation-validity judge. Evaluation uses no LLM, NLI, or embedding judge. Small
synthetic or public-document cases do not establish broad aviation validity.

Generated answers are evidence-navigation aids only. They do not approve
maintenance, establish airworthiness, replace controlling manufacturer
documentation or regulations, or replace qualified engineering judgment.

## Model Context Protocol

Milestone 5 exposes the existing document, retrieval, and grounded-answer
services through the official Python MCP SDK over local stdio:

```shell
uv run python -m app.mcp_server
```

The server opens no network listener. A generic MCP client can launch it with a
configuration equivalent to:

```json
{
  "mcpServers": {
    "aerodocs": {
      "command": "uv",
      "args": ["run", "python", "-m", "app.mcp_server"],
      "cwd": "/absolute/path/to/aerodocs"
    }
  }
}
```

The exact tool allowlist is:

- `list_documents` — bounded validated document summaries.
- `get_document_metadata` — safe public metadata for one document.
- `get_active_index` — validated active-index metadata without loading models.
- `search_evidence` — ranked exact passages and provenance, not an answer.
- `answer_from_documents` — the existing grounded answer or abstention.

The first four tools are read-only and idempotent. `answer_from_documents` is
non-destructive but is not annotated read-only or idempotent: each call reruns
generation when required and successful calls, abstentions, and feasible
failures may persist a distinct sanitized audit record. No MCP upload, index
rebuild, deletion, editing, sampling, shell, filesystem, administrative,
maintenance-approval, or airworthiness-certification tool exists.

Two static metadata resources are available at `aerodocs://status` and
`aerodocs://index`. The
`aerodocs://documents/{document_id}/metadata` resource template accepts only a
complete canonical opaque document ID. Resources never expose PDFs, extracted
text, manifests, runtime paths, or arbitrary file URIs.

Example tool arguments:

```json
{"query":"How should corrosion be inspected?","top_k":5}
```

```json
{
  "question": "How should corrosion be inspected?",
  "top_k": 5,
  "document_ids": ["doc_<sha256>"]
}
```

Search passages and ingested document content remain untrusted data. The answer
tool delegates to the same safety classification, evidence threshold,
injection quarantine, conflict handling, strict parsing, grounding, exact
quotation, server-controlled citation, timeout, and audit code used by the HTTP
API. An abstention remains an abstention; MCP does not make AeroDocs autonomous.

Stdio stdout is reserved for MCP protocol frames. AeroDocs and SDK logs,
warnings, and model diagnostics use stderr, and supported Hugging Face progress
displays are disabled. Model loading remains lazy; metadata-only calls load
neither pinned model. Search and answer may download their exact pinned
snapshots on first use if they are not cached.

MCP cancellation stops awaiting and prevents a late protocol response, but it
cannot forcibly interrupt a worker thread or native PyTorch forward pass
already in progress. The generation provider retains its total lock-wait and
inference deadline; subprocess inference and remote MCP transports are outside
this milestone.

MCP tool results are bounded by
`AERODOCS_MCP_MAXIMUM_TOOL_RESULT_CHARACTERS` and fail instead of truncating
evidence or provenance. The limit counts Unicode code points in the complete
`CallToolResult` JSON produced by the MCP SDK with aliases and omitted null
fields, including text content, structured content, error state, and escaping.
The minimum safe value is 162 characters so the bounded
`mcp_result_too_large` error always fits. Pagination uses the configured MCP
default and maximum.
AeroDocs remains an informational documentation assistant and cannot approve
maintenance, determine airworthiness, authorize return to service, or replace
approved documentation and qualified judgment.

## Evaluation benchmark

The tracked Milestone 6 bundle at `data/evaluation/milestone-6/` contains only
benchmark-authored synthetic source specifications, strict JSONL cases, and
deterministic thresholds. Source URLs in manifests are provenance metadata and
are never fetched. Generated PDFs, ingested documents, indexes, reports, model
caches, audits, and other evaluation runtime artifacts must remain under
ignored runtime roots and must not be committed.

Offline evaluation must use deterministic fake providers and the network-denial
guard. Real-provider evaluation is cache-only by default and can enable model
downloads only for AeroDocs' configured pinned MiniLM and Qwen identities.
Reports deliberately omit questions, prompts, answers, evidence text, provider
output, paths, host/user details, environment values, and secrets. Reproduction
requires the matching versioned dataset bundle.

Evaluation output is advisory engineering evidence. It does not approve
maintenance, airworthiness, or return to service, and optional human-review
ratings cannot change deterministic regression gates.

The canonical Phase 1 evaluation commands are:

```shell
uv run python -m app.evaluation validate --dataset milestone-6
uv run python -m app.evaluation run \
  --suite complete \
  --dataset milestone-6 \
  --mode offline \
  --report-id local-run
uv run python -m app.evaluation compare \
  --baseline local-run \
  --candidate candidate-run \
  --report-id local-comparison
```

Dataset, threshold, and report arguments are bounded identifiers, not paths.
Offline runs use isolated runtime roots, deterministic fake providers, and
process-local network denial. Real runs retain the exact production MiniLM and
Qwen model names and immutable revisions, and are cache-only unless
`--allow-model-downloads` is supplied.

When `--thresholds` is omitted, `run` automatically loads the selected
dataset's contained `thresholds_file`, if declared. An explicit
`--thresholds SAFE_THRESHOLD_ID` replaces that dataset default. Datasets with
no declared default run with mandatory hard invariants only. In every case,
the unrelaxable `unsafe_answer_count <= 0` invariant remains active for answer,
safety, and complete suites. Invalid, missing, unsafe, or incompatible
threshold configuration exits with code 2 before evaluation where detectable;
a valid policy whose gates fail still publishes complete reports and exits
with code 4.

Each threshold gate declares a non-empty closed `suites` scope. The complete
multi-suite threshold set is validated when loaded, then only gates explicitly
scoped to the selected suite are evaluated. Gates for other suites are recorded
as out of scope rather than passed, failed, missing, or ineligible. For example:

```json
{
  "population": "retrieval_quality",
  "metric_id": "recall_at_5",
  "operator": "greater_than_or_equal",
  "value": 0.8,
  "suites": ["retrieval", "complete"]
}
```

Retrieval and performance runs may therefore select zero optional gates from a
valid bundled policy without being rejected. Reports still identify the loaded
threshold set and record declared, selected, mandatory, and out-of-scope gate
counts.

The `performance` and `complete` suites share one performance-case selector and
warm-repetition plan. Each case contributes one quality observation; additional
repetitions are timing-only service executions and do not duplicate quality
denominators. Answer timing repetitions still use `AnswerService` normally and
therefore create separate evaluation-root audit records, while reports retain
only the primary case audit ID and never copy audit payloads.

Phase 1 accepts either bundled synthetic/redistributable sources, which are
processed only in isolated evaluation roots, or manifest-only sources, which
are validated read-only against the configured production corpus. A bundle
that mixes manifest-only and bundled sources fails closed with
`mixed_source_modes_unsupported`; isolated composite-corpus import is not
implemented in Phase 1.

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
