const ABSOLUTE_RESPONSE_BYTES = 512 * 1024 * 1024;

const ABSTENTION_REASONS = new Set([
  "insufficient_evidence",
  "prompt_injection_risk",
  "conflicting_evidence",
  "grounding_validation_failed",
  "safety_boundary",
]);

const ERROR_MESSAGES = Object.freeze({
  invalid_search_request: "The search request is invalid.",
  invalid_answer_request: "The answer request is invalid.",
  answer_resource_limit: "The answer request exceeds configured limits.",
  index_not_ready: "No active evidence index is available.",
  filtered_document_not_found:
    "A selected document is not present in the active index.",
  document_not_found: "The selected document was not found.",
  stale_index: "The document corpus changed; the index is stale.",
  incompatible_index_profile:
    "The active index is incompatible with the current configuration.",
  embedding_dimension_error:
    "The active index is incompatible with the embedding configuration.",
  index_integrity_error: "The evidence index failed integrity validation.",
  corrupt_source_document: "A source document failed integrity validation.",
  document_storage_error: "Document metadata could not be read safely.",
  embedding_shape_error: "The embedding provider returned an invalid result.",
  non_finite_embedding: "The embedding provider returned an invalid result.",
  embedding_model_unavailable:
    "The configured local embedding model is unavailable.",
  llm_model_unavailable:
    "The configured local generation model is unavailable.",
  generation_output_invalid:
    "The local model returned malformed structured output.",
  generation_failed: "The local model could not generate an answer.",
  generation_timeout: "The generation request exceeded its time limit.",
  audit_persistence_failed:
    "The answer audit could not be stored; no answer is presented.",
});

export class ResponseValidationError extends Error {
  constructor(code = "invalid_response") {
    super(code);
    this.name = "ResponseValidationError";
  }
}

function invalid(code) {
  throw new ResponseValidationError(code);
}

function record(value) {
  if (value === null || typeof value !== "object" || Array.isArray(value)) {
    invalid();
  }
  return value;
}

function exactKeys(value, keys) {
  const actual = Object.keys(record(value)).sort();
  const expected = [...keys].sort();
  if (
    actual.length !== expected.length ||
    actual.some((key, index) => key !== expected[index])
  ) {
    invalid();
  }
}

function stringValue(value, maximum = 4096, nullable = false) {
  if (nullable && value === null) {
    return null;
  }
  if (
    typeof value !== "string" ||
    value.length > maximum ||
    /[\u0000-\u0008\u000b\u000c\u000e-\u001f\u007f]/u.test(value)
  ) {
    invalid();
  }
  return value;
}

function nonempty(value, maximum = 4096) {
  const parsed = stringValue(value, maximum);
  if (parsed.length === 0) {
    invalid();
  }
  return parsed;
}

function integer(value, minimum = 0, maximum = Number.MAX_SAFE_INTEGER) {
  if (!Number.isSafeInteger(value) || value < minimum || value > maximum) {
    invalid();
  }
  return value;
}

function finite(value) {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    invalid();
  }
  return value;
}

function booleanValue(value) {
  if (typeof value !== "boolean") {
    invalid();
  }
  return value;
}

function nullableString(value, maximum = 4096) {
  return stringValue(value, maximum, true);
}

function boundedArray(value, maximum) {
  if (!Array.isArray(value) || value.length > maximum) {
    invalid();
  }
  return value;
}

function unique(values) {
  if (new Set(values).size !== values.length) {
    invalid();
  }
}

export async function readBoundedJson(response, maximumBytes) {
  const contentType = response.headers.get("content-type");
  if (contentType === null || !approvedJsonMediaType(contentType)) {
    invalid("unexpected_response_media_type");
  }
  const declared = response.headers.get("content-length");
  if (declared !== null) {
    const size = Number(declared);
    if (!Number.isSafeInteger(size) || size < 0 || size > maximumBytes) {
      invalid();
    }
  }
  if (response.body === null) {
    invalid();
  }
  const reader = response.body.getReader();
  const decoder = new TextDecoder("utf-8", { fatal: true });
  let received = 0;
  let text = "";
  for (;;) {
    const part = await reader.read();
    if (part.done) {
      break;
    }
    received += part.value.byteLength;
    if (received > maximumBytes) {
      await reader.cancel();
      invalid();
    }
    text += decoder.decode(part.value, { stream: true });
  }
  text += decoder.decode();
  if (text.length === 0) {
    invalid();
  }
  try {
    return JSON.parse(text);
  } catch {
    invalid();
  }
}

function approvedJsonMediaType(value) {
  const parts = value.split(";");
  if (parts.shift()?.trim().toLowerCase() !== "application/json") {
    return false;
  }
  return parts.every((parameter) =>
    /^\s*[!#$%&'*+\-.^_`|~0-9A-Za-z]+\s*=\s*(?:"[^"\r\n]*"|[!#$%&'*+\-.^_`|~0-9A-Za-z]+)\s*$/u.test(
      parameter,
    ),
  );
}

function positiveConfigInteger(data, name, maximum = Number.MAX_SAFE_INTEGER) {
  const raw = data[name];
  if (raw === undefined || raw.length === 0 || !/^[0-9]+$/u.test(raw)) {
    invalid("invalid_ui_configuration");
  }
  const value = Number(raw);
  integer(value, 1, maximum);
  return value;
}

function configBoolean(data, name) {
  const raw = data[name];
  if (raw !== "true" && raw !== "false") {
    invalid("invalid_ui_configuration");
  }
  return raw === "true";
}

export function validateUiConfig(element) {
  if (!(element instanceof HTMLElement)) {
    invalid("invalid_ui_configuration");
  }
  const data = element.dataset;
  const responseBytes = Object.freeze({
    health: positiveConfigInteger(
      data,
      "responseHealth",
      ABSOLUTE_RESPONSE_BYTES,
    ),
    documents: positiveConfigInteger(
      data,
      "responseDocuments",
      ABSOLUTE_RESPONSE_BYTES,
    ),
    metadata: positiveConfigInteger(
      data,
      "responseMetadata",
      ABSOLUTE_RESPONSE_BYTES,
    ),
    index: positiveConfigInteger(
      data,
      "responseIndex",
      ABSOLUTE_RESPONSE_BYTES,
    ),
    search: positiveConfigInteger(
      data,
      "responseSearch",
      ABSOLUTE_RESPONSE_BYTES,
    ),
    answer: positiveConfigInteger(
      data,
      "responseAnswer",
      ABSOLUTE_RESPONSE_BYTES,
    ),
    error: positiveConfigInteger(
      data,
      "responseError",
      ABSOLUTE_RESPONSE_BYTES,
    ),
  });
  const parsed = {
    pageSize: positiveConfigInteger(data, "documentPageSize"),
    pageMaximum: positiveConfigInteger(data, "documentPageMaximum"),
    queryCharacters: positiveConfigInteger(data, "queryCharacters"),
    retrievalTopKDefault: positiveConfigInteger(data, "retrievalTopKDefault"),
    retrievalTopK: positiveConfigInteger(data, "retrievalTopKMaximum"),
    questionCharacters: positiveConfigInteger(data, "questionCharacters"),
    answerTopKDefault: positiveConfigInteger(data, "answerTopKDefault"),
    answerTopK: positiveConfigInteger(data, "answerTopKMaximum"),
    documentIdCharacters: positiveConfigInteger(data, "documentIdCharacters"),
    chunkIdCharacters: positiveConfigInteger(data, "chunkIdCharacters"),
    evidenceIdCharacters: positiveConfigInteger(data, "evidenceIdCharacters"),
    claimIdCharacters: positiveConfigInteger(data, "claimIdCharacters"),
    filenameCharacters: positiveConfigInteger(data, "filenameCharacters"),
    titleCharacters: positiveConfigInteger(data, "titleCharacters"),
    subjectCharacters: positiveConfigInteger(data, "subjectCharacters"),
    sectionCharacters: positiveConfigInteger(data, "sectionCharacters"),
    chunkCharacters: positiveConfigInteger(data, "chunkCharacters"),
    answerEvidenceCount: positiveConfigInteger(data, "answerEvidenceCount"),
    evidenceCharacters: positiveConfigInteger(data, "answerEvidenceCharacters"),
    claims: positiveConfigInteger(data, "claimCount"),
    claimCharacters: positiveConfigInteger(data, "claimCharacters"),
    supports: positiveConfigInteger(data, "supportsPerClaim"),
    quoteCharacters: positiveConfigInteger(data, "quoteCharacters"),
    conflicts: positiveConfigInteger(data, "conflictEvidenceCount"),
    searchAvailable: configBoolean(data, "searchAvailable"),
    answerAvailable: configBoolean(data, "answerAvailable"),
    responseBytes,
  };
  if (
    parsed.pageSize > parsed.pageMaximum ||
    parsed.retrievalTopKDefault > parsed.retrievalTopK ||
    parsed.answerTopKDefault > parsed.answerTopK ||
    parsed.answerTopK > parsed.retrievalTopK
  ) {
    invalid("invalid_ui_configuration");
  }
  return Object.freeze(parsed);
}

const PYTHON_WHITESPACE =
  /[\u0009-\u000d\u001c-\u001f\u0020\u0085\u00a0\u1680\u2000-\u200a\u2028\u2029\u202f\u205f\u3000]+/u;

export function normalizeGroundingWhitespace(text) {
  return text.split(PYTHON_WHITESPACE).filter(Boolean).join(" ");
}

export function quoteMatchesEvidence(quote, evidence) {
  const normalizedQuote = normalizeGroundingWhitespace(quote);
  return (
    normalizedQuote.length > 0 &&
    normalizeGroundingWhitespace(evidence).includes(normalizedQuote)
  );
}

function validateProcessingProfile(value) {
  exactKeys(value, [
    "processing_profile_id",
    "processing_format_version",
    "extraction_format_version",
    "chunking_format_version",
    "chunk_max_characters",
    "chunk_overlap_characters",
  ]);
  nonempty(value.processing_profile_id, 128);
  nonempty(value.processing_format_version, 64);
  nonempty(value.extraction_format_version, 64);
  nonempty(value.chunking_format_version, 64);
  integer(value.chunk_max_characters, 1);
  integer(value.chunk_overlap_characters);
  return value;
}

function validateDocumentSummary(value, limits) {
  exactKeys(value, [
    "document_id",
    "filename",
    "title",
    "page_count",
    "chunk_count",
    "processing_profile",
    "ingested_at",
  ]);
  nonempty(value.document_id, limits.documentIdCharacters);
  nonempty(value.filename, limits.filenameCharacters);
  stringValue(value.title, limits.titleCharacters);
  integer(value.page_count, 1);
  integer(value.chunk_count);
  validateProcessingProfile(value.processing_profile);
  nonempty(value.ingested_at, 64);
  return value;
}

export function validateHealth(value) {
  exactKeys(value, ["name", "version", "status"]);
  nonempty(value.name, 128);
  nonempty(value.version, 64);
  if (value.status !== "healthy") {
    invalid();
  }
  return value;
}

export function validateDocumentList(
  value,
  limits,
  expectedOffset,
  expectedLimit,
) {
  exactKeys(value, ["offset", "limit", "total", "documents"]);
  integer(value.offset);
  integer(value.limit, 1, limits.pageMaximum);
  integer(value.total);
  if (
    (expectedOffset !== undefined && value.offset !== expectedOffset) ||
    (expectedLimit !== undefined && value.limit !== expectedLimit) ||
    value.offset > value.total ||
    (value.total > 0 && value.offset === value.total)
  ) {
    invalid();
  }
  const documents = boundedArray(value.documents, value.limit);
  const expectedCount = Math.min(value.limit, value.total - value.offset);
  if (documents.length !== expectedCount) {
    invalid();
  }
  documents.forEach((item) => validateDocumentSummary(item, limits));
  unique(documents.map((item) => item.document_id));
  return value;
}

export function validateDocumentMetadata(value, limits) {
  exactKeys(value, [
    "document_id",
    "filename",
    "sha256",
    "title",
    "author",
    "subject",
    "page_count",
    "extracted_character_count",
    "ingested_at",
    "extraction_format_version",
    "chunking_format_version",
    "processing_format_version",
    "processing_profile_id",
    "chunk_max_characters",
    "chunk_overlap_characters",
    "encrypted",
    "file_size_bytes",
  ]);
  nonempty(value.document_id, limits.documentIdCharacters);
  nonempty(value.filename, limits.filenameCharacters);
  nonempty(value.sha256, 64);
  stringValue(value.title, limits.titleCharacters);
  nullableString(value.author, limits.titleCharacters);
  nullableString(value.subject, limits.subjectCharacters);
  integer(value.page_count, 1);
  integer(value.extracted_character_count);
  nonempty(value.ingested_at, 64);
  nonempty(value.extraction_format_version, 64);
  nonempty(value.chunking_format_version, 64);
  nonempty(value.processing_format_version, 64);
  nonempty(value.processing_profile_id, 128);
  integer(value.chunk_max_characters, 1);
  integer(value.chunk_overlap_characters);
  if (value.encrypted !== false) {
    invalid();
  }
  integer(value.file_size_bytes, 1);
  return value;
}

export function validateIndexStatus(value) {
  exactKeys(value, [
    "status",
    "index_profile_id",
    "corpus_fingerprint",
    "index_format_version",
    "embedding_model_name",
    "embedding_model_revision",
    "index_type",
    "similarity_metric",
    "vector_count",
    "indexed_chunk_count",
    "indexed_character_count",
    "document_count",
    "zero_chunk_document_count",
    "built_at",
  ]);
  if (value.status !== "ready") {
    invalid();
  }
  nonempty(value.index_profile_id, 128);
  nonempty(value.corpus_fingerprint, 128);
  nonempty(value.index_format_version, 64);
  nonempty(value.embedding_model_name, 256);
  nullableString(value.embedding_model_revision, 256);
  if (!["IndexFlatIP", "IndexFlatL2"].includes(value.index_type)) {
    invalid();
  }
  if (!["cosine", "inverse_squared_l2"].includes(value.similarity_metric)) {
    invalid();
  }
  integer(value.vector_count, 1);
  integer(value.indexed_chunk_count, 1);
  integer(value.indexed_character_count, 1);
  integer(value.document_count, 1);
  integer(value.zero_chunk_document_count);
  if (
    value.vector_count !== value.indexed_chunk_count ||
    value.zero_chunk_document_count > value.document_count
  ) {
    invalid();
  }
  nonempty(value.built_at, 64);
  return value;
}

function validateSearchResult(value, limits) {
  exactKeys(value, [
    "rank",
    "score",
    "document_id",
    "filename",
    "title",
    "chunk_id",
    "chunk_order",
    "page_start",
    "page_end",
    "section_heading",
    "text",
    "character_count",
  ]);
  integer(value.rank, 1);
  finite(value.score);
  nonempty(value.document_id, limits.documentIdCharacters);
  nonempty(value.filename, limits.filenameCharacters);
  stringValue(value.title, limits.titleCharacters);
  nonempty(value.chunk_id, limits.chunkIdCharacters);
  integer(value.chunk_order, 1);
  integer(value.page_start, 1);
  integer(value.page_end, value.page_start);
  nullableString(value.section_heading, limits.sectionCharacters);
  nonempty(value.text, limits.chunkCharacters);
  integer(value.character_count, 1);
}

export function validateSearchResponse(value, expectedQuery, limits) {
  exactKeys(value, [
    "query",
    "index_profile_id",
    "corpus_fingerprint",
    "similarity_metric",
    "requested_top_k",
    "returned_count",
    "results",
  ]);
  if (nonempty(value.query, limits.queryCharacters) !== expectedQuery.trim()) {
    invalid();
  }
  nonempty(value.index_profile_id, 128);
  nonempty(value.corpus_fingerprint, 128);
  nonempty(value.similarity_metric, 64);
  integer(value.requested_top_k, 1, limits.retrievalTopK);
  integer(value.returned_count, 0, value.requested_top_k);
  const results = boundedArray(value.results, value.requested_top_k);
  if (results.length !== value.returned_count) {
    invalid();
  }
  results.forEach((item, index) => {
    validateSearchResult(item, limits);
    if (item.rank !== index + 1) {
      invalid();
    }
  });
  unique(results.map((item) => item.chunk_id));
  unique(results.map((item) => item.rank));
  return value;
}

function validateEvidence(value, limits) {
  exactKeys(value, [
    "evidence_id",
    "rank",
    "score",
    "document_id",
    "filename",
    "title",
    "chunk_id",
    "chunk_order",
    "page_start",
    "page_end",
    "section_heading",
    "text",
    "character_count",
    "truncated",
  ]);
  nonempty(value.evidence_id, limits.evidenceIdCharacters);
  integer(value.rank, 1);
  finite(value.score);
  nonempty(value.document_id, limits.documentIdCharacters);
  nonempty(value.filename, limits.filenameCharacters);
  stringValue(value.title, limits.titleCharacters);
  nonempty(value.chunk_id, limits.chunkIdCharacters);
  integer(value.chunk_order, 1);
  integer(value.page_start, 1);
  integer(value.page_end, value.page_start);
  nullableString(value.section_heading, limits.sectionCharacters);
  nonempty(value.text, limits.evidenceCharacters);
  integer(value.character_count, 1);
  booleanValue(value.truncated);
}

function validateCitation(value, limits) {
  exactKeys(value, [
    "number",
    "evidence_id",
    "score",
    "document_id",
    "title",
    "filename",
    "page_start",
    "page_end",
    "section_heading",
    "chunk_id",
  ]);
  integer(value.number, 1);
  nonempty(value.evidence_id, limits.evidenceIdCharacters);
  finite(value.score);
  nonempty(value.document_id, limits.documentIdCharacters);
  stringValue(value.title, limits.titleCharacters);
  nonempty(value.filename, limits.filenameCharacters);
  integer(value.page_start, 1);
  integer(value.page_end, value.page_start);
  nullableString(value.section_heading, limits.sectionCharacters);
  nonempty(value.chunk_id, limits.chunkIdCharacters);
}

function validateGrounding(value) {
  exactKeys(value, [
    "claim_count",
    "supported_claim_count",
    "citation_count",
    "fabricated_evidence_id_count",
    "invalid_quote_count",
    "unsupported_claim_count",
    "semantic_mismatch_count",
  ]);
  Object.values(value).forEach((item) => integer(item));
}

function sameProvenance(citation, evidence) {
  return (
    citation.evidence_id === evidence.evidence_id &&
    citation.score === evidence.score &&
    citation.document_id === evidence.document_id &&
    citation.title === evidence.title &&
    citation.filename === evidence.filename &&
    citation.page_start === evidence.page_start &&
    citation.page_end === evidence.page_end &&
    citation.section_heading === evidence.section_heading &&
    citation.chunk_id === evidence.chunk_id
  );
}

export function validateAnswerResponse(value, expectedQuestion, limits) {
  exactKeys(value, [
    "question",
    "abstained",
    "abstention_reason",
    "explanation",
    "answer",
    "claims",
    "citations",
    "evidence",
    "index_profile_id",
    "corpus_fingerprint",
    "requested_top_k",
    "generation_format_version",
    "provider",
    "context_character_count",
    "prompt_token_count",
    "output_token_count",
    "latency_seconds",
    "audit_id",
    "grounding",
  ]);
  if (
    nonempty(value.question, limits.questionCharacters) !==
    expectedQuestion.trim()
  ) {
    invalid();
  }
  booleanValue(value.abstained);
  nullableString(value.abstention_reason, 64);
  nullableString(value.explanation, 4096);
  nullableString(value.answer, limits.claimCharacters * limits.claims * 2);
  integer(value.requested_top_k, 1, limits.topK);
  nonempty(value.generation_format_version, 64);
  exactKeys(value.provider, ["name", "revision", "device"]);
  nonempty(value.provider.name, 256);
  nonempty(value.provider.revision, 128);
  nonempty(value.provider.device, 64);
  integer(value.context_character_count);
  integer(value.prompt_token_count);
  integer(value.output_token_count);
  finite(value.latency_seconds);
  nonempty(value.audit_id, 128);
  validateGrounding(value.grounding);

  const evidence = boundedArray(value.evidence, limits.evidence);
  evidence.forEach((item) => validateEvidence(item, limits));
  unique(evidence.map((item) => item.evidence_id));
  const evidenceById = new Map(
    evidence.map((item) => [item.evidence_id, item]),
  );

  const citations = boundedArray(
    value.citations,
    limits.claims * limits.supports,
  );
  citations.forEach((item) => validateCitation(item, limits));
  unique(citations.map((item) => item.number));
  unique(citations.map((item) => item.evidence_id));
  const citationByNumber = new Map(
    citations.map((item) => [item.number, item]),
  );
  for (const citation of citations) {
    const passage = evidenceById.get(citation.evidence_id);
    if (passage === undefined || !sameProvenance(citation, passage)) {
      invalid();
    }
  }

  const claims = boundedArray(value.claims, limits.claims);
  unique(claims.map((item) => item.claim_id));
  for (const claim of claims) {
    exactKeys(claim, ["claim_id", "text", "supports"]);
    nonempty(claim.claim_id, limits.claimIdCharacters);
    nonempty(claim.text, limits.claimCharacters);
    const supports = boundedArray(claim.supports, limits.supports);
    if (supports.length === 0) {
      invalid();
    }
    unique(supports.map((item) => item.evidence_id));
    for (const support of supports) {
      exactKeys(support, [
        "claim_id",
        "evidence_id",
        "citation_number",
        "quote",
      ]);
      if (support.claim_id !== claim.claim_id) {
        invalid();
      }
      nonempty(support.evidence_id, limits.evidenceIdCharacters);
      integer(support.citation_number, 1);
      const quote = nullableString(support.quote, limits.quoteCharacters);
      const citation = citationByNumber.get(support.citation_number);
      const passage = evidenceById.get(support.evidence_id);
      if (
        citation === undefined ||
        passage === undefined ||
        citation.evidence_id !== support.evidence_id ||
        (quote !== null && !quoteMatchesEvidence(quote, passage.text))
      ) {
        invalid();
      }
    }
  }

  if (value.abstained) {
    if (
      !ABSTENTION_REASONS.has(value.abstention_reason) ||
      value.explanation === null ||
      value.explanation.length === 0 ||
      value.answer !== null ||
      claims.length !== 0
    ) {
      invalid();
    }
    if (
      value.abstention_reason === "conflicting_evidence" &&
      (citations.length < 2 || citations.length > limits.conflicts)
    ) {
      invalid();
    }
    if (
      value.abstention_reason !== "conflicting_evidence" &&
      citations.length !== 0
    ) {
      invalid();
    }
  } else if (
    value.abstention_reason !== null ||
    value.explanation !== null ||
    value.answer === null ||
    value.answer.length === 0 ||
    claims.length === 0 ||
    citations.length === 0
  ) {
    invalid();
  }
  return value;
}

export function validateErrorResponse(value) {
  const keys = Object.keys(record(value));
  if (
    !keys.includes("error") ||
    keys.some((key) => !["error", "details"].includes(key))
  ) {
    invalid();
  }
  exactKeys(value.error, ["code", "message"]);
  nonempty(value.error.code, 128);
  nonempty(value.error.message, 1024);
  if (
    "details" in value &&
    value.details !== null &&
    (typeof value.details !== "object" || Array.isArray(value.details))
  ) {
    invalid();
  }
  return value;
}

export function createRequestController() {
  let token = 0;
  /** @type {AbortController | null} */
  let controller = null;
  return {
    begin() {
      if (controller !== null) {
        controller.abort();
      }
      token += 1;
      controller = new AbortController();
      return { token, signal: controller.signal };
    },
    current(candidate) {
      return candidate === token;
    },
    cancel() {
      token += 1;
      if (controller !== null) {
        controller.abort();
        controller = null;
      }
    },
    finish(candidate) {
      if (candidate === token) {
        controller = null;
        return true;
      }
      return false;
    },
    active() {
      return controller !== null;
    },
  };
}

function element(name, text, className) {
  const node = document.createElement(name);
  if (text !== undefined) {
    node.textContent = text;
  }
  if (className !== undefined) {
    node.className = className;
  }
  return node;
}

/**
 * @template {Element} T
 * @param {string} selector
 * @param {{new (...args: any[]): T}} expectedType
 * @returns {T}
 */
function requiredElement(selector, expectedType) {
  const value = document.querySelector(selector);
  if (!(value instanceof expectedType)) {
    invalid();
  }
  return /** @type {T} */ (value);
}

function formControl(form, name) {
  const value = form.elements.namedItem(name);
  if (
    !(value instanceof HTMLInputElement) &&
    !(value instanceof HTMLTextAreaElement)
  ) {
    invalid();
  }
  return value;
}

function metadataList(entries) {
  const list = element("dl", undefined, "metadata-grid");
  for (const [label, value] of entries) {
    list.append(element("dt", label), element("dd", String(value)));
  }
  return list;
}

function pages(item) {
  return item.page_start === item.page_end
    ? `Page ${item.page_start}`
    : `Pages ${item.page_start}–${item.page_end}`;
}

function safeError(code) {
  return ERROR_MESSAGES[code] ?? "The request could not be completed safely.";
}

async function apiRequest(url, options, maximum, errorMaximum) {
  const response = await fetch(url, {
    credentials: "omit",
    cache: "no-store",
    ...options,
  });
  const value = await readBoundedJson(
    response,
    response.ok ? maximum : errorMaximum,
  );
  if (!response.ok) {
    if (response.status === 422) {
      throw new Error("request_validation");
    }
    try {
      const parsed = validateErrorResponse(value);
      throw new Error(parsed.error.code);
    } catch (error) {
      if (error instanceof ResponseValidationError) {
        throw new Error("operational_failure");
      }
      throw error;
    }
  }
  return value;
}

function announce(target, text, alert = false) {
  target.textContent = text;
  if (alert) {
    target.setAttribute("role", "alert");
  } else {
    target.removeAttribute("role");
  }
}

function renderFailure(container, message) {
  const state = element("div", undefined, "state state-error");
  state.setAttribute("role", "alert");
  state.append(element("h3", "Request not completed"), element("p", message));
  container.replaceChildren(state);
}

function workflowError(error) {
  if (error instanceof ResponseValidationError) {
    if (error.message === "unexpected_response_media_type") {
      return "The backend returned an unexpected response media type.";
    }
    if (error.message === "invalid_ui_configuration") {
      return "The interface configuration is invalid.";
    }
    return "The backend response failed presentation-integrity validation.";
  }
  if (error instanceof Error && error.name === "AbortError") {
    return "cancelled";
  }
  if (error instanceof Error && error.message === "request_validation") {
    return "The request is invalid. Check the entered values.";
  }
  if (error instanceof Error && error.message === "operational_failure") {
    return "The backend returned an invalid error response.";
  }
  if (error instanceof Error && error.message in ERROR_MESSAGES) {
    return safeError(error.message);
  }
  return "A connectivity failure prevented the request from completing.";
}

function config() {
  const configElement = document.querySelector("#ui-config");
  if (!(configElement instanceof HTMLElement)) {
    invalid("invalid_ui_configuration");
  }
  const parsed = validateUiConfig(configElement);
  return Object.freeze({
    ...parsed,
    answer: Object.freeze({
      questionCharacters: parsed.questionCharacters,
      topK: parsed.answerTopK,
      evidence: parsed.answerEvidenceCount,
      evidenceCharacters: parsed.evidenceCharacters,
      claims: parsed.claims,
      claimCharacters: parsed.claimCharacters,
      supports: parsed.supports,
      quoteCharacters: parsed.quoteCharacters,
      conflicts: parsed.conflicts,
      documentIdCharacters: parsed.documentIdCharacters,
      chunkIdCharacters: parsed.chunkIdCharacters,
      evidenceIdCharacters: parsed.evidenceIdCharacters,
      claimIdCharacters: parsed.claimIdCharacters,
      filenameCharacters: parsed.filenameCharacters,
      titleCharacters: parsed.titleCharacters,
      sectionCharacters: parsed.sectionCharacters,
    }),
  });
}

function renderEvidence(item, citationNumber) {
  const card = element("article", undefined, "evidence-card");
  if (citationNumber !== undefined) {
    card.id = `evidence-${citationNumber}`;
    card.tabIndex = -1;
  }
  card.append(
    element("p", "Untrusted document evidence", "evidence-label"),
    element(
      "h3",
      citationNumber === undefined
        ? `Rank ${item.rank}: ${item.title || item.filename}`
        : `[${citationNumber}] ${item.title || item.filename}`,
    ),
    metadataList([
      ["Filename", item.filename],
      ["Page", pages(item)],
      ["Section", item.section_heading ?? "Not specified"],
      ["Chunk", item.chunk_id],
      ["Score", item.score],
    ]),
    element("blockquote", item.text, "evidence-text"),
  );
  return card;
}

function documentSignature(item) {
  return JSON.stringify(item);
}

export function createDocumentFilterChooser(container, settings) {
  const controller = createRequestController();
  const seen = new Map();
  const selected = new Map();
  let establishedTotal;
  let committedOffset = 0;
  let committedLimit = settings.pageSize;
  let committedTotal = 0;
  let currentDocuments = [];

  const list = element("div", undefined, "filter-list");
  const selectedSummary = element("div", undefined, "selected-documents");
  const pageStatus = element("span", "Loading documents…");
  pageStatus.setAttribute("aria-live", "polite");
  const previous = element("button", "Previous documents", "secondary");
  const next = element("button", "Next documents", "secondary");
  previous.type = "button";
  next.type = "button";
  previous.disabled = true;
  next.disabled = true;
  const controls = element("div", undefined, "filter-pagination");
  controls.append(previous, pageStatus, next);
  container.replaceChildren(selectedSummary, list, controls);

  function renderSelected() {
    const wrapper = element("div");
    wrapper.append(
      element(
        "p",
        `${selected.size} document filter${selected.size === 1 ? "" : "s"} selected`,
        "selected-count",
      ),
    );
    if (selected.size > 0) {
      const values = element("ul", undefined, "selected-document-list");
      for (const [documentId, item] of selected) {
        const row = element("li");
        const remove = element(
          "button",
          `Remove ${item.title || item.filename}`,
          "secondary",
        );
        remove.type = "button";
        remove.addEventListener("click", () => {
          selected.delete(documentId);
          renderSelected();
          renderPage(currentDocuments);
        });
        row.append(
          element("span", `${item.title || item.filename} — ${documentId}`),
          remove,
        );
        values.append(row);
      }
      wrapper.append(values);
    }
    selectedSummary.replaceChildren(wrapper);
  }

  function renderPage(documents) {
    currentDocuments = documents;
    const fragment = document.createDocumentFragment();
    for (const item of documents) {
      const label = element("label", undefined, "filter-option");
      const input = document.createElement("input");
      input.type = "checkbox";
      input.name = "document_ids";
      input.value = item.document_id;
      input.checked = selected.has(item.document_id);
      input.addEventListener("change", () => {
        if (input.checked) {
          selected.set(item.document_id, item);
        } else {
          selected.delete(item.document_id);
        }
        renderSelected();
      });
      const name = item.title || item.filename;
      label.append(input, element("span", `${name} — ${item.document_id}`));
      fragment.append(label);
    }
    if (documents.length === 0) {
      fragment.append(element("p", "No documents are available."));
    }
    list.replaceChildren(fragment);
  }

  function restoreControls() {
    previous.disabled = committedOffset === 0;
    next.disabled =
      committedTotal === 0 ||
      committedOffset + committedLimit >= committedTotal;
  }

  async function load(requestedOffset) {
    const request = controller.begin();
    const requestedLimit = settings.pageSize;
    previous.disabled = true;
    next.disabled = true;
    list.setAttribute("aria-busy", "true");
    try {
      const parsed = validateDocumentList(
        await apiRequest(
          `/documents?offset=${requestedOffset}&limit=${requestedLimit}`,
          { signal: request.signal },
          settings.responseBytes.documents,
          settings.responseBytes.error,
        ),
        settings,
        requestedOffset,
        requestedLimit,
      );
      if (!controller.current(request.token)) {
        return;
      }
      if (establishedTotal !== undefined && parsed.total !== establishedTotal) {
        invalid();
      }
      for (const item of parsed.documents) {
        const prior = seen.get(item.document_id);
        const signature = documentSignature(item);
        if (
          prior !== undefined &&
          (prior.offset !== parsed.offset || prior.signature !== signature)
        ) {
          invalid();
        }
        seen.set(item.document_id, { offset: parsed.offset, signature });
      }
      establishedTotal = parsed.total;
      committedOffset = parsed.offset;
      committedLimit = parsed.limit;
      committedTotal = parsed.total;
      renderPage(parsed.documents);
      renderSelected();
      pageStatus.textContent =
        parsed.total === 0
          ? "No documents"
          : `${parsed.offset + 1}–${parsed.offset + parsed.documents.length} of ${parsed.total}`;
    } catch (error) {
      if (!controller.current(request.token)) {
        return;
      }
      renderFailure(list, workflowError(error));
      pageStatus.textContent = "Document filters failed to load.";
    } finally {
      if (controller.finish(request.token)) {
        list.setAttribute("aria-busy", "false");
        restoreControls();
      }
    }
  }

  previous.addEventListener("click", () => {
    void load(Math.max(0, committedOffset - settings.pageSize));
  });
  next.addEventListener("click", () => {
    void load(committedOffset + committedLimit);
  });
  renderSelected();
  return Object.freeze({
    ready: load(0),
    selectedIds: () => [...selected.keys()],
    loadOffset: load,
    cancel: () => controller.cancel(),
  });
}

async function initializeHome(settings) {
  const health = requiredElement("#health-summary", HTMLElement);
  const index = requiredElement("#index-summary", HTMLElement);
  try {
    const value = validateHealth(
      await apiRequest(
        "/health",
        {},
        settings.responseBytes.health,
        settings.responseBytes.error,
      ),
    );
    health.replaceChildren(
      metadataList([
        ["Name", value.name],
        ["Version", value.version],
        ["Status", value.status],
      ]),
    );
  } catch (error) {
    renderFailure(health, workflowError(error));
  } finally {
    health.setAttribute("aria-busy", "false");
  }
  try {
    const value = validateIndexStatus(
      await apiRequest(
        "/ui/api/index-status",
        {},
        settings.responseBytes.index,
        settings.responseBytes.error,
      ),
    );
    index.replaceChildren(
      metadataList([
        ["Readiness", "Ready"],
        ["Documents", value.document_count],
        ["Zero-chunk documents", value.zero_chunk_document_count],
        ["Chunks", value.indexed_chunk_count],
        ["Vectors", value.vector_count],
        ["Index profile", value.index_profile_id],
        ["Built", value.built_at],
      ]),
    );
  } catch (error) {
    const message = workflowError(error);
    const label =
      error instanceof Error && error.message === "index_not_ready"
        ? "No active index"
        : message;
    renderFailure(index, label);
  } finally {
    index.setAttribute("aria-busy", "false");
  }
}

async function initializeDocuments(settings) {
  const list = requiredElement("#document-list", HTMLElement);
  const status = requiredElement("#document-status", HTMLElement);
  const previous = requiredElement("#documents-previous", HTMLButtonElement);
  const next = requiredElement("#documents-next", HTMLButtonElement);
  const pageLabel = requiredElement("#documents-page", HTMLElement);
  const metadataSection = requiredElement("#document-metadata", HTMLElement);
  const metadataContent = requiredElement("#metadata-content", HTMLElement);
  const metadataClose = requiredElement("#metadata-close", HTMLButtonElement);
  const listController = createRequestController();
  const metadataController = createRequestController();
  let committedOffset = 0;
  let committedLimit = settings.pageSize;
  let committedTotal = 0;
  /** @type {HTMLButtonElement | null} */
  let activeMetadataButton = null;

  function restorePagination() {
    previous.disabled = committedOffset === 0;
    next.disabled =
      committedTotal === 0 ||
      committedOffset + committedLimit >= committedTotal;
  }

  function closeMetadata() {
    metadataController.cancel();
    metadataSection.hidden = true;
    metadataContent.replaceChildren();
    if (activeMetadataButton !== null) {
      activeMetadataButton.disabled = false;
      activeMetadataButton = null;
    }
  }

  async function load(requestedOffset) {
    const requestedLimit = settings.pageSize;
    const request = listController.begin();
    previous.disabled = true;
    next.disabled = true;
    list.setAttribute("aria-busy", "true");
    try {
      const value = validateDocumentList(
        await apiRequest(
          `/documents?offset=${requestedOffset}&limit=${requestedLimit}`,
          { signal: request.signal },
          settings.responseBytes.documents,
          settings.responseBytes.error,
        ),
        settings,
        requestedOffset,
        requestedLimit,
      );
      if (!listController.current(request.token)) {
        return;
      }
      closeMetadata();
      const fragment = document.createDocumentFragment();
      for (const item of value.documents) {
        const card = element("article", undefined, "document-card");
        const button = element("button", "View metadata", "secondary");
        button.type = "button";
        button.setAttribute(
          "aria-label",
          `View metadata for ${item.title || item.filename}`,
        );
        button.addEventListener("click", async () => {
          if (activeMetadataButton !== null) {
            activeMetadataButton.disabled = false;
          }
          activeMetadataButton = button;
          const metadataRequest = metadataController.begin();
          const requestedDocumentId = item.document_id;
          button.disabled = true;
          metadataSection.hidden = false;
          metadataContent.replaceChildren(element("p", "Loading metadata…"));
          try {
            const metadata = validateDocumentMetadata(
              await apiRequest(
                `/documents/${encodeURIComponent(requestedDocumentId)}`,
                { signal: metadataRequest.signal },
                settings.responseBytes.metadata,
                settings.responseBytes.error,
              ),
              settings,
            );
            if (
              !metadataController.current(metadataRequest.token) ||
              metadata.document_id !== requestedDocumentId
            ) {
              return;
            }
            metadataContent.replaceChildren(
              metadataList([
                ["Document ID", metadata.document_id],
                ["Filename", metadata.filename],
                ["Title", metadata.title],
                ["Author", metadata.author ?? "Not specified"],
                ["Subject", metadata.subject ?? "Not specified"],
                ["Pages", metadata.page_count],
                ["Ingested", metadata.ingested_at],
                ["Processing profile", metadata.processing_profile_id],
                ["File size", `${metadata.file_size_bytes} bytes`],
              ]),
            );
            metadataSection.hidden = false;
            requiredElement("#metadata-heading", HTMLElement).focus();
          } catch (error) {
            if (!metadataController.current(metadataRequest.token)) {
              return;
            }
            metadataSection.hidden = false;
            renderFailure(metadataContent, workflowError(error));
            requiredElement("#metadata-heading", HTMLElement).focus();
          } finally {
            if (metadataController.finish(metadataRequest.token)) {
              button.disabled = false;
              activeMetadataButton = null;
            }
          }
        });
        card.append(
          element("h2", item.title || item.filename),
          metadataList([
            ["Document ID", item.document_id],
            ["Filename", item.filename],
            ["Pages", item.page_count],
            ["Chunks", item.chunk_count],
            [
              "Processing profile",
              item.processing_profile.processing_profile_id,
            ],
            ["Ingested", item.ingested_at],
          ]),
          button,
        );
        fragment.append(card);
      }
      if (value.documents.length === 0) {
        fragment.append(element("p", "No documents are available."));
      }
      committedOffset = value.offset;
      committedLimit = value.limit;
      committedTotal = value.total;
      list.replaceChildren(fragment);
      pageLabel.textContent =
        value.total === 0
          ? "No documents"
          : `${value.offset + 1}–${value.offset + value.documents.length} of ${value.total}`;
      announce(status, "Document library loaded.");
    } catch (error) {
      if (!listController.current(request.token)) {
        return;
      }
      renderFailure(list, workflowError(error));
      announce(status, "Document library failed to load.", true);
    } finally {
      if (listController.finish(request.token)) {
        list.setAttribute("aria-busy", "false");
        restorePagination();
      }
    }
  }
  previous.addEventListener("click", () => {
    void load(Math.max(0, committedOffset - settings.pageSize));
  });
  next.addEventListener("click", () => {
    void load(committedOffset + committedLimit);
  });
  metadataClose.addEventListener("click", closeMetadata);
  await load(0);
}

async function initializeSearch(settings) {
  const form = requiredElement("#search-form", HTMLFormElement);
  const filters = requiredElement("#search-documents", HTMLElement);
  const submit = requiredElement("#search-submit", HTMLButtonElement);
  const cancel = requiredElement("#search-cancel", HTMLButtonElement);
  const status = requiredElement("#search-status", HTMLElement);
  const output = requiredElement("#search-output", HTMLElement);
  const results = requiredElement("#search-results", HTMLElement);
  if (!settings.searchAvailable) {
    submit.disabled = true;
    renderFailure(
      results,
      "Evidence search is unavailable in this browser configuration.",
    );
    output.hidden = false;
    announce(status, "Evidence search is unavailable.", true);
    return;
  }
  const controller = createRequestController();
  const documentChooser = createDocumentFilterChooser(filters, settings);
  await documentChooser.ready;
  cancel.addEventListener("click", () => {
    controller.cancel();
    submit.disabled = false;
    cancel.hidden = true;
    announce(status, "Search cancelled.");
  });
  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    if (controller.active() || !form.reportValidity()) {
      return;
    }
    const query = formControl(form, "query").value.trim();
    const topK = Number(formControl(form, "top_k").value);
    const request = controller.begin();
    submit.disabled = true;
    cancel.hidden = false;
    announce(status, "Searching evidence…");
    try {
      const documentIds = documentChooser.selectedIds();
      const payload = { query, top_k: topK };
      if (documentIds.length > 0) {
        payload.document_ids = documentIds;
      }
      const value = validateSearchResponse(
        await apiRequest(
          "/search",
          {
            method: "POST",
            headers: { "content-type": "application/json" },
            body: JSON.stringify(payload),
            signal: request.signal,
          },
          settings.responseBytes.search,
          settings.responseBytes.error,
        ),
        query,
        settings,
      );
      if (!controller.current(request.token)) {
        return;
      }
      const fragment = document.createDocumentFragment();
      value.results.forEach((item) => fragment.append(renderEvidence(item)));
      if (value.results.length === 0) {
        fragment.append(
          element(
            "div",
            "No result survived the configured evidence threshold and filters.",
            "state state-abstention",
          ),
        );
      }
      results.replaceChildren(fragment);
      output.hidden = false;
      requiredElement("#search-results-heading", HTMLElement).focus();
      announce(status, `${value.returned_count} evidence results returned.`);
    } catch (error) {
      if (!controller.current(request.token)) {
        return;
      }
      const message = workflowError(error);
      if (message !== "cancelled") {
        output.hidden = false;
        renderFailure(results, message);
        requiredElement("#search-results-heading", HTMLElement).focus();
        announce(status, "Search failed.", true);
      }
    } finally {
      if (controller.finish(request.token)) {
        submit.disabled = false;
        cancel.hidden = true;
      }
    }
  });
}

function technicalPanel(answer) {
  const panel = element("details", undefined, "technical-panel");
  panel.append(
    element("summary", "Technical details"),
    metadataList([
      ["Audit ID", answer.audit_id],
      ["Provider", answer.provider.name],
      ["Provider revision", answer.provider.revision],
      ["Device", answer.provider.device],
      ["Index profile", answer.index_profile_id || "Not applicable"],
      ["Corpus fingerprint", answer.corpus_fingerprint || "Not applicable"],
      ["Prompt tokens", answer.prompt_token_count],
      ["Output tokens", answer.output_token_count],
      ["Latency", `${answer.latency_seconds} seconds`],
    ]),
  );
  return panel;
}

export function citationCopyText(citation) {
  const section =
    citation.section_heading === null ? "" : `, ${citation.section_heading}`;
  return `[${citation.number}] ${citation.title} (${citation.filename}), ${pages(citation)}${section}, chunk ${citation.chunk_id}`;
}

function copyButton(citation, status) {
  const button = element("button", "Copy citation", "copy-citation");
  button.type = "button";
  button.setAttribute("aria-label", `Copy citation ${citation.number}`);
  button.addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText(citationCopyText(citation));
      announce(status, `Citation ${citation.number} copied.`);
    } catch {
      announce(status, "Citation could not be copied.", true);
    }
  });
  return button;
}

function renderAnswer(answer, container, status) {
  const evidenceById = new Map(
    answer.evidence.map((item) => [item.evidence_id, item]),
  );
  const fragment = document.createDocumentFragment();

  if (!answer.abstained) {
    const success = element("div", undefined, "state state-success");
    success.append(
      element("h3", "Supported answer"),
      element(
        "p",
        "Each generated claim below is linked to backend-validated source evidence.",
      ),
    );
    fragment.append(success);
    for (const claim of answer.claims) {
      const card = element("article", undefined, "claim-card");
      card.id = `claim-${claim.claim_id.replaceAll(/[^A-Za-z0-9_-]/gu, "-")}`;
      card.append(element("h3", claim.text));
      const supports = element("ol", undefined, "claim-supports");
      for (const support of claim.supports) {
        const item = element("li");
        const link = element("a", `Evidence [${support.citation_number}]`);
        link.href = `#evidence-${support.citation_number}`;
        item.append(link);
        if (support.quote !== null) {
          item.append(
            element(
              "p",
              "Supporting quotation verified using AeroDocs grounding whitespace rules.",
              "evidence-label",
            ),
          );
          item.append(element("blockquote", support.quote, "exact-quote"));
        }
        supports.append(item);
      }
      card.append(supports);
      fragment.append(card);
    }
    for (const citation of answer.citations) {
      const evidence = evidenceById.get(citation.evidence_id);
      const card = renderEvidence(evidence, citation.number);
      card.append(
        copyButton(citation, status),
        (() => {
          const back = element("a", "Back to supporting claim");
          const claim = answer.claims.find((candidate) =>
            candidate.supports.some(
              (support) => support.citation_number === citation.number,
            ),
          );
          back.href = `#claim-${claim.claim_id.replaceAll(/[^A-Za-z0-9_-]/gu, "-")}`;
          return back;
        })(),
      );
      fragment.append(card);
    }
  } else if (answer.abstention_reason === "conflicting_evidence") {
    const state = element("div", undefined, "state state-conflict");
    state.append(
      element("h3", "Conflicting evidence"),
      element("p", answer.explanation),
      element(
        "p",
        "Do not select one source as authoritative. Consult approved documentation and qualified engineering review.",
      ),
    );
    fragment.append(state);
    const grid = element("div", undefined, "conflict-grid");
    for (const citation of answer.citations) {
      const evidence = evidenceById.get(citation.evidence_id);
      const card = renderEvidence(evidence, citation.number);
      card.append(copyButton(citation, status));
      grid.append(card);
    }
    fragment.append(grid);
  } else {
    const labels = {
      insufficient_evidence: "Insufficient evidence",
      prompt_injection_risk: "Prompt-injection risk",
      grounding_validation_failed: "Grounding validation failed",
      safety_boundary: "Safety boundary",
    };
    const state = element("div", undefined, "state state-abstention");
    state.append(
      element("h3", labels[answer.abstention_reason]),
      element("p", answer.explanation),
    );
    fragment.append(state);
  }
  fragment.append(technicalPanel(answer));
  container.replaceChildren(fragment);
}

async function initializeAnswer(settings) {
  const form = requiredElement("#answer-form", HTMLFormElement);
  const filters = requiredElement("#answer-documents", HTMLElement);
  const submit = requiredElement("#answer-submit", HTMLButtonElement);
  const cancel = requiredElement("#answer-cancel", HTMLButtonElement);
  const status = requiredElement("#answer-status", HTMLElement);
  const output = requiredElement("#answer-output", HTMLElement);
  const results = requiredElement("#answer-results", HTMLElement);
  if (!settings.answerAvailable) {
    submit.disabled = true;
    renderFailure(
      results,
      "Grounded answers are unavailable in this browser configuration.",
    );
    output.hidden = false;
    announce(status, "Grounded answers are unavailable.", true);
    return;
  }
  const controller = createRequestController();
  const documentChooser = createDocumentFilterChooser(filters, settings);
  await documentChooser.ready;
  cancel.addEventListener("click", () => {
    controller.cancel();
    submit.disabled = false;
    cancel.hidden = true;
    announce(
      status,
      "Answer request cancelled in this browser. Backend work and audit persistence may continue.",
    );
  });
  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    if (controller.active() || !form.reportValidity()) {
      return;
    }
    const question = formControl(form, "question").value.trim();
    const topK = Number(formControl(form, "top_k").value);
    const request = controller.begin();
    submit.disabled = true;
    cancel.hidden = false;
    announce(status, "Requesting a grounded answer…");
    try {
      const documentIds = documentChooser.selectedIds();
      const payload = { question, top_k: topK };
      if (documentIds.length > 0) {
        payload.document_ids = documentIds;
      }
      const value = validateAnswerResponse(
        await apiRequest(
          "/answers",
          {
            method: "POST",
            headers: { "content-type": "application/json" },
            body: JSON.stringify(payload),
            signal: request.signal,
          },
          settings.responseBytes.answer,
          settings.responseBytes.error,
        ),
        question,
        settings.answer,
      );
      if (!controller.current(request.token)) {
        return;
      }
      renderAnswer(value, results, status);
      output.hidden = false;
      requiredElement("#answer-results-heading", HTMLElement).focus();
      announce(
        status,
        value.abstained ? "AeroDocs abstained." : "Grounded answer returned.",
      );
    } catch (error) {
      if (!controller.current(request.token)) {
        return;
      }
      const message = workflowError(error);
      if (message !== "cancelled") {
        output.hidden = false;
        renderFailure(results, message);
        requiredElement("#answer-results-heading", HTMLElement).focus();
        announce(status, "Answer request failed.", true);
      }
    } finally {
      if (controller.finish(request.token)) {
        submit.disabled = false;
        cancel.hidden = true;
      }
    }
  });
}

export async function initialize() {
  let settings;
  try {
    settings = config();
  } catch (error) {
    const main = document.querySelector("#main-content");
    if (main instanceof HTMLElement) {
      renderFailure(main, workflowError(error));
      main.focus();
    }
    const globalStatus = document.querySelector("#global-status");
    if (globalStatus instanceof HTMLElement) {
      announce(globalStatus, "The interface configuration is invalid.", true);
    }
    return;
  }
  switch (document.body.dataset.page) {
    case "home":
      await initializeHome(settings);
      break;
    case "documents":
      await initializeDocuments(settings);
      break;
    case "search":
      await initializeSearch(settings);
      break;
    case "ask":
      await initializeAnswer(settings);
      break;
  }
}

if (typeof document !== "undefined") {
  document.addEventListener("DOMContentLoaded", () => {
    void initialize();
  });
}
