import { beforeEach, describe, expect, it, vi } from "vitest";
import { readFile } from "node:fs/promises";

import {
  ResponseValidationError,
  citationCopyText,
  createDocumentFilterChooser,
  createRequestController,
  initialize,
  quoteMatchesEvidence,
  readBoundedJson,
  validateAnswerResponse,
  validateDocumentList,
  validateHealth,
  validateIndexStatus,
  validateSearchResponse,
  validateUiConfig,
} from "../../app/ui/static/app.js";

const limits = {
  pageSize: 20,
  pageMaximum: 100,
  queryCharacters: 2000,
  retrievalTopK: 20,
  questionCharacters: 2000,
  topK: 10,
  evidence: 8,
  evidenceCharacters: 3000,
  claims: 8,
  claimCharacters: 800,
  supports: 8,
  quoteCharacters: 500,
  conflicts: 8,
  documentIdCharacters: 68,
  chunkIdCharacters: 68,
  evidenceIdCharacters: 68,
  claimIdCharacters: 32,
  filenameCharacters: 200,
  titleCharacters: 300,
  subjectCharacters: 1000,
  sectionCharacters: 2000,
  chunkCharacters: 2000,
};

const configData = {
  documentPageSize: "20",
  documentPageMaximum: "100",
  queryCharacters: "2000",
  retrievalTopKDefault: "5",
  retrievalTopKMaximum: "20",
  questionCharacters: "2000",
  answerTopKDefault: "5",
  answerTopKMaximum: "10",
  documentIdCharacters: "68",
  chunkIdCharacters: "68",
  evidenceIdCharacters: "68",
  claimIdCharacters: "32",
  filenameCharacters: "200",
  titleCharacters: "300",
  subjectCharacters: "1000",
  sectionCharacters: "2000",
  chunkCharacters: "2000",
  answerEvidenceCount: "8",
  answerEvidenceCharacters: "3000",
  claimCount: "8",
  claimCharacters: "800",
  supportsPerClaim: "8",
  quoteCharacters: "500",
  conflictEvidenceCount: "8",
  searchAvailable: "true",
  answerAvailable: "true",
  responseHealth: "4096",
  responseDocuments: "524288",
  responseMetadata: "262144",
  responseIndex: "33554432",
  responseSearch: "2097152",
  responseAnswer: "2097152",
  responseError: "65536",
};

function addConfig(overrides = {}) {
  const value = document.createElement("div");
  value.id = "ui-config";
  Object.assign(value.dataset, configData, overrides);
  document.body.append(value);
  return validateUiConfig(value);
}

function evidence(overrides = {}) {
  const value = {
    evidence_id: "ev_1",
    rank: 1,
    score: 0.9,
    document_id: `doc_${"a".repeat(64)}`,
    filename: "manual.pdf",
    title: "Public Manual",
    chunk_id: "chk_1",
    chunk_order: 1,
    page_start: 2,
    page_end: 2,
    section_heading: "Inspection",
    text: "Inspect the pump visually only.",
    character_count: 31,
    truncated: false,
    ...overrides,
  };
  if ("text" in overrides && !("character_count" in overrides)) {
    value.character_count = value.text.length;
  }
  return value;
}

function citation(overrides = {}) {
  const source = evidence();
  return {
    number: 1,
    evidence_id: source.evidence_id,
    score: source.score,
    document_id: source.document_id,
    title: source.title,
    filename: source.filename,
    page_start: source.page_start,
    page_end: source.page_end,
    section_heading: source.section_heading,
    chunk_id: source.chunk_id,
    ...overrides,
  };
}

function grounding(overrides = {}) {
  return {
    claim_count: 1,
    supported_claim_count: 1,
    citation_count: 1,
    fabricated_evidence_id_count: 0,
    invalid_quote_count: 0,
    unsupported_claim_count: 0,
    semantic_mismatch_count: 0,
    ...overrides,
  };
}

function answer(overrides = {}) {
  return {
    question: "How is the pump inspected?",
    abstained: false,
    abstention_reason: null,
    explanation: null,
    answer: "Inspect the pump visually only. [1]",
    claims: [
      {
        claim_id: "claim_1",
        text: "Inspect the pump visually only.",
        supports: [
          {
            claim_id: "claim_1",
            evidence_id: "ev_1",
            citation_number: 1,
            quote: "Inspect the pump visually only.",
          },
        ],
      },
    ],
    citations: [citation()],
    evidence: [evidence()],
    index_profile_id: "ipr_1",
    corpus_fingerprint: "cor_1",
    requested_top_k: 5,
    generation_format_version: "1",
    provider: {
      name: "Qwen/Qwen2.5-1.5B-Instruct",
      revision: "a".repeat(40),
      device: "cpu",
    },
    context_character_count: 31,
    prompt_token_count: 100,
    output_token_count: 20,
    latency_seconds: 0.5,
    audit_id: "aud_1",
    grounding: grounding(),
    ...overrides,
  };
}

function documentList() {
  return {
    offset: 0,
    limit: 20,
    total: 1,
    documents: [
      {
        document_id: `doc_${"a".repeat(64)}`,
        filename: "manual.pdf",
        title: "Public Manual",
        page_count: 2,
        chunk_count: 3,
        processing_profile: {
          processing_profile_id: "prf_1",
          processing_format_version: "1",
          extraction_format_version: "1",
          chunking_format_version: "1",
          chunk_max_characters: 2000,
          chunk_overlap_characters: 200,
        },
        ingested_at: "2026-01-01T00:00:00Z",
      },
    ],
  };
}

function documentMetadata(documentId, filename) {
  return {
    document_id: documentId,
    filename,
    sha256: "a".repeat(64),
    title: filename,
    author: null,
    subject: null,
    page_count: 2,
    extracted_character_count: 100,
    ingested_at: "2026-01-01T00:00:00Z",
    extraction_format_version: "1",
    chunking_format_version: "1",
    processing_format_version: "1",
    processing_profile_id: "prf_1",
    chunk_max_characters: 2000,
    chunk_overlap_characters: 200,
    encrypted: false,
    file_size_bytes: 1000,
  };
}

function jsonResponse(value, status = 200) {
  return new Response(JSON.stringify(value), {
    status,
    headers: { "content-type": "application/json" },
  });
}

function answerPage() {
  document.body.dataset.page = "ask";
  document.body.innerHTML = `
      <div id="ui-config" hidden></div>
      <form id="answer-form">
        <textarea name="question" required></textarea>
        <input name="top_k" value="5" required>
        <div id="answer-documents"></div>
        <button id="answer-submit" type="submit">Ask</button>
        <button id="answer-cancel" type="button" hidden>Cancel</button>
      </form>
      <div id="answer-status" aria-live="polite"></div>
      <section id="answer-output" hidden>
        <h2 id="answer-results-heading" tabindex="-1">Answer</h2>
        <div id="answer-results"></div>
      </section>
  `;
  Object.assign(document.querySelector("#ui-config").dataset, configData);
}

function documentsPage() {
  document.body.dataset.page = "documents";
  document.body.innerHTML = `
    <div id="ui-config" hidden></div>
    <div id="document-status" aria-live="polite"></div>
    <div id="document-list" aria-busy="true"></div>
    <button id="documents-previous" type="button" disabled>Previous</button>
    <span id="documents-page"></span>
    <button id="documents-next" type="button" disabled>Next</button>
    <section id="document-metadata" hidden>
      <h2 id="metadata-heading" tabindex="-1">Metadata</h2>
      <button id="metadata-close" type="button">Close</button>
      <div id="metadata-content"></div>
    </section>
  `;
  Object.assign(document.querySelector("#ui-config").dataset, configData);
}

function searchPage() {
  document.body.dataset.page = "search";
  document.body.innerHTML = `
    <div id="ui-config" hidden></div>
    <form id="search-form">
      <textarea name="query" required></textarea>
      <input name="top_k" value="1" required>
      <div id="search-documents"></div>
      <button id="search-submit" type="submit">Search</button>
      <button id="search-cancel" type="button" hidden>Cancel</button>
    </form>
    <div id="search-status" aria-live="polite"></div>
    <section id="search-output" hidden>
      <h2 id="search-results-heading" tabindex="-1">Results</h2>
      <div id="search-results"></div>
    </section>
  `;
  Object.assign(document.querySelector("#ui-config").dataset, configData, {
    documentPageSize: "1",
    documentPageMaximum: "2",
  });
}

beforeEach(() => {
  document.body.replaceChildren();
  for (const key of Object.keys(document.body.dataset)) {
    delete document.body.dataset[key];
  }
  vi.restoreAllMocks();
});

describe("strict response validators", () => {
  it("requires complete internally consistent server configuration", () => {
    document.body.replaceChildren();
    expect(addConfig().chunkCharacters).toBe(2000);
    document.body.replaceChildren();
    expect(() => addConfig({ responseSearch: "" })).toThrow(
      ResponseValidationError,
    );
    document.body.replaceChildren();
    expect(() =>
      addConfig({
        retrievalTopKDefault: "21",
        retrievalTopKMaximum: "20",
      }),
    ).toThrow(ResponseValidationError);
  });

  it("accepts compact index counts above 50,000 without corpus arrays", () => {
    const status = {
      status: "ready",
      index_profile_id: "ipr_1",
      corpus_fingerprint: "cor_1",
      index_format_version: "1",
      embedding_model_name: "sentence-transformers/all-MiniLM-L6-v2",
      embedding_model_revision: "a".repeat(40),
      vector_count: 1,
      indexed_chunk_count: 1,
      indexed_character_count: 20,
      document_count: 60001,
      zero_chunk_document_count: 60000,
      similarity_metric: "cosine",
      index_type: "IndexFlatIP",
      built_at: "2026-01-01T00:00:00Z",
    };

    expect(validateIndexStatus(status)).toBe(status);
    expect(() =>
      validateIndexStatus({
        ...status,
        zero_chunk_document_count: 60002,
      }),
    ).toThrow(ResponseValidationError);
  });

  it("fails the rendered workflow closed when configuration is absent", async () => {
    document.body.dataset.page = "home";
    document.body.innerHTML = `
      <main id="main-content" tabindex="-1">Loading</main>
      <div id="global-status"></div>
    `;
    await initialize();
    expect(document.querySelector("#main-content").textContent).toContain(
      "interface configuration is invalid",
    );
    expect(document.querySelector("#global-status").textContent).toContain(
      "interface configuration is invalid",
    );
  });

  it("accepts exact health and document-list contracts", () => {
    expect(
      validateHealth({ name: "AeroDocs", version: "0.1.0", status: "healthy" }),
    ).toBeTruthy();
    expect(validateDocumentList(documentList(), limits).documents).toHaveLength(
      1,
    );
  });

  it("rejects unknown fields and duplicate document identities", () => {
    expect(() =>
      validateHealth({
        name: "AeroDocs",
        version: "0.1.0",
        status: "healthy",
        extra: "unsafe",
      }),
    ).toThrow(ResponseValidationError);
    const value = documentList();
    value.documents.push({ ...value.documents[0] });
    value.total = 2;
    expect(() => validateDocumentList(value, limits)).toThrow(
      ResponseValidationError,
    );
  });

  it("keeps search ordering and identities strict", () => {
    const result = {
      query: "pump",
      index_profile_id: "ipr_1",
      corpus_fingerprint: "cor_1",
      similarity_metric: "cosine",
      requested_top_k: 2,
      returned_count: 1,
      results: [
        {
          rank: 1,
          score: 0.8,
          document_id: `doc_${"a".repeat(64)}`,
          filename: "manual.pdf",
          title: "Manual",
          chunk_id: "chk_1",
          chunk_order: 1,
          page_start: 1,
          page_end: 1,
          section_heading: null,
          text: "pump",
          character_count: 4,
        },
      ],
    };
    expect(validateSearchResponse(result, "pump", limits)).toBe(result);
    result.results.push({ ...result.results[0], rank: 2 });
    result.returned_count = 2;
    expect(() => validateSearchResponse(result, "pump", limits)).toThrow(
      ResponseValidationError,
    );
  });

  it.each([
    [[], true],
    [[1, 2, 3], true],
    [[2, 1], false],
    [[1, 3], false],
    [[2], false],
    [[1, 1], false],
  ])("enforces contiguous backend rank order: %j", (ranks, accepted) => {
    const result = {
      query: "pump",
      index_profile_id: "ipr_1",
      corpus_fingerprint: "cor_1",
      similarity_metric: "cosine",
      requested_top_k: Math.max(3, ranks.length),
      returned_count: ranks.length,
      results: ranks.map((rank, index) => ({
        rank,
        score: 0.8 - index / 10,
        document_id: `doc_${"a".repeat(64)}`,
        filename: "manual.pdf",
        title: "Manual",
        chunk_id: `chk_${index}`,
        chunk_order: index + 1,
        page_start: 1,
        page_end: 1,
        section_heading: null,
        text: `pump ${index}`,
        character_count: `pump ${index}`.length,
      })),
    };
    const action = () => validateSearchResponse(result, "pump", limits);
    if (accepted) {
      expect(action()).toBe(result);
      expect(result.results.map((item) => item.rank)).toEqual(ranks);
    } else {
      expect(action).toThrow(ResponseValidationError);
    }
  });

  it("uses configured query, chunk, and response limits above old constants", async () => {
    const configured = {
      ...limits,
      queryCharacters: 6000,
      chunkCharacters: 20000,
    };
    const query = "q".repeat(5000);
    const text = "x".repeat(12000);
    const result = {
      query,
      index_profile_id: "ipr_1",
      corpus_fingerprint: "cor_1",
      similarity_metric: "cosine",
      requested_top_k: 1,
      returned_count: 1,
      results: [
        {
          rank: 1,
          score: 0.8,
          document_id: `doc_${"a".repeat(64)}`,
          filename: "manual.pdf",
          title: "Manual",
          chunk_id: "chk_1",
          chunk_order: 1,
          page_start: 1,
          page_end: 1,
          section_heading: null,
          text,
          character_count: text.length,
        },
      ],
    };
    expect(validateSearchResponse(result, query, configured)).toBe(result);
    const large = { value: "x".repeat(1024 * 1024) };
    await expect(
      readBoundedJson(jsonResponse(large), 2 * 1024 * 1024),
    ).resolves.toEqual(large);
    await expect(
      readBoundedJson(jsonResponse(large), 1024),
    ).rejects.toBeInstanceOf(ResponseValidationError);
  });

  it("accepts a structurally consistent supported answer", () => {
    const value = answer();
    expect(
      validateAnswerResponse(value, "How is the pump inspected?", limits),
    ).toBe(value);
  });

  it.each([
    "unknown evidence",
    "wrong citation",
    "wrong quote",
    "wrong page",
    "wrong claim link",
  ])("rejects malformed answer relationship: %s", (_name) => {
    const value = answer();
    if (_name === "unknown evidence") {
      value.claims[0].supports[0].evidence_id = "x";
    } else if (_name === "wrong citation") {
      value.claims[0].supports[0].citation_number = 2;
    } else if (_name === "wrong quote") {
      value.claims[0].supports[0].quote = "fabricated";
    } else if (_name === "wrong page") {
      value.citations[0].page_start = 1;
    } else {
      value.claims[0].supports[0].claim_id = "other";
    }
    expect(() =>
      validateAnswerResponse(value, "How is the pump inspected?", limits),
    ).toThrow(ResponseValidationError);
  });

  it("accepts conflict citations and rejects unrelated conflict evidence", () => {
    const conflict = answer({
      abstained: true,
      abstention_reason: "conflicting_evidence",
      explanation: "The passages conflict.",
      answer: null,
      claims: [],
      citations: [
        citation(),
        citation({
          number: 2,
          evidence_id: "ev_2",
          chunk_id: "chk_2",
          page_start: 3,
          page_end: 3,
        }),
      ],
      evidence: [
        evidence(),
        evidence({
          evidence_id: "ev_2",
          chunk_id: "chk_2",
          page_start: 3,
          page_end: 3,
          text: "Do not inspect the pump visually.",
        }),
        evidence({ evidence_id: "ev_unrelated", chunk_id: "chk_3" }),
      ],
      grounding: grounding({
        claim_count: 0,
        supported_claim_count: 0,
        citation_count: 2,
      }),
    });
    expect(
      validateAnswerResponse(conflict, "How is the pump inspected?", limits),
    ).toBe(conflict);
    conflict.citations[1].evidence_id = "ev_unrelated";
    expect(() =>
      validateAnswerResponse(conflict, "How is the pump inspected?", limits),
    ).toThrow(ResponseValidationError);
  });

  it("rejects unknown answer variants", () => {
    const value = answer({
      abstained: true,
      abstention_reason: "new_reason",
      answer: null,
      explanation: "Unknown",
      claims: [],
      citations: [],
    });
    expect(() =>
      validateAnswerResponse(value, "How is the pump inspected?", limits),
    ).toThrow(ResponseValidationError);
  });

  it("bounds JSON before parsing", async () => {
    await expect(
      readBoundedJson(jsonResponse({ value: "x".repeat(100) }), 20),
    ).rejects.toBeInstanceOf(ResponseValidationError);
  });

  it.each([
    ["application/json", true],
    ["application/json; charset=utf-8", true],
    ["Application/JSON; Charset=UTF-8", true],
    [null, false],
    ["text/html", false],
    ["text/plain", false],
    ["application/octet-stream", false],
    ["application/json;;", false],
    ["application/json charset=utf-8", false],
  ])("validates JSON media type %j", async (mediaType, accepted) => {
    const headers = mediaType === null ? {} : { "content-type": mediaType };
    const response = new Response('{"ok":true}', { headers });
    const action = readBoundedJson(response, 1024);
    if (accepted) {
      await expect(action).resolves.toEqual({ ok: true });
    } else {
      await expect(action).rejects.toMatchObject({
        message: "unexpected_response_media_type",
      });
    }
  });
});

describe("production quote conformance", () => {
  it("matches every shared Python/JavaScript vector", async () => {
    const fixture = JSON.parse(
      await readFile("tests/fixtures/quote_matching.json", "utf8"),
    );
    for (const vector of fixture) {
      expect(
        quoteMatchesEvidence(vector.quote, vector.evidence),
        vector.id,
      ).toBe(vector.expected);
    }
  });
});

describe("request state", () => {
  it("aborts the previous request and suppresses stale tokens", () => {
    const controller = createRequestController();
    const first = controller.begin();
    const second = controller.begin();
    expect(first.signal.aborted).toBe(true);
    expect(controller.current(first.token)).toBe(false);
    expect(controller.current(second.token)).toBe(true);
  });

  it("cancels the active request without treating it as current", () => {
    const controller = createRequestController();
    const request = controller.begin();
    controller.cancel();
    expect(request.signal.aborted).toBe(true);
    expect(controller.current(request.token)).toBe(false);
    expect(controller.active()).toBe(false);
  });
});

describe("paginated document filters", () => {
  function filterPage(offset, item, total = 2) {
    const base = documentList();
    return {
      offset,
      limit: 1,
      total,
      documents: item === null ? [] : [{ ...base.documents[0], ...item }],
    };
  }

  it("preserves and submits selections across bounded pages", async () => {
    const settings = addConfig({
      documentPageSize: "1",
      documentPageMaximum: "2",
    });
    const container = document.createElement("div");
    document.body.append(container);
    globalThis.fetch = vi
      .fn()
      .mockResolvedValueOnce(
        jsonResponse(
          filterPage(0, {
            document_id: `doc_${"a".repeat(64)}`,
            filename: "first.pdf",
          }),
        ),
      )
      .mockResolvedValueOnce(
        jsonResponse(
          filterPage(1, {
            document_id: `doc_${"b".repeat(64)}`,
            filename: "second.pdf",
          }),
        ),
      )
      .mockResolvedValueOnce(
        jsonResponse(
          filterPage(0, {
            document_id: `doc_${"a".repeat(64)}`,
            filename: "first.pdf",
          }),
        ),
      );
    const chooser = createDocumentFilterChooser(container, settings);
    await chooser.ready;
    container.querySelector('input[type="checkbox"]').click();
    container.querySelector(".filter-pagination button:last-child").click();
    await vi.waitFor(() => {
      expect(container.querySelector('input[type="checkbox"]').value).toBe(
        `doc_${"b".repeat(64)}`,
      );
    });
    container.querySelector('input[type="checkbox"]').click();
    expect(chooser.selectedIds()).toEqual([
      `doc_${"a".repeat(64)}`,
      `doc_${"b".repeat(64)}`,
    ]);
    container.querySelector(".filter-pagination button:first-child").click();
    await vi.waitFor(() => {
      expect(container.querySelector('input[type="checkbox"]').checked).toBe(
        true,
      );
    });
    const selectedButtons = [
      ...container.querySelectorAll(".selected-documents button"),
    ];
    const removeSecond = selectedButtons[selectedButtons.length - 1];
    removeSecond.click();
    expect(chooser.selectedIds()).toEqual([`doc_${"a".repeat(64)}`]);
    expect(
      globalThis.fetch.mock.calls.every(([url]) =>
        String(url).includes("limit=1"),
      ),
    ).toBe(true);
  });

  it("includes a later-page Ask selection only in the JSON body", async () => {
    answerPage();
    Object.assign(document.querySelector("#ui-config").dataset, {
      documentPageSize: "1",
      documentPageMaximum: "2",
    });
    const firstId = `doc_${"a".repeat(64)}`;
    const secondId = `doc_${"b".repeat(64)}`;
    const first = filterPage(0, {
      document_id: firstId,
      filename: "first.pdf",
    });
    const second = filterPage(1, {
      document_id: secondId,
      filename: "second.pdf",
    });
    globalThis.fetch = vi
      .fn()
      .mockResolvedValueOnce(jsonResponse(first))
      .mockResolvedValueOnce(jsonResponse(second))
      .mockResolvedValueOnce(jsonResponse(answer()));
    await initialize();
    document
      .querySelector("#answer-documents .filter-pagination button:last-child")
      .click();
    await vi.waitFor(() => {
      expect(document.querySelector("#answer-documents input").value).toBe(
        secondId,
      );
    });
    document.querySelector("#answer-documents input").click();
    const form = document.querySelector("#answer-form");
    form.elements.question.value = "How is the pump inspected?";
    form.dispatchEvent(
      new Event("submit", { bubbles: true, cancelable: true }),
    );
    await vi.waitFor(() => {
      expect(globalThis.fetch).toHaveBeenCalledTimes(3);
    });
    const [url, options] = globalThis.fetch.mock.calls[2];
    expect(url).toBe("/answers");
    expect(JSON.parse(options.body).document_ids).toEqual([secondId]);
    expect(String(url)).not.toContain(secondId);
  });

  it("includes a page-two Search selection only in the JSON body", async () => {
    searchPage();
    const firstId = `doc_${"a".repeat(64)}`;
    const secondId = `doc_${"b".repeat(64)}`;
    globalThis.fetch = vi
      .fn()
      .mockResolvedValueOnce(
        jsonResponse(
          filterPage(0, {
            document_id: firstId,
            filename: "first.pdf",
          }),
        ),
      )
      .mockResolvedValueOnce(
        jsonResponse(
          filterPage(1, {
            document_id: secondId,
            filename: "second.pdf",
          }),
        ),
      )
      .mockResolvedValueOnce(
        jsonResponse({
          query: "pump",
          index_profile_id: "ipr_1",
          corpus_fingerprint: "cor_1",
          similarity_metric: "cosine",
          requested_top_k: 1,
          returned_count: 0,
          results: [],
        }),
      );
    await initialize();
    document
      .querySelector("#search-documents .filter-pagination button:last-child")
      .click();
    await vi.waitFor(() => {
      expect(document.querySelector("#search-documents input").value).toBe(
        secondId,
      );
    });
    document.querySelector("#search-documents input").click();
    const form = document.querySelector("#search-form");
    form.elements.query.value = "pump";
    form.dispatchEvent(
      new Event("submit", { bubbles: true, cancelable: true }),
    );
    await vi.waitFor(() => {
      expect(globalThis.fetch).toHaveBeenCalledTimes(3);
    });
    const [url, options] = globalThis.fetch.mock.calls[2];
    expect(url).toBe("/search");
    expect(JSON.parse(options.body).document_ids).toEqual([secondId]);
    expect(String(url)).not.toContain(secondId);
  });

  it("fails closed on a duplicate identity across pages", async () => {
    const settings = addConfig({
      documentPageSize: "1",
      documentPageMaximum: "2",
    });
    const container = document.createElement("div");
    document.body.append(container);
    const same = { document_id: `doc_${"a".repeat(64)}` };
    globalThis.fetch = vi
      .fn()
      .mockResolvedValueOnce(jsonResponse(filterPage(0, same)))
      .mockResolvedValueOnce(jsonResponse(filterPage(1, same)));
    const chooser = createDocumentFilterChooser(container, settings);
    await chooser.ready;
    await chooser.loadOffset(1);
    expect(container.textContent).toContain(
      "presentation-integrity validation",
    );
  });
});

describe("document workflow concurrency", () => {
  it("renders only the latest requested document page", async () => {
    documentsPage();
    Object.assign(document.querySelector("#ui-config").dataset, {
      documentPageSize: "1",
      documentPageMaximum: "2",
    });
    const firstId = `doc_${"a".repeat(64)}`;
    const secondId = `doc_${"b".repeat(64)}`;
    const firstPage = {
      ...documentList(),
      limit: 1,
      total: 2,
      documents: [
        {
          ...documentList().documents[0],
          document_id: firstId,
          filename: "first.pdf",
          title: "First",
        },
      ],
    };
    const secondPage = {
      ...firstPage,
      offset: 1,
      documents: [
        {
          ...firstPage.documents[0],
          document_id: secondId,
          filename: "second.pdf",
          title: "Second",
        },
      ],
    };
    let resolveOlder;
    let resolveLatest;
    const older = new Promise((resolve) => {
      resolveOlder = resolve;
    });
    const latest = new Promise((resolve) => {
      resolveLatest = resolve;
    });
    globalThis.fetch = vi
      .fn()
      .mockResolvedValueOnce(jsonResponse(firstPage))
      .mockReturnValueOnce(older)
      .mockReturnValueOnce(latest);
    await initialize();
    document.querySelector("#documents-next").dispatchEvent(new Event("click"));
    document
      .querySelector("#documents-previous")
      .dispatchEvent(new Event("click"));
    resolveLatest(jsonResponse(firstPage));
    await vi.waitFor(() => {
      expect(document.querySelector("#document-list").textContent).toContain(
        "First",
      );
    });
    resolveOlder(jsonResponse(secondPage));
    await Promise.resolve();
    expect(document.querySelector("#document-list").textContent).toContain(
      "First",
    );
    expect(document.querySelector("#document-list").textContent).not.toContain(
      "Second",
    );
  });

  it("prevents stale metadata from replacing the latest document", async () => {
    documentsPage();
    const firstId = `doc_${"a".repeat(64)}`;
    const secondId = `doc_${"b".repeat(64)}`;
    const listed = documentList();
    listed.total = 2;
    listed.documents.push({
      ...listed.documents[0],
      document_id: secondId,
      filename: "second.pdf",
      title: "Second",
    });
    let resolveFirst;
    let resolveSecond;
    const firstPending = new Promise((resolve) => {
      resolveFirst = resolve;
    });
    const secondPending = new Promise((resolve) => {
      resolveSecond = resolve;
    });
    globalThis.fetch = vi
      .fn()
      .mockResolvedValueOnce(jsonResponse(listed))
      .mockReturnValueOnce(firstPending)
      .mockReturnValueOnce(secondPending);
    await initialize();
    const buttons = [...document.querySelectorAll(".document-card button")];
    buttons[0].click();
    buttons[1].click();
    resolveSecond(jsonResponse(documentMetadata(secondId, "second.pdf")));
    await vi.waitFor(() => {
      expect(document.querySelector("#metadata-content").textContent).toContain(
        "second.pdf",
      );
    });
    resolveFirst(jsonResponse(documentMetadata(firstId, "manual.pdf")));
    await Promise.resolve();
    expect(document.querySelector("#metadata-content").textContent).toContain(
      "second.pdf",
    );
    expect(
      document.querySelector("#metadata-content").textContent,
    ).not.toContain("manual.pdf");
  });

  it("invalidates metadata when the user closes it", async () => {
    documentsPage();
    let resolveMetadata;
    const pending = new Promise((resolve) => {
      resolveMetadata = resolve;
    });
    globalThis.fetch = vi
      .fn()
      .mockResolvedValueOnce(jsonResponse(documentList()))
      .mockReturnValueOnce(pending);
    await initialize();
    document.querySelector(".document-card button").click();
    document.querySelector("#metadata-close").click();
    resolveMetadata(
      jsonResponse(documentMetadata(`doc_${"a".repeat(64)}`, "manual.pdf")),
    );
    await Promise.resolve();
    expect(document.querySelector("#document-metadata").hidden).toBe(true);
    expect(document.querySelector("#metadata-content").textContent).toBe("");
  });
});

describe("dynamic answer presentation", () => {
  it("renders hostile text as text and preserves citation navigation", async () => {
    answerPage();
    const hostile = answer();
    hostile.claims[0].text = "<script>alert(1)</script>";
    hostile.answer = "<script>alert(1)</script> [1]";
    globalThis.fetch = vi
      .fn()
      .mockResolvedValueOnce(jsonResponse(documentList()))
      .mockResolvedValueOnce(jsonResponse(hostile));
    await initialize();
    const form = document.querySelector("#answer-form");
    form.elements.question.value = "How is the pump inspected?";
    form.dispatchEvent(
      new Event("submit", { bubbles: true, cancelable: true }),
    );
    await vi.waitFor(() => {
      expect(document.querySelector("#answer-output").hidden).toBe(false);
    });
    expect(document.querySelector("script")).toBeNull();
    expect(document.body.textContent).toContain("<script>alert(1)</script>");
    expect(
      document.querySelector('.claim-supports a[href="#evidence-1"]'),
    ).not.toBeNull();
    expect(document.activeElement.id).toBe("answer-results-heading");
    expect(document.querySelector("#answer-status").textContent).toContain(
      "Grounded answer returned",
    );
  });

  it("prevents double submission while a request is active", async () => {
    answerPage();
    let resolveAnswer;
    const pending = new Promise((resolve) => {
      resolveAnswer = resolve;
    });
    globalThis.fetch = vi
      .fn()
      .mockResolvedValueOnce(jsonResponse(documentList()))
      .mockReturnValueOnce(pending);
    await initialize();
    const form = document.querySelector("#answer-form");
    form.elements.question.value = "How is the pump inspected?";
    form.dispatchEvent(
      new Event("submit", { bubbles: true, cancelable: true }),
    );
    form.dispatchEvent(
      new Event("submit", { bubbles: true, cancelable: true }),
    );
    expect(globalThis.fetch).toHaveBeenCalledTimes(2);
    resolveAnswer(jsonResponse(answer()));
    await vi.waitFor(() => {
      expect(document.querySelector("#answer-submit").disabled).toBe(false);
    });
  });

  it("cancellation prevents a late answer from rendering", async () => {
    answerPage();
    let resolveAnswer;
    const pending = new Promise((resolve) => {
      resolveAnswer = resolve;
    });
    globalThis.fetch = vi
      .fn()
      .mockResolvedValueOnce(jsonResponse(documentList()))
      .mockReturnValueOnce(pending);
    await initialize();
    const form = document.querySelector("#answer-form");
    form.elements.question.value = "How is the pump inspected?";
    form.dispatchEvent(
      new Event("submit", { bubbles: true, cancelable: true }),
    );
    document.querySelector("#answer-cancel").click();
    resolveAnswer(jsonResponse(answer()));
    await Promise.resolve();
    await Promise.resolve();
    expect(document.querySelector("#answer-output").hidden).toBe(true);
    expect(document.querySelector("#answer-status").textContent).toContain(
      "audit persistence may continue",
    );
  });

  it("copies only visible citation provenance after an explicit action", () => {
    const value = citation();
    const copied = citationCopyText(value);
    expect(copied).toContain("Public Manual");
    expect(copied).toContain("chunk chk_1");
    expect(copied).not.toContain("aud_");
    expect(copied).not.toContain("Inspect the pump");
  });
});
