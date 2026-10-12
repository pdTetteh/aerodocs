# syntax=docker/dockerfile:1
#
# AeroDocs container image (CPU only).
#
# The default command starts the OIDC-only Phase 8A application and refuses to
# start without complete OIDC settings, so the image fails closed. Run the
# unauthenticated development app only on localhost (see docs/deployment-azure.md).

FROM python:3.11-slim AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1

RUN pip install --no-cache-dir uv

WORKDIR /build
COPY pyproject.toml uv.lock ./

# uv.lock resolves the PyPI torch wheel, which pulls gigabytes of CUDA packages
# that a CPU-only deployment never uses. Install every other locked dependency
# from the lockfile, then install torch from the CPU-only index at the same
# pinned version.
RUN uv export --frozen --no-dev --no-hashes --no-emit-project \
        | grep -viE '^(torch==|nvidia-|triton)' > /tmp/requirements.txt \
    && TORCH_VERSION="$(uv export --frozen --no-dev --no-hashes --no-emit-project \
        | sed -n 's/^torch==\([0-9.]*\).*/\1/p')" \
    && uv pip install --system --no-deps -r /tmp/requirements.txt \
    && uv pip install --system --no-deps "torch==${TORCH_VERSION}" \
        --index-url https://download.pytorch.org/whl/cpu


FROM python:3.11-slim AS runtime

# libgomp1: OpenMP runtime used by FAISS and PyTorch CPU kernels.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --uid 10001 aerodocs \
    && mkdir -p /data \
    && chown aerodocs:aerodocs /data

COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

WORKDIR /app
COPY app ./app

# All mutable state lives under /data so one volume holds documents, indexes,
# audit records and the model cache. Mount persistent storage there.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/data/hf-cache \
    AERODOCS_DOCUMENT_STORAGE_ROOT=/data/runtime \
    AERODOCS_INDEX_RUNTIME_ROOT=/data/runtime/indexes \
    AERODOCS_GENERATION_AUDIT_ROOT=/data/runtime/generation/audit \
    AERODOCS_WORK_ORDER_ROOT=/data/runtime/work_orders \
    AERODOCS_SECURITY_MODE=oidc

USER aerodocs
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)"]

# One worker: the generation provider serialises inference per process and the
# PyTorch thread pool is process-global.
CMD ["uvicorn", "app.secure_main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
