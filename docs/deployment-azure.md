# Container and Azure deployment notes

> **Status: review deployment, not a production or pilot deployment.**
> Phase 8A implements direct-bearer OIDC only. There is no browser sign-in,
> readiness endpoint, rate limiter or Origin policy (see the README), so put the
> container behind a private network or an authenticating gateway and give it
> data that is public, synthetic or explicitly authorised. The image has not yet
> been built in CI on a real registry; treat the first build as a test.

## Image

```
docker build -t aerodocs .
```

- **CPU only.** `uv.lock` resolves the PyPI `torch` wheel, which drags in CUDA
  packages. The Dockerfile installs every other locked dependency from the
  lockfile, then installs the same `torch` version from the PyTorch CPU index.
- **Fails closed.** The default command is `app.secure_main:app`. Importing it
  raises unless complete, non-placeholder OIDC settings are present.
- **One worker.** Inference is serialised per process and the PyTorch thread
  pool is process-global, so scale up the machine, not the worker count.
- **State lives in `/data`** (documents, indexes, audit records, model cache).
  Mount persistent storage there or everything is lost on restart.
- **Models download on first use** (pinned MiniLM and Qwen revisions). Either
  allow outbound access to the Hugging Face host for the first requests or
  pre-populate `/data/hf-cache`.

### Local run without authentication (localhost only)

```
docker run --rm -p 127.0.0.1:8000:8000 -v aerodocs-data:/data \
  -e AERODOCS_SECURITY_MODE=disabled \
  aerodocs uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Bind the published port to `127.0.0.1` as shown. Never publish an
unauthenticated instance.

### Local run with OIDC

```
docker run --rm -p 127.0.0.1:8000:8000 -v aerodocs-data:/data \
  -e AERODOCS_OIDC_ISSUER=https://<issuer>/ \
  -e AERODOCS_OIDC_AUDIENCE=<api-audience> \
  -e AERODOCS_OIDC_JWKS_URL=https://<issuer-host>/<jwks-path> \
  -e AERODOCS_OIDC_VIEWER_GROUP=<viewer-group-value> \
  -e AERODOCS_OIDC_OPERATOR_GROUP=<operator-group-value> \
  aerodocs
```

The validator accepts only RS256 access tokens that carry `iss`, `aud`, `sub`,
`iat`, `nbf`, `exp` and a bounded `groups` claim, with a lifetime of at most
15 minutes. The JWKS URL must be on the same HTTPS origin as the issuer, and
placeholder values are rejected.

## Azure Container Apps (suggested path)

1. Create an Azure Container Registry and push the image
   (`az acr build --registry <acr> --image aerodocs:<tag> .`).
2. Create a Container Apps environment and an Azure Files share; mount the share
   at `/data` in the app.
3. Create the container app from the image with:
   - **min and max replicas of 1.** Document and index state is local to the
     replica's filesystem and builds take an OS-level lock, so do not scale out.
   - at least 2 vCPU and 8 GiB memory (FAISS holds every vector in memory and the
     two models need additional RAM; size from your corpus).
   - the five `AERODOCS_OIDC_*` settings above as plain environment variables.
     They are identifiers, not secrets. The application needs no secrets.
   - ingress **internal** until authentication has been verified end to end.
4. Check `GET /health` first. All other routes should return 401 without a valid
   bearer token.

Azure App Service for Containers also works: set `WEBSITES_PORT=8000`, enable
persistent storage mounted at `/data`, and keep a single instance.

### Identity provider caveats (verify before relying on it)

The validator's contract is strict, so confirm your provider can meet it:

- **Token lifetime.** The configured maximum is 15 minutes. Microsoft Entra ID's
  default access-token lifetime is typically longer, so a token lifetime policy
  would be needed. Check against your tenant.
- **`groups` claim.** It is required. Entra emits group object IDs (not names)
  when enabled and omits the claim in "overage" cases with many groups, which
  would make valid users fail closed.
- **Same-origin JWKS.** Issuer and JWKS URL must share an origin.

## CI

`.github/workflows/ci.yml` runs Ruff, formatting, mypy and pytest, the UI
`npm run check`, and builds the image (without pushing), then asserts that the
secure entry point refuses to start without OIDC settings.
