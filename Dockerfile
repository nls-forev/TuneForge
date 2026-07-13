# CUDA 12.6 matches torch 2.7 (cu126), pinned via the unsloth version ceiling.
FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04

# uv brings its own standalone CPython (requires-python = 3.11), so we don't
# apt-install python. git + libgomp1 are needed by triton/bitsandbytes/unsloth.
RUN apt-get update \
    && apt-get install -y --no-install-recommends git libgomp1 ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy UV to docker
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

WORKDIR /app

# Deps first for layer caching. --frozen = fail if uv.lock is stale (run
# `uv lock` after editing pyproject). dev group kept: `dvc` lives there and the
# CMD needs it; pre-commit/detect-secrets are tiny pure-python.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --extra gpu

COPY src ./src/
COPY config ./config/

# from_root anchors on .git or .project-root. .git is dockerignored, so give
# it an explicit anchor at /app.
RUN touch .project-root

# Runs the full DVC pipeline (ingest -> transform -> train) on GPU.
CMD ["uv", "run", "dvc", "repro"]
