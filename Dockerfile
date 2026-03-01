# =============================================================
# Dockerfile – GPU Orchestrator (multi-stage build)
# =============================================================
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies first (for layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# =============================================================
FROM python:3.11-slim AS runtime

LABEL maintainer="GPU Orchestration Team"
LABEL description="Adaptive P2P GPU Orchestration – Orchestrator"

WORKDIR /app

# Non-root user for security
RUN useradd --no-create-home --shell /bin/false appuser

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application source
COPY auth/          auth/
COPY database/      database/
COPY orchestrator/  orchestrator/
COPY scheduler/     scheduler/
COPY utils/         utils/
COPY frontend/      frontend/
COPY config.yaml    .

# Create directories writable by appuser
RUN mkdir -p logs checkpoints data && chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "orchestrator.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "2", \
     "--log-level", "warning"]
