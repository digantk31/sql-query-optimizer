# ── Build stage ────────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ── Runtime stage ──────────────────────────────────────────────────────────
FROM python:3.11-slim

# HF Spaces runs as uid 1000
RUN useradd -m -u 1000 appuser 2>/dev/null || true

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application source
COPY models.py tasks.py db.py env.py app.py inference.py openenv.yaml ./

# Copy static UI assets
COPY static/ ./static/

# Non-root user for security
USER appuser

# HF Spaces expects port 7860
EXPOSE 7860

# Health check — the validator pings /health
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/health')"

CMD ["python", "-m", "uvicorn", "app:app", \
     "--host", "0.0.0.0", \
     "--port", "7860", \
     "--log-level", "info", \
     "--workers", "1"]
