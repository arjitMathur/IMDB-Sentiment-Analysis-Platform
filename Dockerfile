# ============================================================
# Multi-stage Dockerfile for Sentiment Analysis Platform
# ============================================================
# Builds a lightweight image with API + Streamlit.
# Model artifacts should be mounted via volume.
#
# Build:  docker build -t sentiment-platform .
# Run:    docker-compose up
# ============================================================

# Stage 1: Builder — install Python deps
FROM python:3.11-slim AS builder

WORKDIR /build

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# Stage 2: Runtime — copy only what's needed
FROM python:3.11-slim AS runtime

# Security: non-root user
RUN groupadd -r appuser && useradd -r -g appuser -s /bin/false appuser

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application code
COPY src/ src/
COPY api/ api/
COPY streamlit_app/ streamlit_app/
COPY .env.example .env.example

# Create models directory (artifacts mounted as volume)
RUN mkdir -p models/baseline models/lstm models/transformer_onnx \
    && chown -R appuser:appuser /app

USER appuser

# Expose ports for API and Streamlit
EXPOSE 8000 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import httpx; r = httpx.get('http://localhost:8000/api/v1/health'); r.raise_for_status()" || exit 1

# Default: run the API (override in docker-compose for Streamlit)
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
