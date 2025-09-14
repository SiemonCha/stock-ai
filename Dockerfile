# Multi-stage Docker build for Stock AI Enterprise System

# Build stage
FROM python:3.10-slim as builder

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    pkg-config \
    libhdf5-dev \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements and install dependencies
COPY requirements.txt requirements_frontend.txt ./
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN pip install -r requirements_frontend.txt

# Production stage
FROM python:3.10-slim as production

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    ENVIRONMENT=production \
    REDIS_URL=redis://redis:6379 \
    API_HOST=0.0.0.0 \
    API_PORT=8000

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    redis-tools \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user
RUN groupadd -r stockai && useradd -r -g stockai -s /bin/bash stockai

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv

# Create application directory
WORKDIR /app

# Copy application code
COPY src/ ./src/
COPY api/ ./api/
COPY *.py ./
COPY run_dashboard.py ./

# Create necessary directories
RUN mkdir -p models/saved data/cache plots logs config && \
    chown -R stockai:stockai /app

# Switch to non-root user
USER stockai

# Health check for API
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:$API_PORT/health || exit 1

# Expose ports (API and Dashboard)
EXPOSE 8000 8050

# Default command runs dashboard
CMD ["python", "run_dashboard.py"]