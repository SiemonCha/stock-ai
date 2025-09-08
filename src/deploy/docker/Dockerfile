# Stock AI Production Docker Image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install additional production dependencies
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn[standard] \
    redis \
    cachetools \
    psutil \
    python-multipart

# Copy source code
COPY src/ src/
COPY api_server.py .
COPY *.py .

# Create necessary directories
RUN mkdir -p models/saved plots logs

# Set environment variables
ENV PYTHONPATH=/app/src
ENV API_HOST=0.0.0.0
ENV API_PORT=8000
ENV API_WORKERS=4

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the API
CMD ["python", "api_server.py"]