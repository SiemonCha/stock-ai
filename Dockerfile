# Simple Dockerfile for Stock AI Dashboard
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python packages
COPY requirements.txt ./
RUN pip install --upgrade pip

# Install dependencies with relaxed constraints for Docker builds
RUN pip install tensorflow pandas numpy scikit-learn || true && \
    pip install torch transformers xgboost lightgbm catboost || true && \
    pip install yfinance alpha_vantage requests websocket-client || true && \
    pip install ta scipy statsmodels || true && \
    pip install fastapi uvicorn pydantic websockets || true && \
    pip install dash plotly || true && \
    pip install redis sqlalchemy psycopg2-binary || true && \
    pip install python-dotenv joblib tqdm || true && \
    pip install pytest pytest-cov || true

# Copy application code
COPY . .

# Create directories for data
RUN mkdir -p data models logs

# Expose dashboard port
EXPOSE 8050

# Run the dashboard
CMD ["python", "run_dashboard.py"]