# Simple Dockerfile for Stock AI Dashboard
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python packages
COPY requirements.txt requirements_frontend.txt ./
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN pip install -r requirements_frontend.txt

# Copy application code
COPY . .

# Create directories for data
RUN mkdir -p data models logs

# Expose dashboard port
EXPOSE 8050

# Run the dashboard
CMD ["python", "run_dashboard.py"]