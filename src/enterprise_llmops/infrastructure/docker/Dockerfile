# Enterprise LLMOps Platform - Main Application Dockerfile
FROM python:3.9-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    libpq-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install Ollama
RUN curl -fsSL https://ollama.ai/install.sh | sh

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/data/model_registry \
    /app/models/cache \
    /app/models/custom \
    /app/logs \
    /app/config

# Set permissions
RUN chmod +x /app/scripts/*.sh

# Expose ports
EXPOSE 8080 11434

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Start command
CMD ["python", "-m", "src.enterprise_llmops.frontend.fastapi_app"]
