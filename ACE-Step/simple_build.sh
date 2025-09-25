#!/bin/bash

# Simple build script for ACE-Step Docker image
set -e

echo "🚀 Building ACE-Step Docker image..."

# Get the latest bento
LATEST_BENTO=$(bentoml list | head -n 2 | tail -n 1 | awk '{print $1":"$2}')
echo "📦 Using Bento: $LATEST_BENTO"

# Create a simple Dockerfile
cat > Dockerfile.simple << 'EOF'
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=3000 \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    TOKENIZERS_PARALLELISM=false \
    CUDA_VISIBLE_DEVICES=0 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    python3-venv \
    python3-dev \
    build-essential \
    git \
    curl \
    wget \
    ffmpeg \
    sox \
    libsndfile1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && ln -s /usr/bin/python3 /usr/bin/python

# Create a non-root user
RUN useradd -m -u 1001 appuser

# Set working directory
WORKDIR /app

# Copy bento files
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements-bentoml.txt

# Create necessary directories
RUN mkdir -p /tmp/ace_step_outputs /tmp/huggingface_cache && \
    chown -R appuser:appuser /tmp/ace_step_outputs /tmp/huggingface_cache

# Change ownership of app files to appuser
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose the port
EXPOSE 3000

# Set healthcheck
HEALTHCHECK --interval=60s --timeout=10s --start-period=5s --retries=5 \
  CMD curl -f http://localhost:3000/health || exit 1

# Default command
CMD ["python", "-m", "bentoml", "serve", "service:acestepaudoservice", "--host", "0.0.0.0", "--port", "3000"]
EOF

# Build Docker image directly from current directory
echo "🐳 Building Docker image..."
docker build -f Dockerfile.simple -t ace-step-audio:latest .

echo "✅ Docker image built successfully!"
echo "📋 Image details:"
docker images | grep ace-step-audio

echo ""
echo "🎉 Build process completed!"
echo ""
echo "To run the container:"
echo "  docker run --gpus all -p 3000:3000 ace-step-audio:latest"
echo ""
echo "To test the service:"
echo "  curl http://localhost:3000/health"
echo ""
echo "To access the API documentation:"
echo "  http://localhost:3000/docs"

# Cleanup
rm -f Dockerfile.simple
