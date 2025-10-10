# Multi-stage BentoML service Dockerfile for Kubernetes deployment
# Stage 1: Development/Build stage
FROM nvidia/cuda:12.6.0-runtime-ubuntu22.04 AS builder

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    BENTOML_HOME=/bentoml \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies for building
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-dev \
    python3-pip \
    build-essential \
    gcc \
    g++ \
    make \
    pkg-config \
    git \
    curl \
    wget \
    ffmpeg \
    libsndfile1 \
    espeak-ng \
    espeak-ng-data \
    libespeak-ng1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && ln -s /usr/bin/python3 /usr/bin/python

# Upgrade pip
RUN python3 -m pip install --upgrade pip

# Install BentoML and build dependencies
RUN pip install bentoml

# Create non-root user
RUN useradd -m -u 1001 bentoml && \
    mkdir -p /bentoml && \
    chown -R bentoml:bentoml /bentoml

# Set working directory
WORKDIR /app

# Copy project files
COPY . /app/

# Change ownership to bentoml user
RUN chown -R bentoml:bentoml /app

# Switch to non-root user
USER bentoml

# Install Python dependencies with error handling
RUN pip install --no-cache-dir -r requirements.txt || \
    (pip install --no-cache-dir --ignore-installed mojimoji && \
     pip install --no-cache-dir -r requirements.txt)

# Build the BentoML service
RUN bentoml build .

# Stage 2: Runtime stage
FROM nvidia/cuda:12.6.0-runtime-ubuntu22.04 AS runtime

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=3000 \
    BENTOML_HOME=/bentoml \
    DEBIAN_FRONTEND=noninteractive

# Install minimal runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    curl \
    ffmpeg \
    libsndfile1 \
    espeak-ng \
    espeak-ng-data \
    libespeak-ng1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && ln -s /usr/bin/python3 /usr/bin/python

# Upgrade pip
RUN python3 -m pip install --upgrade pip

# Install BentoML runtime
RUN pip install bentoml

# Create non-root user
RUN useradd -m -u 1001 bentoml && \
    mkdir -p /bentoml && \
    chown -R bentoml:bentoml /bentoml

# Set working directory
WORKDIR /app

# Copy the built BentoML service from builder stage
COPY --from=builder --chown=bentoml:bentoml /bentoml /bentoml

# Switch to non-root user
USER bentoml

# Expose port
EXPOSE 3000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:3000/healthz || exit 1

# Run the BentoML service
CMD ["bentoml", "serve", "acesteps_service:latest", "--host", "0.0.0.0", "--port", "3000"]
