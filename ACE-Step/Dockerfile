# ================================
# Stage 2: Runtime
# ================================
FROM nvidia/cuda:12.6.0-runtime-ubuntu22.04 AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=3000 \
    BENTOML_HOME=/bentoml \
    DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv curl ffmpeg libsndfile1 espeak-ng espeak-ng-data libespeak-ng1 \
    && apt-get clean && rm -rf /var/lib/apt/lists/* \
    && ln -s /usr/bin/python3 /usr/bin/python

RUN python3 -m pip install --upgrade pip bentoml

RUN useradd -m -u 1001 bentoml && mkdir -p /bentoml && chown -R bentoml:bentoml /bentoml
USER bentoml

# Create and activate virtual environment
RUN python -m venv /bentoml/venv
ENV PATH="/bentoml/venv/bin:$PATH"

# Copy requirements.txt and install dependencies
COPY requirements.txt .

# Install specific PyTorch version compatible with CUDA 12.6
RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir hf_transfer peft && \
    pip3 install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu126

WORKDIR /app
COPY acesteps_service-latest.bento /app/acesteps_service-latest.bento
RUN bentoml import /app/acesteps_service-latest.bento

EXPOSE 3000
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:3000/v1/healthz || exit 1

CMD ["sh", "-c", "bentoml serve acesteps_service --host 0.0.0.0 --port 3000"]