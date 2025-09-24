# ACE-Steps BentoML Service

A production-ready BentoML service for ACE-Steps audio generation, providing scalable API endpoints for text-to-music and audio-to-audio generation.

## Features

- **Text-to-Music Generation**: Convert text prompts to music
- **Audio-to-Audio Processing**: Transform existing audio with various parameters
- **Multi-language Support**: 17+ supported languages including English, Chinese, Japanese, Korean
- **Scalable Architecture**: Built on BentoML for production deployment
- **Resource Management**: GPU memory optimization and CPU offloading
- **Monitoring**: Built-in health checks and metrics

## Quick Start

### 1. Installation

```bash
# Install BentoML
pip install bentoml

# Install service dependencies
pip install -r requirements_bentoml.txt
```

### 2. Run Locally

```bash
# Start the service
bentoml serve bentoml_service.service:ACEStepBentoService

# The service will be available at http://localhost:3000
```

### 3. API Usage

#### Generate Music from Text

```python
import requests

# Generate music from text prompt
response = requests.post("http://localhost:3000/music", json={
    "prompt": "A cheerful pop song with upbeat tempo",
    "duration": 30,
    "infer_steps": 60,
    "guidance_scale": 15.0,
    "seed": 42
})

result = response.json()
print(f"Generated audio: {result['audio_path']}")
```

#### Generate Audio with Lyrics

```python
# Generate audio with lyrics
response = requests.post("http://localhost:3000/generate", json={
    "audio_duration": 30.0,
    "prompt": "A romantic ballad",
    "lyrics": "In the moonlight, we dance together...",
    "infer_step": 60,
    "guidance_scale": 15.0,
    "scheduler_type": "euler",
    "cfg_type": "classifier_free",
    "omega_scale": 10.0,
    "actual_seeds": [42, 43, 44],
    "output_format": "wav"
})

result = response.json()
print(f"Generated audio: {result['output_path']}")
```

## API Endpoints

### POST `/generate`
Generate audio with full parameter control.

**Request Body:**
```json
{
    "audio_duration": 30.0,
    "prompt": "A cheerful song",
    "lyrics": "Optional lyrics",
    "infer_step": 60,
    "guidance_scale": 15.0,
    "scheduler_type": "euler",
    "cfg_type": "classifier_free",
    "omega_scale": 10.0,
    "actual_seeds": [42],
    "guidance_interval": 0.1,
    "guidance_interval_decay": 0.95,
    "min_guidance_scale": 1.0,
    "use_erg_tag": false,
    "use_erg_lyric": false,
    "use_erg_diffusion": false,
    "oss_steps": [],
    "guidance_scale_text": 0.0,
    "guidance_scale_lyric": 0.0,
    "output_format": "wav"
}
```

**Response:**
```json
{
    "status": "success",
    "output_path": "/tmp/ace_step_outputs/generated_abc123.wav",
    "generation_time": 45.2,
    "metadata": {
        "generation_time": 45.2,
        "parameters": {...},
        "output_format": "wav",
        "file_size": 1024000
    },
    "error_message": null
}
```

### POST `/music`
Generate music from text prompt (simplified interface).

**Request Body:**
```json
{
    "prompt": "A cheerful pop song",
    "duration": 30,
    "infer_steps": 60,
    "guidance_scale": 15.0,
    "omega_scale": 10.0,
    "seed": 42
}
```

### GET `/health`
Check service health and status.

**Response:**
```json
{
    "status": "healthy",
    "model_loaded": true,
    "gpu_available": true,
    "gpu_count": 1,
    "output_directory_accessible": true,
    "service_uptime": 3600.5,
    "error_message": null
}
```

### POST `/download`
Download generated audio file.

**Request Body:**
```json
{
    "file_path": "/tmp/ace_step_outputs/generated_abc123.wav"
}
```

## Configuration

### Environment Variables

```bash
# Model configuration
export ACE_STEP_CHECKPOINT_PATH="ACE-Step/ACE-Step-v1-3.5B"
export ACE_STEP_DEVICE="cuda"
export ACE_STEP_DTYPE="bfloat16"
export ACE_STEP_DEVICE_ID="0"

# Resource configuration
export MAX_CONCURRENT_REQUESTS="2"
export REQUEST_TIMEOUT="300"
export GPU_MEMORY_LIMIT="8Gi"

# Output configuration
export OUTPUT_DIR="/tmp/ace_step_outputs"
export MAX_OUTPUT_SIZE="104857600"  # 100MB

# Performance configuration
export ENABLE_TORCH_COMPILE="false"
export ENABLE_CPU_OFFLOAD="false"
export ENABLE_OVERLAPPED_DECODE="false"
```

### BentoML Configuration

The service is configured via `bentofile.yaml`:

```yaml
service: "bentoml_service.service:ACEStepBentoService"
include:
  - "bentoml_service/**/*.py"
  - "acestep/**/*.py"
  - "examples/**/*.json"
python:
  requirements_txt: "./requirements_bentoml.txt"
docker:
  distro: "debian"
  python_version: "3.10"
  cuda_version: "12.1"
  system_packages:
    - "ffmpeg"
    - "libsndfile1"
```

## Deployment

### 1. Local Development

```bash
# Install dependencies
pip install -r requirements_bentoml.txt

# Run service
bentoml serve bentoml_service.service:ACEStepBentoService
```

### 2. Docker Deployment

```bash
# Build BentoML service
bentoml build .

# Run with Docker
bentoml containerize ace-step-audio-service:latest
docker run --gpus all -p 3000:3000 ace-step-audio-service:latest
```

### 3. BentoCloud Deployment

```bash
# Login to BentoCloud
bentoml cloud login

# Deploy to BentoCloud
bentoml deploy .
```

## Monitoring

### Health Checks

The service provides comprehensive health checks:

- Model loading status
- GPU availability and utilization
- Output directory accessibility
- Service uptime

### Metrics

Built-in Prometheus metrics for:
- Request latency and throughput
- GPU/CPU utilization
- Memory usage
- Error rates

## Performance Optimization

### Resource Management

- **GPU Memory**: Automatic memory management and cleanup
- **CPU Offloading**: Optional CPU offloading for memory-constrained environments
- **Concurrent Requests**: Configurable request concurrency limits
- **Request Timeout**: Configurable timeout for long-running generations

### Caching

- Model loading is cached after first initialization
- Output files are managed with automatic cleanup
- Temporary files are cleaned up based on age

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce `MAX_CONCURRENT_REQUESTS`
   - Enable `ENABLE_CPU_OFFLOAD`
   - Reduce `GPU_MEMORY_LIMIT`

2. **Model Loading Failures**
   - Check checkpoint path
   - Verify GPU availability
   - Check disk space

3. **Audio Generation Errors**
   - Validate input parameters
   - Check output directory permissions
   - Verify audio format support

### Logs

Service logs are available at:
- Console output for development
- `/tmp/ace_step_outputs/logs/` for production
- BentoML logs for deployment issues

## Development

### Project Structure

```
bentoml_service/
├── service.py              # Main BentoML service
├── models/
│   └── pipeline_manager.py # Pipeline management
├── api/
│   └── schemas.py          # Request/response schemas
├── utils/
│   ├── audio_utils.py      # Audio processing utilities
│   └── config.py           # Configuration management
└── __init__.py
```

### Adding New Features

1. **New Endpoints**: Add to `service.py`
2. **New Schemas**: Add to `api/schemas.py`
3. **New Utilities**: Add to `utils/`
4. **Configuration**: Update `utils/config.py`

## License

This service is part of the ACE-Steps project and follows the same Apache 2.0 license.
