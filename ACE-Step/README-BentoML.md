# ACE-Step OpenAI-Compatible BentoML Service

This is an OpenAI-compatible BentoML 1.4.25 service for ACE-Step audio generation. It provides a REST API that follows OpenAI's API conventions for generating audio from text prompts.

## Features

- **OpenAI-Compatible API**: Follows OpenAI's API structure for easy integration
- **Text-to-Audio Generation**: Generate high-quality audio from text prompts and lyrics
- **Multiple Audio Formats**: Support for WAV, MP3, and OGG formats
- **Configurable Parameters**: Control generation with various parameters like guidance scale, inference steps, etc.
- **File Management**: Automatic cleanup of generated files with configurable retention
- **Error Handling**: Comprehensive error handling with OpenAI-compatible error responses using BentoML exceptions
- **CORS Support**: Built-in CORS support for web applications
- **Monitoring & Observability**: Built-in metrics and access logging with BentoML 1.4.25
- **Enhanced Security**: Proper exception handling and input validation

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements-bentoml.txt
```

**Note**: This service requires BentoML 1.4.25 with enhanced features including:
- Improved metrics and observability
- Better exception handling
- Enhanced logging capabilities
- Pydantic Settings integration

### 2. Set Environment Variables

Create a `.env` file or set environment variables:

```bash
export ACE_STEP_CHECKPOINT_PATH="/path/to/ace-step/checkpoints"
export CUDA_VISIBLE_DEVICES=0
export OUTPUT_DIR="/tmp/ace_step_outputs"
```

### 3. Start the Service

```bash
bentoml serve service:acestepaudoservice
```

The service will be available at `http://localhost:3000`

### 4. Test the API

```bash
curl -X POST "http://localhost:3000/v1/audio/generations" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "ace-step-v1",
    "prompt": "A peaceful piano melody",
    "lyrics": "La la la, peaceful sounds",
    "duration": 30.0,
    "response_format": "wav"
  }'
```

## API Endpoints

### Audio Generation

**POST** `/v1/audio/generations`

Generate audio from text prompt.

**Request Body:**
```json
{
  "model": "ace-step-v1",
  "prompt": "A peaceful piano melody",
  "lyrics": "Optional lyrics text",
  "duration": 30.0,
  "guidance_scale": 15.0,
  "num_inference_steps": 60,
  "seed": 42,
  "scheduler": "euler",
  "cfg_type": "apg",
  "omega_scale": 10.0,
  "response_format": "wav"
}
```

**Response:**
```json
{
  "id": "audio_gen_abc123",
  "object": "audio.generation",
  "created": 1234567890,
  "model": "ace-step-v1",
  "data": [
    {
      "url": "/v1/audio/files/audio_20240101_120000_abc12345.wav",
      "revised_prompt": "A peaceful piano melody"
    }
  ]
}
```

### File Operations

**GET** `/v1/audio/files/{filename}`
Download generated audio file.

**DELETE** `/v1/audio/files/{filename}`
Delete generated audio file.

**POST** `/v1/audio/upload`
Upload audio file (for future features).

### Model Information

**GET** `/v1/models`
List available models.

**GET** `/v1/models/{model_id}`
Get specific model information.

### Utility

**GET** `/v1/health`
Health check endpoint.

**GET** `/v1/audio/status/{generation_id}`
Get generation status (placeholder for async processing).

## Configuration

The service can be configured through environment variables or a `.env` file:

### Model Settings
- `ACE_STEP_CHECKPOINT_PATH`: Path to model checkpoints
- `CUDA_VISIBLE_DEVICES`: GPU device ID (default: 0)
- `ACE_PIPELINE_DTYPE`: Model dtype (default: bfloat16)
- `TORCH_COMPILE`: Enable torch.compile (default: false)
- `CPU_OFFLOAD`: Enable CPU offloading (default: false)
- `OVERLAPPED_DECODE`: Enable overlapped decoding (default: false)

### Service Settings
- `MAX_AUDIO_DURATION`: Maximum audio duration in seconds (default: 240.0)
- `DEFAULT_AUDIO_DURATION`: Default audio duration (default: 30.0)
- `MAX_INFERENCE_STEPS`: Maximum inference steps (default: 200)
- `DEFAULT_INFERENCE_STEPS`: Default inference steps (default: 60)

### File Storage
- `OUTPUT_DIR`: Directory for output files (default: /tmp/ace_step_outputs)
- `CLEANUP_FILES`: Auto cleanup files (default: true)
- `FILE_RETENTION_HOURS`: File retention period (default: 24)

### API Settings
- `ENABLE_CORS`: Enable CORS (default: true)
- `API_KEY`: API key for authentication (optional)
- `RATE_LIMIT_PER_MINUTE`: Rate limit per client (default: 10)

## Python Client Example

```python
import requests
import json

# Generate audio
response = requests.post(
    "http://localhost:3000/v1/audio/generations",
    headers={"Content-Type": "application/json"},
    json={
        "model": "ace-step-v1",
        "prompt": "A cheerful folk song with guitar",
        "lyrics": "Walking down the sunny road, feeling good today",
        "duration": 45.0,
        "guidance_scale": 12.0,
        "num_inference_steps": 50,
        "response_format": "wav"
    }
)

if response.status_code == 200:
    result = response.json()
    audio_url = result["data"][0]["url"]
    
    # Download the audio file
    audio_response = requests.get(f"http://localhost:3000{audio_url}")
    with open("generated_audio.wav", "wb") as f:
        f.write(audio_response.content)
    
    print("Audio generated and saved!")
else:
    print(f"Error: {response.status_code} - {response.text}")
```

## OpenAI Python Client Example

You can also use the OpenAI Python client by pointing it to your BentoML service:

```python
from openai import OpenAI

# Point to your BentoML service
client = OpenAI(
    base_url="http://localhost:3000/v1",
    api_key="not-needed"  # API key not required for local service
)

# This would work if OpenAI client supported audio generation
# For now, use the requests example above
```

## Deployment

### Build Bento

```bash
bentoml build
```

### Deploy to BentoCloud

```bash
# Login to BentoCloud
bentoml cloud login

# Deploy the service
bentoml deploy
```

### Docker Deployment

```bash
# Build container
bentoml containerize ace-step-audio-service:latest

# Run container with GPU support
docker run --gpus all -p 3000:3000 \
  -e ACE_STEP_CHECKPOINT_PATH=/path/to/checkpoints \
  -v /path/to/checkpoints:/path/to/checkpoints \
  ace-step-audio-service:latest
```

### Monitoring

With BentoML 1.4.25, the service includes built-in metrics:

- **Request Duration**: Histogram with custom buckets for audio generation timing
- **Request Count**: Counter for API calls
- **Error Rate**: Error tracking with proper categorization
- **Access Logs**: Detailed request/response logging

Access metrics at `http://localhost:3000/metrics` (Prometheus format)

## Development

### Running Tests

```bash
pytest tests/
```

### Code Formatting

```bash
black service.py config.py
isort service.py config.py
```

### Linting

```bash
flake8 service.py config.py
mypy service.py config.py
```

## Troubleshooting

### Common Issues

1. **Model not loading**: Check that `ACE_STEP_CHECKPOINT_PATH` points to valid checkpoints
2. **CUDA out of memory**: Reduce batch size or enable CPU offloading
3. **File not found**: Check that output directory exists and has write permissions
4. **Generation timeout**: Increase timeout in service configuration

### Logs

Check service logs for detailed error information:

```bash
bentoml serve service:acestepaudoservice --reload
```

## License

This service wrapper is provided under the same license as the original ACE-Step project.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## Support

For issues related to:
- **BentoML service**: Open an issue in this repository
- **ACE-Step model**: Check the original ACE-Step repository
- **Audio generation quality**: Adjust model parameters or check model documentation
