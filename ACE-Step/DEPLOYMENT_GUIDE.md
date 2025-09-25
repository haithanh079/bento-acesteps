# ACE-Step BentoML Deployment Guide

Hướng dẫn deploy ACE-Step service sử dụng BentoML và Docker.

## 📋 Yêu cầu hệ thống

- **Python**: 3.10+
- **BentoML**: 1.4.19+
- **Docker**: 20.10+
- **NVIDIA GPU**: CUDA 12.6+ (khuyến nghị)
- **RAM**: Tối thiểu 8GB
- **Storage**: Tối thiểu 20GB free space

## 🚀 Các bước deployment

### 1. Chuẩn bị môi trường

```bash
# Cài đặt BentoML
pip install bentoml==1.4.25

# Cài đặt dependencies
pip install -r requirements-bentoml.txt

# Kiểm tra Docker
docker --version
```

### 2. Build BentoML service

```bash
# Chạy build script tự động
./build.sh

# Hoặc chạy thủ công:
bentoml build
```

### 3. Build Docker image

```bash
# Sử dụng custom Dockerfile
bentoml containerize <bento-name>:<version> \
    --dockerfile Dockerfile.bentoml \
    --tag ace-step-audio:latest
```

### 4. Chạy container

```bash
# Chạy với GPU support
docker run --gpus all \
    -p 3000:3000 \
    -v /tmp/ace_step_outputs:/tmp/ace_step_outputs \
    -v /tmp/huggingface_cache:/tmp/huggingface_cache \
    ace-step-audio:latest

# Chạy với CPU only (không khuyến nghị)
docker run -p 3000:3000 ace-step-audio:latest
```

### 5. Test service

```bash
# Chạy test suite
python test_service.py

# Test thủ công
curl http://localhost:3000/health
curl http://localhost:3000/v1/models
```

## 🔧 Cấu hình

### Environment Variables

```bash
# GPU settings
export CUDA_VISIBLE_DEVICES=0

# Model settings
export ACE_STEP_CHECKPOINT_PATH=/path/to/checkpoints
export ACE_PIPELINE_DTYPE=bfloat16
export TORCH_COMPILE=false
export CPU_OFFLOAD=false

# Service settings
export MAX_AUDIO_DURATION=240.0
export DEFAULT_AUDIO_DURATION=30.0
export OUTPUT_DIR=/tmp/ace_step_outputs
export CLEANUP_FILES=true
export FILE_RETENTION_HOURS=24
```

### Docker Compose (khuyến nghị)

Tạo file `docker-compose.yml`:

```yaml
version: '3.8'

services:
  ace-step-service:
    build:
      context: .
      dockerfile: Dockerfile.bentoml
    ports:
      - "3000:3000"
    volumes:
      - ./outputs:/tmp/ace_step_outputs
      - ./cache:/tmp/huggingface_cache
    environment:
      - CUDA_VISIBLE_DEVICES=0
      - ACE_STEP_CHECKPOINT_PATH=/app/checkpoints
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    restart: unless-stopped
```

Chạy với Docker Compose:

```bash
docker-compose up -d
```

## 📡 API Endpoints

### OpenAI Compatible API

- **Base URL**: `http://localhost:3000/v1`
- **Health**: `GET /health`
- **Models**: `GET /models`
- **Audio Generation**: `POST /audio/generations`

### BentoML Native API

- **Base URL**: `http://localhost:3000`
- **Generate Audio**: `POST /generate_audio`

### Example Usage

```python
import requests

# Generate audio
response = requests.post(
    "http://localhost:3000/v1/audio/generations",
    json={
        "model": "ace-step-v1",
        "prompt": "A beautiful piano melody",
        "duration": 30.0,
        "guidance_scale": 15.0,
        "num_inference_steps": 60
    }
)

print(response.json())
```

## 🐛 Troubleshooting

### Lỗi thường gặp

1. **CUDA out of memory**
   ```bash
   # Giảm batch size hoặc enable CPU offload
   export CPU_OFFLOAD=true
   ```

2. **Model not found**
   ```bash
   # Kiểm tra checkpoint path
   export ACE_STEP_CHECKPOINT_PATH=/correct/path/to/checkpoints
   ```

3. **Port already in use**
   ```bash
   # Thay đổi port
   docker run -p 3001:3000 ace-step-audio:latest
   ```

### Logs và monitoring

```bash
# Xem logs
docker logs <container_id>

# Monitor resource usage
docker stats <container_id>

# Health check
curl http://localhost:3000/health
```

## 🔄 Updates và Maintenance

### Update service

```bash
# Rebuild BentoML
bentoml build

# Rebuild Docker image
docker build -f Dockerfile.bentoml -t ace-step-audio:latest .

# Restart container
docker-compose down && docker-compose up -d
```

### Cleanup

```bash
# Clean old files
docker exec <container_id> python -c "
from acestep.service import file_manager
file_manager.cleanup_old_files()
"

# Remove old containers
docker system prune -f
```

## 📊 Performance Tuning

### GPU Optimization

```bash
# Enable torch.compile
export TORCH_COMPILE=true

# Use mixed precision
export ACE_PIPELINE_DTYPE=bfloat16

# Enable overlapped decode
export OVERLAPPED_DECODE=true
```

### Memory Optimization

```bash
# Enable CPU offload
export CPU_OFFLOAD=true

# Reduce max duration
export MAX_AUDIO_DURATION=120.0

# Cleanup files more frequently
export FILE_RETENTION_HOURS=12
```

## 🚨 Security Notes

- Service chạy với non-root user
- File access được giới hạn trong output directory
- CORS được enable mặc định (có thể tắt)
- API key authentication có thể được enable

## 📞 Support

Nếu gặp vấn đề:

1. Kiểm tra logs: `docker logs <container_id>`
2. Chạy test suite: `python test_service.py`
3. Kiểm tra GPU: `nvidia-smi`
4. Kiểm tra ports: `netstat -tlnp | grep 3000`
