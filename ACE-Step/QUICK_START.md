# ACE-Step BentoML Quick Start Guide

Hướng dẫn nhanh để build và deploy ACE-Step service với BentoML.

## 🚀 Các bước thực hiện

### 1. Build BentoML Service
```bash
# Build BentoML service
bentoml build

# Kiểm tra bento đã tạo
bentoml list
```

### 2. Build Docker Image
```bash
# Sử dụng script tự động
./build_docker.sh

# Hoặc build thủ công
docker build -f Dockerfile.bentoml -t ace-step-audio:latest .
```

### 3. Chạy Container
```bash
# Chạy với GPU support
docker run --gpus all -p 3000:3000 ace-step-audio:latest

# Chạy với CPU only (không khuyến nghị)
docker run -p 3000:3000 ace-step-audio:latest
```

### 4. Test Service
```bash
# Chạy test nhanh
python quick_test.py

# Hoặc test thủ công
curl http://localhost:3000/health
curl http://localhost:3000/v1/models
```

## 📁 Files đã tạo

- `bentofile.yaml` - Cấu hình BentoML (đã tối ưu)
- `Dockerfile.bentoml` - Custom Dockerfile cho BentoML
- `build.sh` - Script build tự động
- `build_docker.sh` - Script build Docker đơn giản
- `test_service.py` - Test suite đầy đủ
- `quick_test.py` - Test nhanh
- `DEPLOYMENT_GUIDE.md` - Hướng dẫn chi tiết

## 🔧 Cấu hình quan trọng

### Environment Variables
```bash
export CUDA_VISIBLE_DEVICES=0
export ACE_STEP_CHECKPOINT_PATH=/path/to/checkpoints
export OUTPUT_DIR=/tmp/ace_step_outputs
```

### API Endpoints
- **Health**: `GET /health`
- **Models**: `GET /v1/models`
- **Audio Generation**: `POST /v1/audio/generations`
- **BentoML Native**: `POST /generate_audio`

## 🐛 Troubleshooting

### Lỗi thường gặp

1. **CUDA version không hỗ trợ**
   - Sửa `bentofile.yaml`: `cuda_version: "12.1"`

2. **Docker build lỗi**
   - Sử dụng `build_docker.sh` thay vì `bentoml containerize`

3. **Service không start**
   - Kiểm tra GPU: `nvidia-smi`
   - Kiểm tra ports: `netstat -tlnp | grep 3000`

## 📊 Monitoring

```bash
# Xem logs
docker logs <container_id>

# Monitor resources
docker stats <container_id>

# Health check
curl http://localhost:3000/health
```

## 🎯 Next Steps

1. **Deploy to production**: Sử dụng Docker Compose hoặc Kubernetes
2. **Scale service**: Sử dụng load balancer
3. **Monitor performance**: Sử dụng Prometheus/Grafana
4. **Add authentication**: Sử dụng API keys

## 📞 Support

Nếu gặp vấn đề:
1. Kiểm tra logs: `docker logs <container_id>`
2. Chạy test: `python quick_test.py`
3. Kiểm tra GPU: `nvidia-smi`
4. Kiểm tra ports: `netstat -tlnp | grep 3000`
