#!/bin/bash

# Rebuild ACE-Step service with device compatibility fix
set -e

echo "🔧 Rebuilding ACE-Step service with device compatibility fix..."

# Test device compatibility first
echo "🧪 Testing device compatibility..."
python test_device.py

echo ""
echo "📦 Rebuilding BentoML service..."

# Clean previous builds
echo "🧹 Cleaning previous builds..."
bentoml delete ace-step-audio-service --yes 2>/dev/null || true

# Build new service
echo "🏗️  Building new service..."
bentoml build --force

echo "✅ Service rebuilt successfully!"

# Get the latest bento
LATEST_BENTO=$(bentoml list | head -n 2 | tail -n 1 | awk '{print $1":"$2}')
echo "📦 New Bento: $LATEST_BENTO"

echo ""
echo "🐳 Rebuilding Docker image..."

# Rebuild Docker image
docker build -f Dockerfile.simple -t ace-step-audio:latest .

echo "✅ Docker image rebuilt successfully!"

echo ""
echo "🎉 Rebuild completed!"
echo ""
echo "To run the service:"
echo "  docker run --gpus all -p 3000:3000 ace-step-audio:latest"
echo ""
echo "To test the service:"
echo "  curl http://localhost:3000/health"
echo ""
echo "Device compatibility:"
echo "  - MPS: Fallback to CPU (Apple Silicon)"
echo "  - CUDA: Full support (NVIDIA GPU)"
echo "  - CPU: Full support (any system)"
