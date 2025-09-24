#!/bin/bash

# Multi-Arch Build Script for BentoML
# Build both AMD64 and ARM64 images from macOS ARM for deployment on Kubernetes

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# ==== Config (change for your project) ====
BENTO_FILE="bentofile.yaml"
SERVICE_NAME="ace-steps-openai"
REGISTRY="your-docker-registry"   # ví dụ: docker.io/username hoặc ghcr.io/org
TAG="latest"
PLATFORMS="linux/amd64,linux/arm64"
# ==========================================

check_prerequisites() {
    print_status "Checking prerequisites..."
    command -v bentoml >/dev/null 2>&1 || { print_error "BentoML not found"; exit 1; }
    command -v docker >/dev/null 2>&1 || { print_error "Docker not found"; exit 1; }
    docker buildx version >/dev/null 2>&1 || { print_error "Docker buildx not available"; exit 1; }
    print_success "All prerequisites met"
}

setup_buildx() {
    print_status "Setting up Docker buildx..."
    if ! docker buildx ls | grep -q "multiarch-builder"; then
        docker buildx create --name multiarch-builder --use
    else
        docker buildx use multiarch-builder
    fi
    docker buildx inspect --bootstrap
    print_success "Docker buildx ready"
}

build_bento() {
    print_status "Building Bento..."
    bentoml build -f $BENTO_FILE
    print_success "Bento built"
}

containerize_bento() {
    print_status "Containerizing Bento..."
    # Chạy containerize để tạo Dockerfile và context (image local có thể bỏ qua)
    bentoml containerize ${SERVICE_NAME}:$TAG 
    print_success "Bento containerized"
}

multiarch_build() {
    print_status "Building multi-arch image..."
    BENTO_PATH=$(bentoml list | grep $SERVICE_NAME | head -n 1 | awk '{print $1 ":" $2}')
    BENTO_DIR=$(bentoml get $BENTO_PATH --output path)

    docker buildx build \
        --platform $PLATFORMS \
        -t $REGISTRY/$SERVICE_NAME:$TAG \
        --push \
        $BENTO_DIR

    print_success "Multi-arch image pushed: $REGISTRY/$SERVICE_NAME:$TAG"
}

verify_image() {
    print_status "Verifying image manifest..."
    docker buildx imagetools inspect $REGISTRY/$SERVICE_NAME:$TAG
}

main() {
    echo "🚀 Multi-Arch Build for BentoML Service"
    echo "======================================="
    check_prerequisites
    setup_buildx
    build_bento
    containerize_bento
    multiarch_build
    verify_image
    echo ""
    echo "✅ Done! Your image is available at:"
    echo "   $REGISTRY/$SERVICE_NAME:$TAG"
}

main "$@"