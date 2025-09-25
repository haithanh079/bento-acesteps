#!/bin/bash

# Build script for ACE-Step BentoML service
set -e

echo "🚀 Starting ACE-Step BentoML build process..."

# Check if BentoML is installed
if ! command -v bentoml &> /dev/null; then
    echo "❌ BentoML is not installed. Please install it first:"
    echo "   pip install bentoml"
    exit 1
fi

# Check if Docker is running
if ! docker info &> /dev/null; then
    echo "❌ Docker is not running. Please start Docker first."
    exit 1
fi

# Set variables
BENTO_NAME="ace-step-audio-service"
DOCKER_IMAGE_NAME="ace-step-audio"
DOCKER_TAG="latest"

echo "📦 Building BentoML service..."
bentoml build

echo "✅ BentoML build completed successfully!"

# Get the latest bento
LATEST_BENTO=$(bentoml list | head -n 2 | tail -n 1 | awk '{print $1":"$2}')

echo "🐳 Building Docker image from Bento: $LATEST_BENTO"

# Build Docker image using custom Dockerfile
bentoml containerize $LATEST_BENTO \
    --dockerfile Dockerfile.bentoml \
    --tag $DOCKER_IMAGE_NAME:$DOCKER_TAG

echo "✅ Docker image built successfully!"
echo "📋 Image details:"
docker images | grep $DOCKER_IMAGE_NAME

echo ""
echo "🎉 Build process completed!"
echo ""
echo "To run the container:"
echo "  docker run --gpus all -p 3000:3000 $DOCKER_IMAGE_NAME:$DOCKER_TAG"
echo ""
echo "To test the service:"
echo "  curl http://localhost:3000/health"
echo ""
echo "To access the API documentation:"
echo "  http://localhost:3000/docs"
