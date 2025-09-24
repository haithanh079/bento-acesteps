#!/bin/bash
# Setup script for ACE-Step BentoML service

# Create necessary directories
mkdir -p /tmp/ace_step_outputs
mkdir -p /tmp/huggingface_cache

# Set permissions
chmod 777 /tmp/ace_step_outputs
chmod 777 /tmp/huggingface_cache

echo "Setup completed successfully"
