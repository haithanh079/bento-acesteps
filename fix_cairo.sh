#!/bin/bash

# Quick fix for Cairo dependency issues
# This script installs the missing Cairo libraries that cause pycairo build failures

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_info "🔧 Fixing Cairo dependency issues for pycairo..."

# Check if we're on Ubuntu/Debian
if ! command -v apt &> /dev/null; then
    log_error "This script is for Ubuntu/Debian systems with apt package manager"
    exit 1
fi

# Update package list
log_info "Updating package list..."
apt update

# Install Cairo and related development libraries
log_info "Installing Cairo development libraries..."
apt install -y \
    libcairo2-dev \
    libgirepository1.0-dev \
    libglib2.0-dev \
    libgtk-3-dev \
    libpango1.0-dev \
    libgdk-pixbuf2.0-dev \
    libffi-dev \
    pkg-config

# Install additional build dependencies that might be missing
log_info "Installing additional build dependencies..."
apt install -y \
    build-essential \
    python3-dev \
    python3-pip \
    cmake \
    meson \
    ninja-build

log_success "Cairo dependencies installed successfully!"

# Test if Cairo is now available
log_info "Testing Cairo installation..."
if pkg-config --exists cairo; then
    CAIRO_VERSION=$(pkg-config --modversion cairo)
    log_success "Cairo $CAIRO_VERSION is now available"
else
    log_error "Cairo installation may have failed"
    exit 1
fi

# Now try to install pycairo
log_info "Attempting to install pycairo..."
if python3 -m pip install pycairo; then
    log_success "pycairo installed successfully!"
else
    log_error "pycairo installation failed. You may need to:"
    echo "1. Ensure you're in the correct Python virtual environment"
    echo "2. Try: pip install --no-cache-dir pycairo"
    echo "3. Or try: pip install --upgrade pip setuptools wheel"
    exit 1
fi

log_success "🎉 Cairo fix completed! You should now be able to install packages that depend on pycairo."

# Additional recommendations
echo
log_info "Additional recommendations:"
echo "- If you still have issues, try: pip install --no-cache-dir --force-reinstall pycairo"
echo "- For matplotlib issues, try: pip install --no-cache-dir matplotlib"
echo "- Make sure you're in your virtual environment when installing Python packages"
