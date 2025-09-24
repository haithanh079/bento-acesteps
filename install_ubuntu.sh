#!/bin/bash

# ACE-Step BentoML Ubuntu Installation and Setup Script
# This script installs BentoML, sets up the environment, and prepares the system
# for running the ACE-Step audio generation service

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running as root
check_root() {
    if [[ $EUID -eq 0 ]]; then
        log_error "This script should not be run as root. Please run as a regular user."
        exit 1
    fi
}

# Check Ubuntu version
check_ubuntu() {
    if ! grep -q "Ubuntu" /etc/os-release; then
        log_error "This script is designed for Ubuntu. Other distributions may not work correctly."
        exit 1
    fi
    
    local ubuntu_version=$(lsb_release -rs)
    log_info "Detected Ubuntu version: $ubuntu_version"
    
    # Check if version is supported (18.04+)
    if (( $(echo "$ubuntu_version 18.04" | awk '{print ($1 < $2)}') )); then
        log_warning "Ubuntu version $ubuntu_version may not be fully supported. Recommended: 20.04+"
    fi
}

# Update system packages
update_system() {
    log_info "Updating system packages..."
    sudo apt update
    sudo apt upgrade -y
    log_success "System packages updated"
}

# Install system dependencies
install_system_deps() {
    log_info "Installing system dependencies..."
    
    # Essential build tools and libraries
    sudo apt install -y \
        build-essential \
        curl \
        wget \
        git \
        software-properties-common \
        apt-transport-https \
        ca-certificates \
        gnupg \
        lsb-release
    
    # Audio processing libraries
    sudo apt install -y \
        ffmpeg \
        sox \
        libsox-fmt-all \
        libsndfile1 \
        libsndfile1-dev \
        libasound2-dev \
        portaudio19-dev \
        libportaudio2 \
        libportaudiocpp0
    
    # Python development dependencies
    sudo apt install -y \
        python3-dev \
        python3-pip \
        python3-venv \
        python3-setuptools \
        python3-wheel
    
    # Additional libraries for ML/AI
    sudo apt install -y \
        libblas-dev \
        liblapack-dev \
        libatlas-base-dev \
        gfortran \
        libhdf5-dev \
        pkg-config
    
    log_success "System dependencies installed"
}

# Install Python 3.10 if not available
install_python310() {
    local python_version=$(python3 --version 2>&1 | grep -oP '\d+\.\d+')
    
    if [[ "$python_version" == "3.10" ]]; then
        log_info "Python 3.10 is already installed"
        return
    fi
    
    log_info "Installing Python 3.10..."
    
    # Add deadsnakes PPA for newer Python versions
    sudo add-apt-repository ppa:deadsnakes/ppa -y
    sudo apt update
    
    # Install Python 3.10
    sudo apt install -y \
        python3.10 \
        python3.10-dev \
        python3.10-venv \
        python3.10-distutils
    
    # Update alternatives to use Python 3.10 as default python3
    sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1
    
    # Install pip for Python 3.10
    curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10
    
    log_success "Python 3.10 installed and configured"
}

# Install NVIDIA drivers and CUDA (optional)
install_nvidia_cuda() {
    log_info "Checking for NVIDIA GPU..."
    
    if ! lspci | grep -i nvidia > /dev/null; then
        log_warning "No NVIDIA GPU detected. Skipping CUDA installation."
        return
    fi
    
    log_info "NVIDIA GPU detected. Installing NVIDIA drivers and CUDA..."
    
    # Install NVIDIA drivers
    sudo apt install -y nvidia-driver-535
    
    # Add NVIDIA package repositories
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb
    sudo dpkg -i cuda-keyring_1.0-1_all.deb
    sudo apt update
    
    # Install CUDA 11.8 (compatible with PyTorch)
    sudo apt install -y cuda-11-8
    
    # Add CUDA to PATH
    echo 'export PATH=/usr/local/cuda-11.8/bin:$PATH' >> ~/.bashrc
    echo 'export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
    
    log_success "NVIDIA drivers and CUDA installed"
    log_warning "Please reboot your system before using GPU acceleration"
}

# Install Docker (optional, for containerization)
install_docker() {
    log_info "Installing Docker..."
    
    # Remove old versions
    sudo apt remove -y docker docker-engine docker.io containerd runc || true
    
    # Add Docker's official GPG key
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
    
    # Add Docker repository
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
    
    # Install Docker
    sudo apt update
    sudo apt install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
    
    # Add user to docker group
    sudo usermod -aG docker $USER
    
    # Install NVIDIA Container Toolkit for GPU support
    if command -v nvidia-smi &> /dev/null; then
        log_info "Installing NVIDIA Container Toolkit..."
        distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
        curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
        curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
        sudo apt update
        sudo apt install -y nvidia-container-toolkit
        sudo systemctl restart docker
    fi
    
    log_success "Docker installed"
    log_warning "Please log out and log back in for Docker group changes to take effect"
}

# Create project directory and virtual environment
setup_project_env() {
    local project_dir="$HOME/ace-step-bentoml"
    
    log_info "Setting up project environment in $project_dir..."
    
    # Create project directory
    mkdir -p "$project_dir"
    cd "$project_dir"
    
    # Create Python virtual environment
    python3 -m venv venv
    source venv/bin/activate
    
    # Upgrade pip and install basic tools
    pip install --upgrade pip setuptools wheel
    
    # Install BentoML and dependencies
    log_info "Installing BentoML and dependencies..."
    pip install bentoml==1.4.25
    pip install fastapi>=0.104.0
    pip install pydantic>=2.0.0
    pip install pydantic-settings>=2.0.0
    pip install uvicorn[standard]>=0.24.0
    
    # Install PyTorch with CUDA support if available
    if command -v nvidia-smi &> /dev/null; then
        log_info "Installing PyTorch with CUDA support..."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    else
        log_info "Installing PyTorch CPU version..."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    fi
    
    # Install additional ML/Audio dependencies
    pip install \
        transformers==4.50.0 \
        diffusers>=0.33.0 \
        librosa==0.11.0 \
        soundfile==0.13.1 \
        datasets==3.4.1 \
        accelerate==1.6.0 \
        loguru==0.7.3 \
        tqdm \
        numpy \
        matplotlib==3.10.1 \
        python-multipart>=0.0.6 \
        aiofiles>=23.0.0 \
        huggingface-hub>=0.19.0
    
    # Create activation script
    cat > activate_env.sh << 'EOF'
#!/bin/bash
# Activation script for ACE-Step BentoML environment

# Activate virtual environment
source venv/bin/activate

# Set environment variables
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export HF_HUB_CACHE=$HOME/.cache/huggingface
export ACE_STEP_CHECKPOINT_PATH=${ACE_STEP_CHECKPOINT_PATH:-""}
export OUTPUT_DIR=${OUTPUT_DIR:-"/tmp/ace_step_outputs"}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"0"}

# Create output directory
mkdir -p "$OUTPUT_DIR"
mkdir -p "$HF_HUB_CACHE"

echo "ACE-Step BentoML environment activated!"
echo "Project directory: $(pwd)"
echo "Python: $(which python)"
echo "BentoML version: $(bentoml --version)"

# Check GPU availability
if command -v nvidia-smi &> /dev/null; then
    echo "GPU Status:"
    nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader,nounits
else
    echo "No GPU detected - using CPU mode"
fi
EOF
    
    chmod +x activate_env.sh
    
    # Create environment file template
    cat > .env.example << 'EOF'
# ACE-Step BentoML Service Configuration

# Model Settings
ACE_STEP_CHECKPOINT_PATH=/path/to/ace-step/checkpoints
CUDA_VISIBLE_DEVICES=0
ACE_PIPELINE_DTYPE=bfloat16
TORCH_COMPILE=false
CPU_OFFLOAD=false
OVERLAPPED_DECODE=false

# Service Settings
MAX_AUDIO_DURATION=240.0
DEFAULT_AUDIO_DURATION=30.0
MAX_INFERENCE_STEPS=200
DEFAULT_INFERENCE_STEPS=60

# File Storage Settings
OUTPUT_DIR=/tmp/ace_step_outputs
CLEANUP_FILES=true
FILE_RETENTION_HOURS=24

# API Settings
ENABLE_CORS=true
RATE_LIMIT_PER_MINUTE=10

# Additional PyTorch Settings
TOKENIZERS_PARALLELISM=false
PYTHONUNBUFFERED=1
HF_HUB_CACHE=$HOME/.cache/huggingface
EOF
    
    log_success "Project environment created in $project_dir"
    
    # Save project path for later use
    echo "export ACE_STEP_PROJECT_DIR=\"$project_dir\"" >> ~/.bashrc
}

# Create systemd service (optional)
create_systemd_service() {
    local project_dir="$HOME/ace-step-bentoml"
    local service_file="/etc/systemd/system/ace-step-bentoml.service"
    
    log_info "Creating systemd service..."
    
    sudo tee "$service_file" > /dev/null << EOF
[Unit]
Description=ACE-Step BentoML Audio Generation Service
After=network.target

[Service]
Type=simple
User=$USER
Group=$USER
WorkingDirectory=$project_dir
Environment=PATH=$project_dir/venv/bin
ExecStart=$project_dir/venv/bin/bentoml serve service:ACEStepAudioService
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

# Environment variables
Environment=TOKENIZERS_PARALLELISM=false
Environment=PYTHONUNBUFFERED=1
Environment=HF_HUB_CACHE=$HOME/.cache/huggingface
Environment=OUTPUT_DIR=/tmp/ace_step_outputs

[Install]
WantedBy=multi-user.target
EOF
    
    sudo systemctl daemon-reload
    log_success "Systemd service created. Enable with: sudo systemctl enable ace-step-bentoml"
}

# Main installation function
main() {
    log_info "Starting ACE-Step BentoML Ubuntu installation..."
    
    # Parse command line arguments
    INSTALL_CUDA=false
    INSTALL_DOCKER=false
    CREATE_SERVICE=false
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --cuda)
                INSTALL_CUDA=true
                shift
                ;;
            --docker)
                INSTALL_DOCKER=true
                shift
                ;;
            --service)
                CREATE_SERVICE=true
                shift
                ;;
            --help)
                echo "Usage: $0 [OPTIONS]"
                echo "Options:"
                echo "  --cuda      Install NVIDIA drivers and CUDA"
                echo "  --docker    Install Docker and NVIDIA Container Toolkit"
                echo "  --service   Create systemd service"
                echo "  --help      Show this help message"
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                exit 1
                ;;
        esac
    done
    
    # Run installation steps
    check_root
    check_ubuntu
    update_system
    install_system_deps
    install_python310
    
    if [[ "$INSTALL_CUDA" == true ]]; then
        install_nvidia_cuda
    fi
    
    if [[ "$INSTALL_DOCKER" == true ]]; then
        install_docker
    fi
    
    setup_project_env
    
    if [[ "$CREATE_SERVICE" == true ]]; then
        create_systemd_service
    fi
    
    # Final instructions
    log_success "Installation completed successfully!"
    echo
    log_info "Next steps:"
    echo "1. Activate the environment: cd ~/ace-step-bentoml && source activate_env.sh"
    echo "2. Copy your ACE-Step model checkpoints to the project directory"
    echo "3. Update the .env file with your configuration"
    echo "4. Clone/copy your service files to the project directory"
    echo "5. Build the BentoML service: bentoml build"
    echo "6. Serve the service: bentoml serve"
    echo
    
    if [[ "$INSTALL_CUDA" == true ]]; then
        log_warning "CUDA was installed. Please reboot your system before using GPU acceleration."
    fi
    
    if [[ "$INSTALL_DOCKER" == true ]]; then
        log_warning "Docker was installed. Please log out and log back in for group changes to take effect."
    fi
    
    log_info "For help and documentation, check the README-BentoML.md file"
}

# Run main function with all arguments
main "$@"
