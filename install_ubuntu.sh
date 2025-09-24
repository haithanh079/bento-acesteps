#!/bin/bash

# ACE-Step BentoML Ubuntu Installation and Setup Script (Root Compatible)
# This script installs BentoML, sets up the environment, and prepares the system
# for running the ACE-Step audio generation service
# Can be run as root or regular user

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

# Detect if running as root
is_root() {
    [[ $EUID -eq 0 ]]
}

# Get the actual user (even when running with sudo)
get_actual_user() {
    if is_root; then
        if [[ -n "$SUDO_USER" ]]; then
            echo "$SUDO_USER"
        else
            echo "root"
        fi
    else
        echo "$USER"
    fi
}

# Get user home directory
get_user_home() {
    local actual_user=$(get_actual_user)
    if [[ "$actual_user" == "root" ]]; then
        echo "/root"
    else
        eval echo "~$actual_user"
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
    apt update
    apt upgrade -y
    log_success "System packages updated"
}

# Install system dependencies
install_system_deps() {
    log_info "Installing system dependencies..."
    
    # Essential build tools and libraries
    apt install -y \
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
    apt install -y \
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
    apt install -y \
        python3-dev \
        python3-pip \
        python3-venv \
        python3-setuptools \
        python3-wheel
    
    # Additional libraries for ML/AI
    apt install -y \
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
    add-apt-repository ppa:deadsnakes/ppa -y
    apt update
    
    # Install Python 3.10
    apt install -y \
        python3.10 \
        python3.10-dev \
        python3.10-venv \
        python3.10-distutils
    
    # Update alternatives to use Python 3.10 as default python3
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1
    
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
    apt install -y nvidia-driver-535
    
    # Add NVIDIA package repositories
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb
    dpkg -i cuda-keyring_1.0-1_all.deb
    apt update
    
    # Install CUDA 11.8 (compatible with PyTorch)
    apt install -y cuda-11-8
    
    # Add CUDA to PATH for all users
    echo 'export PATH=/usr/local/cuda-11.8/bin:$PATH' >> /etc/environment
    echo 'export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH' >> /etc/environment
    
    # Also add to current session
    export PATH=/usr/local/cuda-11.8/bin:$PATH
    export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH
    
    log_success "NVIDIA drivers and CUDA installed"
    log_warning "Please reboot your system before using GPU acceleration"
}

# Install Docker (optional, for containerization)
install_docker() {
    log_info "Installing Docker..."
    
    # Remove old versions
    apt remove -y docker docker-engine docker.io containerd runc || true
    
    # Add Docker's official GPG key
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
    
    # Add Docker repository
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | tee /etc/apt/sources.list.d/docker.list > /dev/null
    
    # Install Docker
    apt update
    apt install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
    
    # Start and enable Docker service
    systemctl start docker
    systemctl enable docker
    
    # Add user to docker group (if not root)
    local actual_user=$(get_actual_user)
    if [[ "$actual_user" != "root" ]]; then
        usermod -aG docker "$actual_user"
        log_info "Added user $actual_user to docker group"
    fi
    
    # Install NVIDIA Container Toolkit for GPU support
    if command -v nvidia-smi &> /dev/null; then
        log_info "Installing NVIDIA Container Toolkit..."
        distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
        curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | apt-key add -
        curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | tee /etc/apt/sources.list.d/nvidia-docker.list
        apt update
        apt install -y nvidia-container-toolkit
        systemctl restart docker
    fi
    
    log_success "Docker installed"
    if [[ "$actual_user" != "root" ]]; then
        log_warning "User $actual_user added to docker group. Please log out and log back in for changes to take effect"
    fi
}

# Create project directory and virtual environment
setup_project_env() {
    local actual_user=$(get_actual_user)
    local user_home=$(get_user_home)
    local project_dir="$user_home/ace-step-bentoml"
    
    log_info "Setting up project environment in $project_dir..."
    
    # Create project directory
    mkdir -p "$project_dir"
    cd "$project_dir"
    
    # Set ownership if running as root
    if is_root && [[ "$actual_user" != "root" ]]; then
        chown -R "$actual_user:$actual_user" "$project_dir"
    fi
    
    # Create Python virtual environment (run as actual user if possible)
    if is_root && [[ "$actual_user" != "root" ]]; then
        # Run as the actual user
        su - "$actual_user" -c "cd '$project_dir' && python3 -m venv venv"
        su - "$actual_user" -c "cd '$project_dir' && source venv/bin/activate && pip install --upgrade pip setuptools wheel"
    else
        # Run as current user (root or regular user)
        python3 -m venv venv
        source venv/bin/activate
        pip install --upgrade pip setuptools wheel
    fi
    
    # Install BentoML and dependencies
    log_info "Installing BentoML and dependencies..."
    
    local install_cmd="cd '$project_dir' && source venv/bin/activate && "
    install_cmd+="pip install bentoml==1.4.25 && "
    install_cmd+="pip install fastapi>=0.104.0 pydantic>=2.0.0 pydantic-settings>=2.0.0 uvicorn[standard]>=0.24.0 && "
    
    # Install PyTorch with CUDA support if available
    if command -v nvidia-smi &> /dev/null; then
        log_info "Installing PyTorch with CUDA support..."
        install_cmd+="pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 && "
    else
        log_info "Installing PyTorch CPU version..."
        install_cmd+="pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu && "
    fi
    
    # Install additional ML/Audio dependencies
    install_cmd+="pip install transformers==4.50.0 diffusers>=0.33.0 librosa==0.11.0 soundfile==0.13.1 && "
    install_cmd+="pip install datasets==3.4.1 accelerate==1.6.0 loguru==0.7.3 tqdm numpy matplotlib==3.10.1 && "
    install_cmd+="pip install python-multipart>=0.0.6 aiofiles>=23.0.0 huggingface-hub>=0.19.0"
    
    if is_root && [[ "$actual_user" != "root" ]]; then
        su - "$actual_user" -c "$install_cmd"
    else
        bash -c "$install_cmd"
    fi
    
    # Create activation script
    cat > "$project_dir/activate_env.sh" << 'EOF'
#!/bin/bash
# Activation script for ACE-Step BentoML environment

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Activate virtual environment
source "$SCRIPT_DIR/venv/bin/activate"

# Set environment variables
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export HF_HUB_CACHE=${HF_HUB_CACHE:-"$HOME/.cache/huggingface"}
export ACE_STEP_CHECKPOINT_PATH=${ACE_STEP_CHECKPOINT_PATH:-""}
export OUTPUT_DIR=${OUTPUT_DIR:-"/tmp/ace_step_outputs"}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"0"}

# Create output directory
mkdir -p "$OUTPUT_DIR"
mkdir -p "$HF_HUB_CACHE"

echo "ACE-Step BentoML environment activated!"
echo "Project directory: $SCRIPT_DIR"
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
    
    chmod +x "$project_dir/activate_env.sh"
    
    # Create environment file template
    cat > "$project_dir/.env.example" << 'EOF'
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
    
    # Set ownership if running as root
    if is_root && [[ "$actual_user" != "root" ]]; then
        chown -R "$actual_user:$actual_user" "$project_dir"
    fi
    
    log_success "Project environment created in $project_dir"
    
    # Save project path for later use
    echo "export ACE_STEP_PROJECT_DIR=\"$project_dir\"" >> /etc/environment
}

# Create systemd service (optional)
create_systemd_service() {
    local actual_user=$(get_actual_user)
    local user_home=$(get_user_home)
    local project_dir="$user_home/ace-step-bentoml"
    local service_file="/etc/systemd/system/ace-step-bentoml.service"
    
    log_info "Creating systemd service..."
    
    tee "$service_file" > /dev/null << EOF
[Unit]
Description=ACE-Step BentoML Audio Generation Service
After=network.target

[Service]
Type=simple
User=$actual_user
Group=$actual_user
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
Environment=HF_HUB_CACHE=$user_home/.cache/huggingface
Environment=OUTPUT_DIR=/tmp/ace_step_outputs

[Install]
WantedBy=multi-user.target
EOF
    
    systemctl daemon-reload
    log_success "Systemd service created. Enable with: systemctl enable ace-step-bentoml"
}

# Create directories and set permissions
setup_directories() {
    log_info "Setting up directories and permissions..."
    
    # Create common directories
    mkdir -p /tmp/ace_step_outputs
    mkdir -p /var/log/ace-step
    
    # Set permissions
    chmod 755 /tmp/ace_step_outputs
    chmod 755 /var/log/ace-step
    
    # If not root, make sure the actual user can write to these directories
    local actual_user=$(get_actual_user)
    if [[ "$actual_user" != "root" ]]; then
        chown "$actual_user:$actual_user" /tmp/ace_step_outputs
        chown "$actual_user:$actual_user" /var/log/ace-step
    fi
    
    log_success "Directories created and permissions set"
}

# Main installation function
main() {
    log_info "Starting ACE-Step BentoML Ubuntu installation..."
    
    if is_root; then
        log_info "Running as root user"
    else
        log_info "Running as regular user: $(whoami)"
    fi
    
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
                echo ""
                echo "This script can be run as root or regular user."
                echo "When run as root, it will set up the environment for the actual user."
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                exit 1
                ;;
        esac
    done
    
    # Run installation steps
    check_ubuntu
    update_system
    install_system_deps
    install_python310
    setup_directories
    
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
    local actual_user=$(get_actual_user)
    local user_home=$(get_user_home)
    local project_dir="$user_home/ace-step-bentoml"
    
    log_success "Installation completed successfully!"
    echo
    log_info "Next steps:"
    if [[ "$actual_user" == "root" ]]; then
        echo "1. Activate the environment: cd $project_dir && source activate_env.sh"
    else
        echo "1. Activate the environment: cd $project_dir && source activate_env.sh"
        if is_root; then
            echo "   (Switch to user $actual_user first: su - $actual_user)"
        fi
    fi
    echo "2. Copy your ACE-Step model checkpoints to the project directory"
    echo "3. Update the .env file with your configuration"
    echo "4. Clone/copy your service files to the project directory"
    echo "5. Build the BentoML service: bentoml build"
    echo "6. Serve the service: bentoml serve"
    echo
    
    if [[ "$INSTALL_CUDA" == true ]]; then
        log_warning "CUDA was installed. Please reboot your system before using GPU acceleration."
    fi
    
    if [[ "$INSTALL_DOCKER" == true ]] && [[ "$actual_user" != "root" ]]; then
        log_warning "Docker was installed. User $actual_user was added to docker group. Please log out and log back in for changes to take effect."
    fi
    
    log_info "Project directory: $project_dir"
    log_info "For help and documentation, check the README-BentoML.md file"
}

# Run main function with all arguments
main "$@"
