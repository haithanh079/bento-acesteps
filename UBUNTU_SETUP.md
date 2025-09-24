# Ubuntu Setup Guide for ACE-Step BentoML Service

This guide provides step-by-step instructions for setting up the ACE-Step BentoML audio generation service on Ubuntu.

## Quick Start

### Option 1: Automated Installation (Recommended)

```bash
# Download and run the installation script
wget https://raw.githubusercontent.com/your-repo/ace-step-bentoml/main/install_ubuntu.sh
chmod +x install_ubuntu.sh

# Basic installation
./install_ubuntu.sh

# Full installation with GPU support and Docker
./install_ubuntu.sh --cuda --docker --service
```

### Option 2: Manual Installation

Follow the steps below for manual installation.

## System Requirements

- **OS**: Ubuntu 18.04+ (20.04+ recommended)
- **Python**: 3.10+
- **RAM**: 8GB minimum, 16GB+ recommended
- **Storage**: 20GB+ free space
- **GPU**: NVIDIA GPU with 8GB+ VRAM (optional but recommended)

## Step 1: Update System

```bash
sudo apt update && sudo apt upgrade -y
```

## Step 2: Install System Dependencies

```bash
# Essential build tools
sudo apt install -y build-essential curl wget git software-properties-common

# Audio processing libraries
sudo apt install -y ffmpeg sox libsox-fmt-all libsndfile1 libsndfile1-dev

# Python development
sudo apt install -y python3-dev python3-pip python3-venv python3-setuptools

# ML/AI libraries
sudo apt install -y libblas-dev liblapack-dev libatlas-base-dev gfortran libhdf5-dev
```

## Step 3: Install Python 3.10

```bash
# Add deadsnakes PPA
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt update

# Install Python 3.10
sudo apt install -y python3.10 python3.10-dev python3.10-venv python3.10-distutils

# Set as default python3
sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# Install pip
curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10
```

## Step 4: Install NVIDIA Drivers and CUDA (Optional)

Only if you have an NVIDIA GPU:

```bash
# Install NVIDIA drivers
sudo apt install -y nvidia-driver-535

# Add CUDA repository
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt update

# Install CUDA 11.8
sudo apt install -y cuda-11-8

# Add to PATH
echo 'export PATH=/usr/local/cuda-11.8/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc

# Reboot required
sudo reboot
```

## Step 5: Set Up Project Environment

```bash
# Create project directory
mkdir -p ~/ace-step-bentoml
cd ~/ace-step-bentoml

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel
```

## Step 6: Install BentoML and Dependencies

```bash
# Install BentoML
pip install bentoml==1.4.25

# Install core dependencies
pip install fastapi>=0.104.0 pydantic>=2.0.0 pydantic-settings>=2.0.0 uvicorn[standard]>=0.24.0

# Install PyTorch (with CUDA if available)
# For GPU:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CPU only:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install ML/Audio dependencies
pip install transformers==4.50.0 diffusers>=0.33.0 librosa==0.11.0 soundfile==0.13.1
pip install datasets==3.4.1 accelerate==1.6.0 loguru==0.7.3 tqdm numpy matplotlib==3.10.1
pip install python-multipart>=0.0.6 aiofiles>=23.0.0 huggingface-hub>=0.19.0
```

## Step 7: Set Up Service Files

```bash
# Clone or copy your service files
# Example structure:
# ~/ace-step-bentoml/
# ├── service.py
# ├── config.py
# ├── bentofile.yaml
# ├── requirements-bentoml.txt
# ├── setup.sh
# └── acestep/
```

## Step 8: Configure Environment

Create `.env` file:

```bash
cat > .env << 'EOF'
# Model Settings
ACE_STEP_CHECKPOINT_PATH=/path/to/your/checkpoints
CUDA_VISIBLE_DEVICES=0
ACE_PIPELINE_DTYPE=bfloat16

# Service Settings
MAX_AUDIO_DURATION=240.0
DEFAULT_AUDIO_DURATION=30.0
OUTPUT_DIR=/tmp/ace_step_outputs

# API Settings
ENABLE_CORS=true
TOKENIZERS_PARALLELISM=false
PYTHONUNBUFFERED=1
EOF
```

## Step 9: Build and Test Service

```bash
# Build BentoML service
bentoml build

# Test the service
bentoml serve service:acestepaudoservice
```

## Step 10: Docker Setup (Optional)

```bash
# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER

# Install NVIDIA Container Toolkit (for GPU support)
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt update && sudo apt install -y nvidia-container-toolkit
sudo systemctl restart docker

# Build container
bentoml containerize ace-step-audio-service:latest

# Run container
docker run --gpus all -p 3000:3000 \
  -e ACE_STEP_CHECKPOINT_PATH=/checkpoints \
  -v /path/to/checkpoints:/checkpoints \
  ace-step-audio-service:latest
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   ```bash
   export CUDA_VISIBLE_DEVICES=0
   # Or enable CPU offloading in config
   ```

2. **Permission Denied for Output Directory**
   ```bash
   sudo mkdir -p /tmp/ace_step_outputs
   sudo chown $USER:$USER /tmp/ace_step_outputs
   ```

3. **Missing Audio Libraries**
   ```bash
   sudo apt install -y libsndfile1-dev portaudio19-dev
   ```

4. **Python Version Issues**
   ```bash
   # Ensure Python 3.10 is default
   python3 --version
   # Should show Python 3.10.x
   ```

### Performance Optimization

1. **Enable GPU Acceleration**
   - Ensure CUDA is properly installed
   - Set `CUDA_VISIBLE_DEVICES=0`
   - Use `torch.compile=True` for faster inference

2. **Memory Optimization**
   - Enable CPU offloading: `CPU_OFFLOAD=true`
   - Use bfloat16 precision: `ACE_PIPELINE_DTYPE=bfloat16`
   - Reduce batch size if needed

3. **Storage Optimization**
   - Use SSD for model checkpoints
   - Enable automatic file cleanup
   - Set appropriate retention periods

## Systemd Service (Production)

Create a systemd service for production deployment:

```bash
sudo tee /etc/systemd/system/ace-step-bentoml.service > /dev/null << EOF
[Unit]
Description=ACE-Step BentoML Audio Generation Service
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$HOME/ace-step-bentoml
Environment=PATH=$HOME/ace-step-bentoml/venv/bin
ExecStart=$HOME/ace-step-bentoml/venv/bin/bentoml serve service:acestepaudoservice
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Enable and start service
sudo systemctl daemon-reload
sudo systemctl enable ace-step-bentoml
sudo systemctl start ace-step-bentoml

# Check status
sudo systemctl status ace-step-bentoml
```

## Monitoring and Logs

```bash
# View service logs
sudo journalctl -u ace-step-bentoml -f

# Monitor GPU usage
nvidia-smi -l 1

# Monitor system resources
htop

# Check service health
curl http://localhost:3000/v1/health
```

## Security Considerations

1. **Firewall Configuration**
   ```bash
   sudo ufw allow 3000/tcp
   sudo ufw enable
   ```

2. **User Permissions**
   - Run service as non-root user
   - Set appropriate file permissions
   - Use environment variables for sensitive data

3. **API Security**
   - Enable CORS only for trusted domains
   - Implement rate limiting
   - Use HTTPS in production

## Support

For issues and questions:
- Check the logs: `sudo journalctl -u ace-step-bentoml`
- Verify GPU status: `nvidia-smi`
- Test dependencies: `python -c "import torch; print(torch.cuda.is_available())"`
- Review configuration: Check `.env` file and service settings

## Updates

To update the service:

```bash
cd ~/ace-step-bentoml
source venv/bin/activate
pip install --upgrade bentoml
bentoml build
sudo systemctl restart ace-step-bentoml
```
