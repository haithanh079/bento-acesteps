#!/usr/bin/env python3
"""
Dependency conflict resolution script for ACE-Step BentoML
This script fixes common dependency conflicts, especially the fsspec version issue
"""

import subprocess
import sys
import importlib
import pkg_resources
from typing import List, Tuple, Dict

def run_pip_command(cmd: List[str]) -> Tuple[int, str, str]:
    """Run a pip command and return exit code, stdout, stderr"""
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip"] + cmd,
            capture_output=True,
            text=True,
            timeout=300
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return 1, "", "Command timed out"

def check_package_version(package_name: str) -> str:
    """Check installed version of a package"""
    try:
        return pkg_resources.get_distribution(package_name).version
    except pkg_resources.DistributionNotFound:
        return None

def fix_fsspec_conflict():
    """Fix the fsspec version conflict"""
    print("🔧 Fixing fsspec dependency conflict...")
    
    current_version = check_package_version("fsspec")
    if current_version:
        print(f"Current fsspec version: {current_version}")
    else:
        print("fsspec not installed")
    
    # Upgrade fsspec to meet BentoML requirements
    print("Upgrading fsspec to >=2025.7.0...")
    exit_code, stdout, stderr = run_pip_command([
        "install", "--upgrade", "fsspec>=2025.7.0"
    ])
    
    if exit_code == 0:
        new_version = check_package_version("fsspec")
        print(f"✅ fsspec upgraded to version: {new_version}")
        return True
    else:
        print(f"❌ Failed to upgrade fsspec: {stderr}")
        return False

def fix_bentoml_dependencies():
    """Fix BentoML and related dependencies"""
    print("🔧 Fixing BentoML dependencies...")
    
    # Key dependencies that often cause conflicts
    dependencies = [
        "fsspec>=2025.7.0",
        "pydantic>=2.0.0",
        "pydantic-settings>=2.0.0",
        "fastapi>=0.104.0",
        "uvicorn[standard]>=0.24.0"
    ]
    
    for dep in dependencies:
        print(f"Installing/upgrading {dep}...")
        exit_code, stdout, stderr = run_pip_command([
            "install", "--upgrade", dep
        ])
        
        if exit_code != 0:
            print(f"⚠️ Warning: Failed to install {dep}: {stderr}")
    
    # Reinstall BentoML to ensure compatibility
    print("Reinstalling BentoML...")
    exit_code, stdout, stderr = run_pip_command([
        "install", "--force-reinstall", "bentoml==1.4.25"
    ])
    
    if exit_code == 0:
        print("✅ BentoML reinstalled successfully")
        return True
    else:
        print(f"❌ Failed to reinstall BentoML: {stderr}")
        return False

def check_dependencies():
    """Check for dependency conflicts"""
    print("🔍 Checking for dependency conflicts...")
    
    exit_code, stdout, stderr = run_pip_command(["check"])
    
    if exit_code == 0:
        print("✅ No dependency conflicts found")
        return True
    else:
        print("❌ Dependency conflicts detected:")
        print(stderr)
        return False

def fix_pytorch_dependencies():
    """Fix PyTorch related dependencies"""
    print("🔧 Checking PyTorch installation...")
    
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__} is installed")
        
        # Check CUDA availability
        if torch.cuda.is_available():
            print(f"✅ CUDA {torch.version.cuda} is available")
            print(f"GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("ℹ️ CUDA not available, using CPU mode")
        
        return True
    except ImportError:
        print("❌ PyTorch not installed")
        
        # Try to install PyTorch
        print("Installing PyTorch...")
        
        # Check if NVIDIA GPU is available
        try:
            result = subprocess.run(["nvidia-smi"], capture_output=True)
            has_gpu = result.returncode == 0
        except FileNotFoundError:
            has_gpu = False
        
        if has_gpu:
            print("Installing PyTorch with CUDA support...")
            cmd = [
                "install", "torch", "torchvision", "torchaudio",
                "--index-url", "https://download.pytorch.org/whl/cu118"
            ]
        else:
            print("Installing PyTorch CPU version...")
            cmd = [
                "install", "torch", "torchvision", "torchaudio",
                "--index-url", "https://download.pytorch.org/whl/cpu"
            ]
        
        exit_code, stdout, stderr = run_pip_command(cmd)
        
        if exit_code == 0:
            print("✅ PyTorch installed successfully")
            return True
        else:
            print(f"❌ Failed to install PyTorch: {stderr}")
            return False

def install_missing_packages():
    """Install any missing packages"""
    print("🔧 Installing missing packages...")
    
    required_packages = [
        "transformers>=4.50.0",
        "diffusers>=0.33.0",
        "librosa>=0.11.0",
        "soundfile>=0.13.1",
        "datasets>=3.4.1",
        "accelerate>=1.6.0",
        "loguru>=0.7.3",
        "tqdm",
        "numpy",
        "matplotlib>=3.10.1",
        "python-multipart>=0.0.6",
        "aiofiles>=23.0.0",
        "huggingface-hub>=0.19.0"
    ]
    
    failed_packages = []
    
    for package in required_packages:
        print(f"Installing {package}...")
        exit_code, stdout, stderr = run_pip_command([
            "install", "--upgrade", package
        ])
        
        if exit_code != 0:
            failed_packages.append(package)
            print(f"⚠️ Failed to install {package}")
    
    if failed_packages:
        print(f"❌ Failed to install: {', '.join(failed_packages)}")
        return False
    else:
        print("✅ All packages installed successfully")
        return True

def check_system_dependencies():
    """Check if required system dependencies are installed"""
    print("🔍 Checking system dependencies...")
    
    # Check for Cairo libraries (common issue)
    try:
        result = subprocess.run(["pkg-config", "--exists", "cairo"], capture_output=True)
        if result.returncode == 0:
            print("✅ Cairo libraries are installed")
        else:
            print("❌ Cairo libraries missing")
            print("Install with: apt install -y libcairo2-dev libgirepository1.0-dev")
            return False
    except FileNotFoundError:
        print("⚠️ pkg-config not found, cannot check Cairo libraries")
    
    # Check for other common dependencies
    system_deps = [
        ("gcc", "build-essential"),
        ("pkg-config", "pkg-config"),
        ("python3-dev", "python3-dev")
    ]
    
    for cmd, package in system_deps:
        try:
            result = subprocess.run(["which", cmd], capture_output=True)
            if result.returncode == 0:
                print(f"✅ {cmd} is available")
            else:
                print(f"❌ {cmd} missing (install: apt install -y {package})")
        except Exception:
            print(f"⚠️ Cannot check {cmd}")
    
    return True

def main():
    """Main function to fix all dependency issues"""
    print("🚀 ACE-Step BentoML Dependency Fixer")
    print("=" * 40)
    
    # Check current environment
    print(f"Python: {sys.version}")
    print(f"pip: {subprocess.check_output([sys.executable, '-m', 'pip', '--version']).decode().strip()}")
    print()
    
    # Check system dependencies first
    check_system_dependencies()
    print()
    
    # Step 1: Fix fsspec conflict (most critical)
    if not fix_fsspec_conflict():
        print("❌ Critical: Failed to fix fsspec conflict")
        return 1
    
    # Step 2: Fix BentoML dependencies
    if not fix_bentoml_dependencies():
        print("❌ Failed to fix BentoML dependencies")
        return 1
    
    # Step 3: Fix PyTorch
    if not fix_pytorch_dependencies():
        print("⚠️ PyTorch installation issues (may affect GPU support)")
    
    # Step 4: Install missing packages
    if not install_missing_packages():
        print("⚠️ Some packages failed to install")
    
    # Step 5: Final dependency check
    print("\n🔍 Final dependency check...")
    if check_dependencies():
        print("\n🎉 All dependency conflicts resolved!")
        
        # Test BentoML import
        try:
            import bentoml
            print(f"✅ BentoML {bentoml.__version__} imported successfully")
        except ImportError as e:
            print(f"❌ Failed to import BentoML: {e}")
            return 1
        
        return 0
    else:
        print("\n❌ Some dependency conflicts remain")
        print("You may need to manually resolve these conflicts")
        return 1

if __name__ == "__main__":
    sys.exit(main())
