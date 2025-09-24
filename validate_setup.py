#!/usr/bin/env python3
"""
Validation script for ACE-Step BentoML Ubuntu setup
This script checks if all dependencies and configurations are properly installed
"""

import sys
import subprocess
import importlib
import os
import platform
from pathlib import Path

# Colors for output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def print_status(message, status="info"):
    """Print colored status messages"""
    if status == "success":
        print(f"{Colors.GREEN}✓{Colors.ENDC} {message}")
    elif status == "error":
        print(f"{Colors.RED}✗{Colors.ENDC} {message}")
    elif status == "warning":
        print(f"{Colors.YELLOW}⚠{Colors.ENDC} {message}")
    else:
        print(f"{Colors.BLUE}ℹ{Colors.ENDC} {message}")

def check_python_version():
    """Check Python version"""
    version = sys.version_info
    if version.major == 3 and version.minor >= 10:
        print_status(f"Python {version.major}.{version.minor}.{version.micro}", "success")
        return True
    else:
        print_status(f"Python {version.major}.{version.minor}.{version.micro} - Need Python 3.10+", "error")
        return False

def check_system_packages():
    """Check if required system packages are installed"""
    packages = [
        ("ffmpeg", "ffmpeg -version"),
        ("sox", "sox --version"),
        ("git", "git --version"),
        ("curl", "curl --version"),
        ("build tools", "gcc --version")
    ]
    
    all_good = True
    for name, cmd in packages:
        try:
            result = subprocess.run(cmd.split(), capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                print_status(f"{name} is installed", "success")
            else:
                print_status(f"{name} is not working properly", "error")
                all_good = False
        except (subprocess.TimeoutExpired, FileNotFoundError):
            print_status(f"{name} is not installed", "error")
            all_good = False
    
    return all_good

def check_python_packages():
    """Check if required Python packages are installed"""
    packages = [
        ("bentoml", "1.4.25"),
        ("torch", None),
        ("transformers", "4.50.0"),
        ("diffusers", None),
        ("librosa", "0.11.0"),
        ("soundfile", "0.13.1"),
        ("fastapi", None),
        ("pydantic", None),
        ("uvicorn", None),
        ("numpy", None),
        ("tqdm", None)
    ]
    
    all_good = True
    for package, expected_version in packages:
        try:
            module = importlib.import_module(package)
            version = getattr(module, '__version__', 'unknown')
            
            if expected_version and version != expected_version:
                print_status(f"{package} {version} (expected {expected_version})", "warning")
            else:
                print_status(f"{package} {version}", "success")
        except ImportError:
            print_status(f"{package} is not installed", "error")
            all_good = False
    
    return all_good

def check_gpu_support():
    """Check GPU and CUDA support"""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            cuda_version = torch.version.cuda
            print_status(f"CUDA {cuda_version} with {gpu_count} GPU(s): {gpu_name}", "success")
            
            # Test GPU memory
            try:
                device = torch.device('cuda:0')
                x = torch.randn(1000, 1000, device=device)
                memory_allocated = torch.cuda.memory_allocated(0) / 1024**3
                print_status(f"GPU memory test passed ({memory_allocated:.2f}GB allocated)", "success")
                del x
                torch.cuda.empty_cache()
                return True
            except Exception as e:
                print_status(f"GPU memory test failed: {e}", "error")
                return False
        else:
            print_status("CUDA not available - using CPU mode", "warning")
            return False
    except ImportError:
        print_status("PyTorch not installed", "error")
        return False

def check_environment_variables():
    """Check important environment variables"""
    env_vars = [
        ("CUDA_VISIBLE_DEVICES", "0"),
        ("TOKENIZERS_PARALLELISM", "false"),
        ("PYTHONUNBUFFERED", "1"),
        ("HF_HUB_CACHE", None),
        ("ACE_STEP_CHECKPOINT_PATH", None),
        ("OUTPUT_DIR", "/tmp/ace_step_outputs")
    ]
    
    all_good = True
    for var, default in env_vars:
        value = os.environ.get(var)
        if value:
            print_status(f"{var}={value}", "success")
        elif default:
            print_status(f"{var} not set (default: {default})", "warning")
        else:
            print_status(f"{var} not set", "warning")
    
    return all_good

def check_directories():
    """Check if required directories exist and are writable"""
    dirs = [
        os.environ.get("OUTPUT_DIR", "/tmp/ace_step_outputs"),
        os.environ.get("HF_HUB_CACHE", os.path.expanduser("~/.cache/huggingface")),
        "/tmp"
    ]
    
    all_good = True
    for dir_path in dirs:
        path = Path(dir_path)
        if path.exists():
            if os.access(path, os.W_OK):
                print_status(f"Directory {dir_path} exists and is writable", "success")
            else:
                print_status(f"Directory {dir_path} exists but is not writable", "error")
                all_good = False
        else:
            try:
                path.mkdir(parents=True, exist_ok=True)
                print_status(f"Created directory {dir_path}", "success")
            except Exception as e:
                print_status(f"Cannot create directory {dir_path}: {e}", "error")
                all_good = False
    
    return all_good

def check_bentoml_service():
    """Check if BentoML can import the service"""
    service_files = ["service.py", "config.py", "bentofile.yaml"]
    
    all_good = True
    for file in service_files:
        if Path(file).exists():
            print_status(f"Service file {file} exists", "success")
        else:
            print_status(f"Service file {file} missing", "error")
            all_good = False
    
    # Try to validate bentofile.yaml
    if Path("bentofile.yaml").exists():
        try:
            import yaml
            with open("bentofile.yaml", 'r') as f:
                config = yaml.safe_load(f)
            if "service" in config:
                print_status("bentofile.yaml is valid", "success")
            else:
                print_status("bentofile.yaml missing 'service' field", "error")
                all_good = False
        except Exception as e:
            print_status(f"bentofile.yaml validation failed: {e}", "error")
            all_good = False
    
    return all_good

def check_network_connectivity():
    """Check network connectivity for downloading models"""
    urls = [
        "https://huggingface.co",
        "https://pypi.org",
        "https://github.com"
    ]
    
    all_good = True
    for url in urls:
        try:
            result = subprocess.run(
                ["curl", "-s", "--connect-timeout", "5", url],
                capture_output=True,
                timeout=10
            )
            if result.returncode == 0:
                print_status(f"Can connect to {url}", "success")
            else:
                print_status(f"Cannot connect to {url}", "warning")
        except Exception:
            print_status(f"Cannot test connection to {url}", "warning")
    
    return all_good

def main():
    """Main validation function"""
    print(f"{Colors.BOLD}ACE-Step BentoML Ubuntu Setup Validation{Colors.ENDC}")
    print("=" * 50)
    
    # System information
    print(f"\n{Colors.BOLD}System Information:{Colors.ENDC}")
    print_status(f"OS: {platform.system()} {platform.release()}")
    print_status(f"Architecture: {platform.machine()}")
    print_status(f"Python executable: {sys.executable}")
    
    # Run all checks
    checks = [
        ("Python Version", check_python_version),
        ("System Packages", check_system_packages),
        ("Python Packages", check_python_packages),
        ("GPU Support", check_gpu_support),
        ("Environment Variables", check_environment_variables),
        ("Directories", check_directories),
        ("BentoML Service Files", check_bentoml_service),
        ("Network Connectivity", check_network_connectivity)
    ]
    
    results = {}
    for name, check_func in checks:
        print(f"\n{Colors.BOLD}{name}:{Colors.ENDC}")
        try:
            results[name] = check_func()
        except Exception as e:
            print_status(f"Check failed with error: {e}", "error")
            results[name] = False
    
    # Summary
    print(f"\n{Colors.BOLD}Summary:{Colors.ENDC}")
    print("=" * 30)
    
    passed = sum(results.values())
    total = len(results)
    
    for name, result in results.items():
        status = "success" if result else "error"
        print_status(f"{name}: {'PASS' if result else 'FAIL'}", status)
    
    print(f"\nOverall: {passed}/{total} checks passed")
    
    if passed == total:
        print_status("All checks passed! Your environment is ready.", "success")
        return 0
    else:
        print_status(f"{total - passed} checks failed. Please review the issues above.", "error")
        return 1

if __name__ == "__main__":
    sys.exit(main())
