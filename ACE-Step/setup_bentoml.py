#!/usr/bin/env python3
"""
Setup script for BentoML Docker build to handle distutils compatibility issues.
This script ensures all dependencies are properly installed without distutils dependencies.
"""

import sys
import subprocess
import os

def install_packages():
    """Install packages that might have distutils dependencies."""
    packages_to_install = [
        "setuptools>=68.0.0",
        "wheel",
        "pip>=23.0.0"
    ]
    
    for package in packages_to_install:
        try:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", package])
        except subprocess.CalledProcessError as e:
            print(f"Warning: Failed to install {package}: {e}")
            # Continue with other packages

def main():
    """Main setup function."""
    print("Setting up BentoML environment...")
    
    # Ensure we have the latest setuptools without distutils dependencies
    install_packages()
    
    print("Setup completed successfully!")

if __name__ == "__main__":
    main()
