#!/usr/bin/env python3
"""
Health check script for ACE-Step BentoML service
"""

import requests
import sys
import time
import argparse


def check_health(base_url: str, timeout: int = 30) -> bool:
    """Check if the service is healthy"""
    try:
        response = requests.get(f"{base_url}/v1/health", timeout=timeout)
        return response.status_code == 200
    except Exception as e:
        print(f"Health check failed: {e}")
        return False


def check_models(base_url: str, timeout: int = 30) -> bool:
    """Check if models are available"""
    try:
        response = requests.get(f"{base_url}/v1/models", timeout=timeout)
        if response.status_code == 200:
            data = response.json()
            return len(data.get("data", [])) > 0
        return False
    except Exception as e:
        print(f"Models check failed: {e}")
        return False


def wait_for_service(base_url: str, max_wait: int = 300) -> bool:
    """Wait for service to become ready"""
    print(f"Waiting for service at {base_url} to become ready...")
    
    start_time = time.time()
    while time.time() - start_time < max_wait:
        if check_health(base_url):
            print("✓ Service is healthy")
            if check_models(base_url):
                print("✓ Models are available")
                return True
            else:
                print("⚠ Service is healthy but models are not ready")
        
        print(".", end="", flush=True)
        time.sleep(5)
    
    print(f"\n❌ Service did not become ready within {max_wait} seconds")
    return False


def main():
    parser = argparse.ArgumentParser(description="Health check for ACE-Step BentoML service")
    parser.add_argument("--url", default="http://localhost:3000", help="Service URL")
    parser.add_argument("--timeout", type=int, default=30, help="Request timeout in seconds")
    parser.add_argument("--wait", type=int, default=300, help="Max wait time for service to be ready")
    parser.add_argument("--check-only", action="store_true", help="Only check once, don't wait")
    
    args = parser.parse_args()
    
    if args.check_only:
        # Single health check
        if check_health(args.url, args.timeout):
            print("✓ Service is healthy")
            if check_models(args.url, args.timeout):
                print("✓ Models are available")
                sys.exit(0)
            else:
                print("⚠ Service is healthy but models are not ready")
                sys.exit(1)
        else:
            print("❌ Service is not healthy")
            sys.exit(1)
    else:
        # Wait for service to be ready
        if wait_for_service(args.url, args.wait):
            print("🎉 Service is ready!")
            sys.exit(0)
        else:
            sys.exit(1)


if __name__ == "__main__":
    main()
