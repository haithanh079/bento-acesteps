#!/usr/bin/env python3
"""
Test script for ACE-Step BentoML service
"""

import requests
import json
import time
import sys
from pathlib import Path

# Configuration
BASE_URL = "http://localhost:3000"
API_BASE_URL = f"{BASE_URL}/v1"

def test_health():
    """Test health endpoint"""
    print("🔍 Testing health endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=10)
        if response.status_code == 200:
            print("✅ Health check passed")
            print(f"   Response: {response.json()}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False

def test_models():
    """Test models endpoint"""
    print("\n🔍 Testing models endpoint...")
    try:
        response = requests.get(f"{API_BASE_URL}/models", timeout=10)
        if response.status_code == 200:
            print("✅ Models endpoint working")
            models = response.json()
            print(f"   Available models: {[m['id'] for m in models['data']]}")
            return True
        else:
            print(f"❌ Models endpoint failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Models endpoint error: {e}")
        return False

def test_audio_generation():
    """Test audio generation endpoint"""
    print("\n🔍 Testing audio generation...")
    
    # Test data
    test_request = {
        "model": "ace-step-v1",
        "prompt": "A beautiful piano melody",
        "lyrics": "",
        "duration": 10.0,
        "guidance_scale": 15.0,
        "num_inference_steps": 20,  # Reduced for faster testing
        "scheduler": "euler",
        "cfg_type": "apg",
        "omega_scale": 10.0,
        "response_format": "wav"
    }
    
    try:
        print(f"   Sending request: {test_request['prompt']}")
        response = requests.post(
            f"{API_BASE_URL}/audio/generations",
            json=test_request,
            timeout=300  # 5 minutes timeout for generation
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Audio generation successful!")
            print(f"   Generation ID: {result['id']}")
            print(f"   Model: {result['model']}")
            print(f"   Created: {result['created']}")
            
            # Check if audio file URL is provided
            if result['data'] and len(result['data']) > 0:
                audio_data = result['data'][0]
                if 'url' in audio_data:
                    print(f"   Audio URL: {audio_data['url']}")
                    
                    # Test downloading the audio file
                    audio_url = f"{BASE_URL}{audio_data['url']}"
                    print(f"   Testing audio download from: {audio_url}")
                    
                    audio_response = requests.get(audio_url, timeout=30)
                    if audio_response.status_code == 200:
                        print("✅ Audio file download successful!")
                        print(f"   Content-Type: {audio_response.headers.get('content-type', 'unknown')}")
                        print(f"   Content-Length: {len(audio_response.content)} bytes")
                    else:
                        print(f"❌ Audio file download failed: {audio_response.status_code}")
                else:
                    print("⚠️  No audio URL in response")
            else:
                print("⚠️  No audio data in response")
            
            return True
        else:
            print(f"❌ Audio generation failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except requests.exceptions.Timeout:
        print("❌ Audio generation timed out (this might be normal for first run)")
        return False
    except Exception as e:
        print(f"❌ Audio generation error: {e}")
        return False

def test_bentoml_api():
    """Test BentoML native API"""
    print("\n🔍 Testing BentoML native API...")
    
    try:
        # Test the native BentoML API endpoint
        response = requests.post(
            f"{BASE_URL}/generate_audio",
            json={
                "prompt": "A simple test melody",
                "duration": 5.0,
                "guidance_scale": 15.0,
                "num_inference_steps": 10
            },
            timeout=120
        )
        
        if response.status_code == 200:
            print("✅ BentoML native API working")
            return True
        else:
            print(f"❌ BentoML native API failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ BentoML native API error: {e}")
        return False

def main():
    """Main test function"""
    print("🧪 ACE-Step BentoML Service Test Suite")
    print("=" * 50)
    
    # Wait a bit for service to start
    print("⏳ Waiting for service to start...")
    time.sleep(5)
    
    tests = [
        ("Health Check", test_health),
        ("Models Endpoint", test_models),
        ("BentoML Native API", test_bentoml_api),
        ("Audio Generation", test_audio_generation),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ Test {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*50)
    print("📊 TEST SUMMARY")
    print("="*50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        if result:
            passed += 1
    
    print(f"\nResults: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Service is working correctly.")
        sys.exit(0)
    else:
        print("⚠️  Some tests failed. Check the logs above.")
        sys.exit(1)

if __name__ == "__main__":
    main()