#!/usr/bin/env python3
"""
Quick test script for ACE-Step BentoML service
"""

import requests
import json
import time

def test_service():
    """Test the ACE-Step service"""
    base_url = "http://localhost:3000"
    
    print("🧪 Testing ACE-Step BentoML Service")
    print("=" * 50)
    
    # Test 1: Health check
    print("1. Testing health endpoint...")
    try:
        response = requests.get(f"{base_url}/health", timeout=10)
        if response.status_code == 200:
            print("✅ Health check passed")
            print(f"   Response: {response.json()}")
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False
    
    # Test 2: Models endpoint
    print("\n2. Testing models endpoint...")
    try:
        response = requests.get(f"{base_url}/v1/models", timeout=10)
        if response.status_code == 200:
            print("✅ Models endpoint working")
            models = response.json()
            print(f"   Available models: {[m['id'] for m in models['data']]}")
        else:
            print(f"❌ Models endpoint failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Models endpoint error: {e}")
        return False
    
    # Test 3: Simple audio generation (with reduced parameters for faster testing)
    print("\n3. Testing audio generation...")
    test_request = {
        "model": "ace-step-v1",
        "prompt": "A simple test melody",
        "duration": 5.0,  # Very short for testing
        "guidance_scale": 15.0,
        "num_inference_steps": 10,  # Reduced for faster testing
        "scheduler": "euler",
        "cfg_type": "apg",
        "omega_scale": 10.0,
        "response_format": "wav"
    }
    
    try:
        print(f"   Sending request: {test_request['prompt']}")
        response = requests.post(
            f"{base_url}/v1/audio/generations",
            json=test_request,
            timeout=120  # 2 minutes timeout
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Audio generation successful!")
            print(f"   Generation ID: {result['id']}")
            print(f"   Model: {result['model']}")
            
            if result['data'] and len(result['data']) > 0:
                audio_data = result['data'][0]
                if 'url' in audio_data:
                    print(f"   Audio URL: {audio_data['url']}")
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

if __name__ == "__main__":
    print("⏳ Waiting for service to start...")
    time.sleep(5)
    
    success = test_service()
    
    if success:
        print("\n🎉 All tests passed! Service is working correctly.")
    else:
        print("\n⚠️  Some tests failed. Check the logs above.")
