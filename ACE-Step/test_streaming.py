#!/usr/bin/env python3
"""
Test script for the new OpenAI Speech API compatible streaming endpoint
"""

import requests
import json
import base64
import os
from pathlib import Path

def test_speech_streaming():
    """Test the new streaming speech endpoint"""
    
    # Service URL (adjust based on your deployment)
    base_url = "http://localhost:3000"  # Default BentoML port
    endpoint = f"{base_url}/v1/audio/speech"
    
    # Test request payload
    payload = {
        "model": "ace-step-v1",
        "input": "Generate a happy upbeat electronic music track",
        "response_format": "mp3",
        "duration": 10.0,
        "guidance_scale": 15.0,
        "num_inference_steps": 30
    }
    
    print(f"Testing streaming endpoint: {endpoint}")
    print(f"Request payload: {json.dumps(payload, indent=2)}")
    
    try:
        # Make streaming request
        response = requests.post(
            endpoint,
            json=payload,
            stream=True,
            headers={
                "Content-Type": "application/json",
                "Accept": "text/plain"
            }
        )
        
        if response.status_code != 200:
            print(f"Error: {response.status_code} - {response.text}")
            return
        
        print("\nStreaming response:")
        print("=" * 50)
        
        # Process streaming response
        audio_chunks = []
        for line in response.iter_lines(decode_unicode=True):
            if line.startswith("data: "):
                try:
                    data = json.loads(line[6:])  # Remove "data: " prefix
                    
                    if "error" in data:
                        print(f"Error in stream: {data['error']}")
                        break
                    
                    if "data" in data and "audio" in data["data"]:
                        audio_chunk = data["data"]["audio"]
                        if audio_chunk:  # Non-empty chunk
                            audio_chunks.append(audio_chunk)
                            print(f"Received chunk: {len(audio_chunk)} characters")
                        else:
                            print("End of stream")
                            break
                            
                except json.JSONDecodeError as e:
                    print(f"Failed to parse JSON: {e}")
                    print(f"Raw line: {line}")
        
        # Combine all chunks and save audio
        if audio_chunks:
            combined_base64 = "".join(audio_chunks)
            audio_data = base64.b64decode(combined_base64)
            
            # Save to file
            output_path = Path("test_streaming_output.mp3")
            with open(output_path, "wb") as f:
                f.write(audio_data)
            
            print(f"\nAudio saved to: {output_path}")
            print(f"Total audio size: {len(audio_data)} bytes")
        else:
            print("No audio chunks received")
            
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

def test_regular_endpoint():
    """Test the regular (non-streaming) endpoint for comparison"""
    
    base_url = "http://localhost:3000"
    endpoint = f"{base_url}/v1/audio/generations"
    
    payload = {
        "model": "ace-step-v1",
        "prompt": "Generate a happy upbeat electronic music track",
        "response_format": "mp3",
        "duration": 10.0,
        "guidance_scale": 15.0,
        "num_inference_steps": 30
    }
    
    print(f"\nTesting regular endpoint: {endpoint}")
    
    try:
        response = requests.post(endpoint, json=payload)
        
        if response.status_code == 200:
            result = response.json()
            print(f"Regular endpoint response: {json.dumps(result, indent=2)}")
        else:
            print(f"Regular endpoint error: {response.status_code} - {response.text}")
            
    except Exception as e:
        print(f"Regular endpoint test failed: {e}")

if __name__ == "__main__":
    print("OpenAI Speech API Streaming Test")
    print("=" * 40)
    
    # Test streaming endpoint
    test_speech_streaming()
    
    # Test regular endpoint for comparison
    test_regular_endpoint()
    
    print("\nTest completed!")
