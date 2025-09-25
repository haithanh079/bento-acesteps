#!/usr/bin/env python3
"""
Example client for the OpenAI Speech API compatible streaming endpoint
"""

import requests
import json
import base64
import os
from pathlib import Path

class ACEStepStreamingClient:
    """Client for ACE-Step streaming audio generation"""
    
    def __init__(self, base_url="http://localhost:3000"):
        self.base_url = base_url
        self.speech_endpoint = f"{base_url}/v1/audio/speech"
        self.generations_endpoint = f"{base_url}/v1/audio/generations"
    
    def generate_speech_stream(self, text, **kwargs):
        """
        Generate speech/music from text using streaming endpoint
        
        Args:
            text (str): Text prompt for generation
            **kwargs: Additional parameters (duration, guidance_scale, etc.)
        
        Returns:
            bytes: Generated audio data
        """
        
        # Default parameters
        params = {
            "model": "ace-step-v1",
            "input": text,
            "response_format": kwargs.get("response_format", "mp3"),
            "duration": kwargs.get("duration", 10.0),
            "guidance_scale": kwargs.get("guidance_scale", 15.0),
            "num_inference_steps": kwargs.get("num_inference_steps", 30),
            "lyrics": kwargs.get("lyrics", ""),
            "seed": kwargs.get("seed", None),
            "scheduler": kwargs.get("scheduler", "euler"),
            "cfg_type": kwargs.get("cfg_type", "apg"),
            "omega_scale": kwargs.get("omega_scale", 10.0)
        }
        
        print(f"Generating audio for: '{text}'")
        print(f"Parameters: {json.dumps(params, indent=2)}")
        
        try:
            response = requests.post(
                self.speech_endpoint,
                json=params,
                stream=True,
                headers={
                    "Content-Type": "application/json",
                    "Accept": "text/plain"
                }
            )
            
            if response.status_code != 200:
                raise Exception(f"Request failed: {response.status_code} - {response.text}")
            
            # Process streaming response
            audio_chunks = []
            chunk_count = 0
            
            print("Receiving audio stream...")
            for line in response.iter_lines(decode_unicode=True):
                if line.startswith("data: "):
                    try:
                        data = json.loads(line[6:])  # Remove "data: " prefix
                        
                        if "error" in data:
                            raise Exception(f"Stream error: {data['error']}")
                        
                        if "data" in data and "audio" in data["data"]:
                            audio_chunk = data["data"]["audio"]
                            if audio_chunk:  # Non-empty chunk
                                audio_chunks.append(audio_chunk)
                                chunk_count += 1
                                print(f"  Received chunk {chunk_count}: {len(audio_chunk)} characters")
                            else:
                                print("  End of stream")
                                break
                                
                    except json.JSONDecodeError as e:
                        print(f"Failed to parse JSON: {e}")
                        print(f"Raw line: {line}")
            
            # Combine all chunks
            if audio_chunks:
                combined_base64 = "".join(audio_chunks)
                audio_data = base64.b64decode(combined_base64)
                print(f"Total audio size: {len(audio_data)} bytes")
                return audio_data
            else:
                raise Exception("No audio chunks received")
                
        except Exception as e:
            print(f"Error: {e}")
            return None
    
    def generate_speech_file(self, text, output_path, **kwargs):
        """
        Generate speech/music and save to file
        
        Args:
            text (str): Text prompt for generation
            output_path (str): Path to save the audio file
            **kwargs: Additional parameters
        """
        audio_data = self.generate_speech_stream(text, **kwargs)
        
        if audio_data:
            with open(output_path, "wb") as f:
                f.write(audio_data)
            print(f"Audio saved to: {output_path}")
            return True
        else:
            print("Failed to generate audio")
            return False
    
    def generate_regular(self, text, **kwargs):
        """
        Generate audio using the regular (non-streaming) endpoint
        
        Args:
            text (str): Text prompt for generation
            **kwargs: Additional parameters
        
        Returns:
            dict: Response from the regular endpoint
        """
        params = {
            "model": "ace-step-v1",
            "prompt": text,
            "response_format": kwargs.get("response_format", "mp3"),
            "duration": kwargs.get("duration", 10.0),
            "guidance_scale": kwargs.get("guidance_scale", 15.0),
            "num_inference_steps": kwargs.get("num_inference_steps", 30),
            "lyrics": kwargs.get("lyrics", ""),
            "seed": kwargs.get("seed", None),
            "scheduler": kwargs.get("scheduler", "euler"),
            "cfg_type": kwargs.get("cfg_type", "apg"),
            "omega_scale": kwargs.get("omega_scale", 10.0)
        }
        
        try:
            response = requests.post(self.generations_endpoint, json=params)
            
            if response.status_code == 200:
                return response.json()
            else:
                print(f"Regular endpoint error: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            print(f"Regular endpoint failed: {e}")
            return None

def main():
    """Example usage"""
    
    # Initialize client
    client = ACEStepStreamingClient()
    
    # Test prompts
    prompts = [
        "Generate a happy upbeat electronic music track",
        "Create a relaxing ambient soundscape",
        "Make a energetic rock song"
    ]
    
    print("ACE-Step Streaming Client Example")
    print("=" * 40)
    
    for i, prompt in enumerate(prompts, 1):
        print(f"\n--- Test {i}: {prompt} ---")
        
        # Test streaming generation
        output_file = f"test_streaming_{i}.mp3"
        success = client.generate_speech_file(
            prompt,
            output_file,
            duration=5.0,  # Shorter for testing
            num_inference_steps=20
        )
        
        if success:
            print(f"✓ Streaming generation successful: {output_file}")
        else:
            print("✗ Streaming generation failed")
    
    print("\n" + "=" * 40)
    print("Example completed!")

if __name__ == "__main__":
    main()
