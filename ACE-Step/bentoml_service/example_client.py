"""
Example client for ACE-Steps BentoML Service
Demonstrates how to use the service API
"""

import asyncio
import json
import time
from typing import Dict, Any
import httpx
import requests

class ACEStepClient:
    """
    Client for ACE-Steps BentoML Service
    """
    
    def __init__(self, base_url: str = "http://localhost:3000"):
        self.base_url = base_url
        self.session = requests.Session()
    
    def health_check(self) -> Dict[str, Any]:
        """Check service health"""
        try:
            response = self.session.get(f"{self.base_url}/health")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def generate_music(
        self, 
        prompt: str, 
        duration: int = 30,
        infer_steps: int = 60,
        guidance_scale: float = 15.0,
        omega_scale: float = 10.0,
        seed: int = None
    ) -> Dict[str, Any]:
        """Generate music from text prompt"""
        try:
            payload = {
                "prompt": prompt,
                "duration": duration,
                "infer_steps": infer_steps,
                "guidance_scale": guidance_scale,
                "omega_scale": omega_scale,
                "seed": seed
            }
            
            response = self.session.post(f"{self.base_url}/music", json=payload)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def generate_audio(
        self,
        audio_duration: float,
        prompt: str = "",
        lyrics: str = "",
        infer_step: int = 60,
        guidance_scale: float = 15.0,
        scheduler_type: str = "euler",
        cfg_type: str = "classifier_free",
        omega_scale: float = 10.0,
        actual_seeds: list = None,
        output_format: str = "wav"
    ) -> Dict[str, Any]:
        """Generate audio with full parameter control"""
        try:
            payload = {
                "audio_duration": audio_duration,
                "prompt": prompt,
                "lyrics": lyrics,
                "infer_step": infer_step,
                "guidance_scale": guidance_scale,
                "scheduler_type": scheduler_type,
                "cfg_type": cfg_type,
                "omega_scale": omega_scale,
                "actual_seeds": actual_seeds or [],
                "guidance_interval": 0.1,
                "guidance_interval_decay": 0.95,
                "min_guidance_scale": 1.0,
                "use_erg_tag": False,
                "use_erg_lyric": False,
                "use_erg_diffusion": False,
                "oss_steps": [],
                "guidance_scale_text": 0.0,
                "guidance_scale_lyric": 0.0,
                "output_format": output_format
            }
            
            response = self.session.post(f"{self.base_url}/generate", json=payload)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def download_audio(self, file_path: str) -> bytes:
        """Download generated audio file"""
        try:
            payload = {"file_path": file_path}
            response = self.session.post(f"{self.base_url}/download", json=payload)
            response.raise_for_status()
            return response.content
        except Exception as e:
            raise Exception(f"Download failed: {str(e)}")

async def async_client_example():
    """Example using async HTTP client"""
    async with httpx.AsyncClient() as client:
        # Health check
        response = await client.get("http://localhost:3000/health")
        print("Health check:", response.json())
        
        # Generate music
        music_request = {
            "prompt": "A cheerful pop song with electronic elements",
            "duration": 20,
            "infer_steps": 40,
            "guidance_scale": 12.0,
            "seed": 123
        }
        
        response = await client.post("http://localhost:3000/music", json=music_request)
        result = response.json()
        print("Music generation result:", result)

def main():
    """Main example function"""
    print("ACE-Steps BentoML Service Client Example")
    print("=" * 50)
    
    # Initialize client
    client = ACEStepClient()
    
    # Check service health
    print("1. Checking service health...")
    health = client.health_check()
    print(f"Health status: {health}")
    
    if "error" in health:
        print("Service is not available. Please start the service first.")
        return
    
    # Example 1: Generate music from text
    print("\n2. Generating music from text...")
    music_result = client.generate_music(
        prompt="A peaceful acoustic guitar melody",
        duration=15,
        infer_steps=40,
        guidance_scale=12.0,
        seed=42
    )
    
    if "error" in music_result:
        print(f"Music generation failed: {music_result['error']}")
    else:
        print(f"Music generated successfully!")
        print(f"Output path: {music_result['audio_path']}")
        print(f"Generation time: {music_result['generation_time']:.2f} seconds")
    
    # Example 2: Generate audio with lyrics
    print("\n3. Generating audio with lyrics...")
    audio_result = client.generate_audio(
        audio_duration=20.0,
        prompt="A romantic ballad",
        lyrics="In the moonlight we dance together, forever and ever...",
        infer_step=50,
        guidance_scale=15.0,
        scheduler_type="euler",
        output_format="wav"
    )
    
    if "error" in audio_result:
        print(f"Audio generation failed: {audio_result['error']}")
    else:
        print(f"Audio generated successfully!")
        print(f"Output path: {audio_result['output_path']}")
        print(f"Generation time: {audio_result['generation_time']:.2f} seconds")
        print(f"Metadata: {audio_result['metadata']}")
    
    # Example 3: Batch generation
    print("\n4. Batch generation example...")
    prompts = [
        "A happy jazz tune",
        "A sad piano melody", 
        "An energetic rock song"
    ]
    
    results = []
    for i, prompt in enumerate(prompts):
        print(f"Generating {i+1}/3: {prompt}")
        result = client.generate_music(
            prompt=prompt,
            duration=10,
            infer_steps=30,
            seed=100 + i
        )
        results.append(result)
        
        if "error" not in result:
            print(f"  ✓ Generated in {result['generation_time']:.2f}s")
        else:
            print(f"  ✗ Failed: {result['error']}")
    
    # Summary
    successful = sum(1 for r in results if "error" not in r)
    print(f"\nBatch generation completed: {successful}/{len(results)} successful")
    
    # Example 4: Download audio file
    if results and "error" not in results[0]:
        print("\n5. Downloading audio file...")
        try:
            audio_data = client.download_audio(results[0]["audio_path"])
            print(f"Downloaded {len(audio_data)} bytes")
            
            # Save to local file
            with open("downloaded_audio.wav", "wb") as f:
                f.write(audio_data)
            print("Audio saved as 'downloaded_audio.wav'")
            
        except Exception as e:
            print(f"Download failed: {str(e)}")

def performance_test():
    """Performance testing example"""
    print("Performance Test")
    print("=" * 30)
    
    client = ACEStepClient()
    
    # Test parameters
    test_cases = [
        {"duration": 10, "infer_steps": 30, "name": "Fast (30 steps)"},
        {"duration": 20, "infer_steps": 60, "name": "Standard (60 steps)"},
        {"duration": 30, "infer_steps": 100, "name": "High Quality (100 steps)"}
    ]
    
    for test_case in test_cases:
        print(f"\nTesting {test_case['name']}...")
        
        start_time = time.time()
        result = client.generate_music(
            prompt="A test melody",
            duration=test_case["duration"],
            infer_steps=test_case["infer_steps"],
            seed=999
        )
        total_time = time.time() - start_time
        
        if "error" not in result:
            generation_time = result.get("generation_time", 0)
            print(f"  Total time: {total_time:.2f}s")
            print(f"  Generation time: {generation_time:.2f}s")
            print(f"  Overhead: {total_time - generation_time:.2f}s")
        else:
            print(f"  Failed: {result['error']}")

if __name__ == "__main__":
    # Run main example
    main()
    
    # Uncomment to run performance test
    # performance_test()
    
    # Uncomment to run async example
    # asyncio.run(async_client_example())
