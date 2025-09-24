"""
OpenAI-Compatible Client Example for ACE-Steps Service
Demonstrates how to use the service with OpenAI-compatible clients
"""

import asyncio
import json
import time
from typing import Dict, Any, Optional
import httpx
import requests
import base64

class OpenAICompatibleClient:
    """
    OpenAI-compatible client for ACE-Steps service
    """
    
    def __init__(self, base_url: str = "http://localhost:3000", api_key: str = "sk-ace-steps"):
        self.base_url = base_url
        self.api_key = api_key
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        })
    
    def create_speech(
        self,
        model: str = "ace-step-audio",
        input_text: str = "Hello, this is a test of the ACE-Steps audio generation service.",
        voice: str = "alloy",
        response_format: str = "mp3",
        speed: float = 1.0,
        duration: int = 30,
        guidance_scale: float = 15.0,
        infer_steps: int = 60
    ) -> Dict[str, Any]:
        """
        Create speech using OpenAI-compatible API
        
        Args:
            model: Model to use (ace-step-audio)
            input_text: Text to convert to speech
            voice: Voice to use (OpenAI compatible)
            response_format: Audio format (mp3, wav, flac)
            speed: Speech speed multiplier
            duration: Duration in seconds
            guidance_scale: Generation guidance scale
            infer_steps: Number of inference steps
        """
        try:
            payload = {
                "model": model,
                "input": input_text,
                "voice": voice,
                "response_format": response_format,
                "speed": speed,
                "duration": duration,
                "guidance_scale": guidance_scale,
                "infer_steps": infer_steps
            }
            
            response = self.session.post(f"{self.base_url}/v1/audio/speech", json=payload)
            response.raise_for_status()
            return response.json()
            
        except Exception as e:
            return {"error": str(e)}
    
    def create_music(
        self,
        model: str = "ace-step-music",
        prompt: str = "A cheerful pop song",
        lyrics: str = "",
        style: str = "pop",
        response_format: str = "mp3",
        duration: int = 30,
        guidance_scale: float = 15.0,
        infer_steps: int = 60
    ) -> Dict[str, Any]:
        """
        Create music using OpenAI-compatible API
        
        Args:
            model: Model to use (ace-step-music)
            prompt: Music description
            lyrics: Optional lyrics
            style: Music style/genre
            response_format: Audio format
            duration: Duration in seconds
            guidance_scale: Generation guidance scale
            infer_steps: Number of inference steps
        """
        try:
            payload = {
                "model": model,
                "prompt": prompt,
                "lyrics": lyrics,
                "style": style,
                "response_format": response_format,
                "duration": duration,
                "guidance_scale": guidance_scale,
                "infer_steps": infer_steps
            }
            
            response = self.session.post(f"{self.base_url}/v1/audio/music", json=payload)
            response.raise_for_status()
            return response.json()
            
        except Exception as e:
            return {"error": str(e)}
    
    def list_models(self) -> Dict[str, Any]:
        """List available models"""
        try:
            response = self.session.get(f"{self.base_url}/v1/models")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def get_model(self, model_id: str) -> Dict[str, Any]:
        """Get specific model information"""
        try:
            response = self.session.get(f"{self.base_url}/v1/models/{model_id}")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def health_check(self) -> Dict[str, Any]:
        """Check service health"""
        try:
            response = self.session.get(f"{self.base_url}/health")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def save_audio(self, response: Dict[str, Any], filename: str) -> bool:
        """
        Save audio from response to file
        
        Args:
            response: API response containing audio data
            filename: Output filename
        """
        try:
            if "data" in response and len(response["data"]) > 0:
                audio_data = response["data"][0].get("audio", "")
                if audio_data:
                    # Decode base64 audio data
                    audio_bytes = base64.b64decode(audio_data)
                    
                    # Save to file
                    with open(filename, "wb") as f:
                        f.write(audio_bytes)
                    
                    print(f"Audio saved to: {filename}")
                    return True
            return False
        except Exception as e:
            print(f"Failed to save audio: {str(e)}")
            return False

async def async_client_example():
    """Example using async HTTP client"""
    async with httpx.AsyncClient() as client:
        # Health check
        response = await client.get("http://localhost:3000/health")
        print("Health check:", response.json())
        
        # Create speech
        speech_request = {
            "model": "ace-step-audio",
            "input": "Hello from ACE-Steps OpenAI-compatible service!",
            "voice": "alloy",
            "response_format": "mp3",
            "duration": 15,
            "infer_steps": 40
        }
        
        response = await client.post("http://localhost:3000/v1/audio/speech", json=speech_request)
        result = response.json()
        print("Speech generation result:", result)

def main():
    """Main example function"""
    print("ACE-Steps OpenAI-Compatible Service Client Example")
    print("=" * 60)
    
    # Initialize client
    client = OpenAICompatibleClient()
    
    # Check service health
    print("1. Checking service health...")
    health = client.health_check()
    print(f"Health status: {health}")
    
    if "error" in health:
        print("Service is not available. Please start the service first.")
        return
    
    # List available models
    print("\n2. Listing available models...")
    models = client.list_models()
    if "error" not in models:
        print("Available models:")
        for model in models.get("data", []):
            print(f"  - {model['id']}: {model.get('capabilities', [])}")
    else:
        print(f"Failed to list models: {models['error']}")
    
    # Example 1: Generate speech
    print("\n3. Generating speech...")
    speech_result = client.create_speech(
        input_text="Welcome to the ACE-Steps audio generation service. This is a demonstration of OpenAI-compatible speech synthesis.",
        duration=20,
        infer_steps=40,
        guidance_scale=12.0
    )
    
    if "error" in speech_result:
        print(f"Speech generation failed: {speech_result['error']}")
    else:
        print(f"Speech generated successfully!")
        print(f"Model: {speech_result['model']}")
        print(f"Duration: {speech_result['data'][0]['duration']} seconds")
        print(f"Generation time: {speech_result['usage']['generation_time']:.2f} seconds")
        
        # Save audio
        if client.save_audio(speech_result, "openai_speech.mp3"):
            print("Audio saved as 'openai_speech.mp3'")
    
    # Example 2: Generate music
    print("\n4. Generating music...")
    music_result = client.create_music(
        prompt="A peaceful acoustic guitar melody",
        style="acoustic",
        duration=25,
        infer_steps=50,
        guidance_scale=15.0
    )
    
    if "error" in music_result:
        print(f"Music generation failed: {music_result['error']}")
    else:
        print(f"Music generated successfully!")
        print(f"Model: {music_result['model']}")
        print(f"Style: {music_result['data'][0]['style']}")
        print(f"Duration: {music_result['data'][0]['duration']} seconds")
        print(f"Generation time: {music_result['usage']['generation_time']:.2f} seconds")
        
        # Save audio
        if client.save_audio(music_result, "openai_music.mp3"):
            print("Audio saved as 'openai_music.mp3'")
    
    # Example 3: Generate music with lyrics
    print("\n5. Generating music with lyrics...")
    lyrics_music_result = client.create_music(
        prompt="A romantic ballad",
        lyrics="In the moonlight we dance together, forever and ever, this moment will last forever...",
        style="ballad",
        duration=30,
        infer_steps=60,
        guidance_scale=18.0
    )
    
    if "error" in lyrics_music_result:
        print(f"Music with lyrics generation failed: {lyrics_music_result['error']}")
    else:
        print(f"Music with lyrics generated successfully!")
        print(f"Has lyrics: {lyrics_music_result['data'][0]['has_lyrics']}")
        print(f"Generation time: {lyrics_music_result['usage']['generation_time']:.2f} seconds")
        
        # Save audio
        if client.save_audio(lyrics_music_result, "openai_lyrics_music.mp3"):
            print("Audio saved as 'openai_lyrics_music.mp3'")
    
    # Example 4: Batch generation
    print("\n6. Batch generation example...")
    prompts = [
        "A happy jazz tune",
        "A sad piano melody",
        "An energetic rock song"
    ]
    
    results = []
    for i, prompt in enumerate(prompts):
        print(f"Generating {i+1}/3: {prompt}")
        result = client.create_music(
            prompt=prompt,
            style=["jazz", "piano", "rock"][i],
            duration=15,
            infer_steps=30,
            guidance_scale=12.0
        )
        results.append(result)
        
        if "error" not in result:
            print(f"  ✓ Generated in {result['usage']['generation_time']:.2f}s")
        else:
            print(f"  ✗ Failed: {result['error']}")
    
    # Summary
    successful = sum(1 for r in results if "error" not in r)
    print(f"\nBatch generation completed: {successful}/{len(results)} successful")

def test_openai_compatibility():
    """Test OpenAI API compatibility"""
    print("OpenAI Compatibility Test")
    print("=" * 40)
    
    client = OpenAICompatibleClient()
    
    # Test 1: Model listing (OpenAI format)
    print("1. Testing model listing...")
    models = client.list_models()
    if "error" not in models:
        print("✓ Model listing works")
        print(f"  Found {len(models.get('data', []))} models")
    else:
        print(f"✗ Model listing failed: {models['error']}")
    
    # Test 2: Model retrieval (OpenAI format)
    print("\n2. Testing model retrieval...")
    model_info = client.get_model("ace-step-audio")
    if "error" not in model_info:
        print("✓ Model retrieval works")
        print(f"  Model: {model_info['id']}")
        print(f"  Owner: {model_info['owned_by']}")
    else:
        print(f"✗ Model retrieval failed: {model_info['error']}")
    
    # Test 3: Speech generation (OpenAI format)
    print("\n3. Testing speech generation...")
    speech = client.create_speech(
        input_text="OpenAI compatibility test",
        duration=10,
        infer_steps=20
    )
    if "error" not in speech:
        print("✓ Speech generation works")
        print(f"  Response format: OpenAI compatible")
        print(f"  Has usage metrics: {'usage' in speech}")
    else:
        print(f"✗ Speech generation failed: {speech['error']}")
    
    # Test 4: Music generation (OpenAI format)
    print("\n4. Testing music generation...")
    music = client.create_music(
        prompt="Test music generation",
        duration=10,
        infer_steps=20
    )
    if "error" not in music:
        print("✓ Music generation works")
        print(f"  Response format: OpenAI compatible")
        print(f"  Has usage metrics: {'usage' in music}")
    else:
        print(f"✗ Music generation failed: {music['error']}")

def performance_test():
    """Performance testing with OpenAI-compatible API"""
    print("OpenAI-Compatible Performance Test")
    print("=" * 45)
    
    client = OpenAICompatibleClient()
    
    # Test parameters
    test_cases = [
        {"duration": 10, "infer_steps": 20, "name": "Fast (20 steps)"},
        {"duration": 20, "infer_steps": 40, "name": "Standard (40 steps)"},
        {"duration": 30, "infer_steps": 60, "name": "High Quality (60 steps)"}
    ]
    
    for test_case in test_cases:
        print(f"\nTesting {test_case['name']}...")
        
        start_time = time.time()
        result = client.create_speech(
            input_text="Performance test",
            duration=test_case["duration"],
            infer_steps=test_case["infer_steps"]
        )
        total_time = time.time() - start_time
        
        if "error" not in result:
            generation_time = result.get("usage", {}).get("generation_time", 0)
            print(f"  Total time: {total_time:.2f}s")
            print(f"  Generation time: {generation_time:.2f}s")
            print(f"  Overhead: {total_time - generation_time:.2f}s")
            print(f"  OpenAI compatible: ✓")
        else:
            print(f"  Failed: {result['error']}")

if __name__ == "__main__":
    # Run main example
    main()
    
    # Uncomment to run compatibility test
    # test_openai_compatibility()
    
    # Uncomment to run performance test
    # performance_test()
    
    # Uncomment to run async example
    # asyncio.run(async_client_example())
