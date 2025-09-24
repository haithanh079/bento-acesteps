#!/usr/bin/env python3
"""
Example client for ACE-Step OpenAI-compatible BentoML service
"""

import requests
import json
import time
import argparse
from pathlib import Path


class ACEStepClient:
    """Client for ACE-Step audio generation service"""
    
    def __init__(self, base_url: str = "http://localhost:3000"):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
    
    def generate_audio(
        self,
        prompt: str,
        lyrics: str = "",
        duration: float = 30.0,
        guidance_scale: float = 15.0,
        num_inference_steps: int = 60,
        seed: int = None,
        scheduler: str = "euler",
        cfg_type: str = "apg",
        omega_scale: float = 10.0,
        response_format: str = "wav"
    ) -> dict:
        """Generate audio from text prompt"""
        
        payload = {
            "model": "ace-step-v1",
            "prompt": prompt,
            "lyrics": lyrics,
            "duration": duration,
            "guidance_scale": guidance_scale,
            "num_inference_steps": num_inference_steps,
            "scheduler": scheduler,
            "cfg_type": cfg_type,
            "omega_scale": omega_scale,
            "response_format": response_format
        }
        
        if seed is not None:
            payload["seed"] = seed
        
        print(f"Generating audio with prompt: '{prompt}'")
        print(f"Parameters: {json.dumps(payload, indent=2)}")
        
        response = self.session.post(
            f"{self.base_url}/v1/audio/generations",
            headers={"Content-Type": "application/json"},
            json=payload
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"API Error {response.status_code}: {response.text}")
    
    def download_audio(self, audio_url: str, output_path: str) -> None:
        """Download generated audio file"""
        
        full_url = f"{self.base_url}{audio_url}"
        print(f"Downloading audio from: {full_url}")
        
        response = self.session.get(full_url)
        
        if response.status_code == 200:
            with open(output_path, "wb") as f:
                f.write(response.content)
            print(f"Audio saved to: {output_path}")
        else:
            raise Exception(f"Download Error {response.status_code}: {response.text}")
    
    def list_models(self) -> dict:
        """List available models"""
        response = self.session.get(f"{self.base_url}/v1/models")
        
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"API Error {response.status_code}: {response.text}")
    
    def health_check(self) -> dict:
        """Check service health"""
        response = self.session.get(f"{self.base_url}/v1/health")
        
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Health Check Error {response.status_code}: {response.text}")
    
    def delete_audio(self, filename: str) -> dict:
        """Delete generated audio file"""
        response = self.session.delete(f"{self.base_url}/v1/audio/files/{filename}")
        
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Delete Error {response.status_code}: {response.text}")


def main():
    parser = argparse.ArgumentParser(description="ACE-Step Audio Generation Client")
    parser.add_argument("--url", default="http://localhost:3000", help="Service URL")
    parser.add_argument("--prompt", required=True, help="Text prompt for audio generation")
    parser.add_argument("--lyrics", default="", help="Lyrics for the audio")
    parser.add_argument("--duration", type=float, default=30.0, help="Audio duration in seconds")
    parser.add_argument("--guidance-scale", type=float, default=15.0, help="Guidance scale")
    parser.add_argument("--steps", type=int, default=60, help="Number of inference steps")
    parser.add_argument("--seed", type=int, help="Random seed")
    parser.add_argument("--scheduler", default="euler", choices=["euler", "heun", "pingpong"], help="Scheduler type")
    parser.add_argument("--cfg-type", default="apg", choices=["apg", "cfg", "cfg_star"], help="CFG type")
    parser.add_argument("--omega-scale", type=float, default=10.0, help="Omega scale")
    parser.add_argument("--format", default="wav", choices=["wav", "mp3", "ogg"], help="Audio format")
    parser.add_argument("--output", default="generated_audio.wav", help="Output file path")
    parser.add_argument("--health-check", action="store_true", help="Just check service health")
    parser.add_argument("--list-models", action="store_true", help="List available models")
    
    args = parser.parse_args()
    
    client = ACEStepClient(args.url)
    
    try:
        if args.health_check:
            health = client.health_check()
            print("Service Health:", json.dumps(health, indent=2))
            return
        
        if args.list_models:
            models = client.list_models()
            print("Available Models:", json.dumps(models, indent=2))
            return
        
        # Generate audio
        start_time = time.time()
        
        result = client.generate_audio(
            prompt=args.prompt,
            lyrics=args.lyrics,
            duration=args.duration,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.steps,
            seed=args.seed,
            scheduler=args.scheduler,
            cfg_type=args.cfg_type,
            omega_scale=args.omega_scale,
            response_format=args.format
        )
        
        generation_time = time.time() - start_time
        print(f"Generation completed in {generation_time:.2f} seconds")
        print("Result:", json.dumps(result, indent=2))
        
        # Download the audio file
        audio_url = result["data"][0]["url"]
        client.download_audio(audio_url, args.output)
        
        print(f"Audio generation complete! File saved as: {args.output}")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
