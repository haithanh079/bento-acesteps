"""
OpenAI-Compatible BentoML Service for ACE-Step Audio Generation
"""

import os
import uuid
import time
import tempfile
import shutil
import asyncio
import base64
import json
from typing import List, Optional, Dict, Any, Union, AsyncGenerator
from pathlib import Path
from datetime import datetime, timedelta

import bentoml
from pydantic import BaseModel, Field, validator
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Depends, BackgroundTasks
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import torch

from acestep.pipeline_ace_step import ACEStepPipeline
from config import config

# Device detection and MPS fallback
def get_optimal_device():
    """Get the optimal device for inference"""
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        # Test MPS with problematic operations
        try:
            # Test MPS with large output channels (known limitation)
            test_tensor = torch.randn(1, 1, 1000, device='mps')
            conv = torch.nn.Conv1d(1, 100000, 3, device='mps')
            _ = conv(test_tensor)
            return "mps"
        except Exception as e:
            print(f"MPS test failed: {e}, falling back to CPU")
            return "cpu"
    else:
        return "cpu"

# Set device globally
DEVICE = get_optimal_device()
print(f"Using device: {DEVICE}")

# Override device in config if needed
if DEVICE != "cuda":
    config.device_id = 0  # Use CPU or MPS
    print(f"Device overridden to: {DEVICE}")

# Set environment variables for better compatibility
import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# OpenAI-compatible request/response models
class AudioGenerationRequest(BaseModel):
    """OpenAI-compatible audio generation request"""
    model: str = Field(default="ace-step-v1", description="Model to use for generation")
    prompt: str = Field(..., description="Text prompt for audio generation", min_length=1, max_length=1000)
    lyrics: Optional[str] = Field(default="", description="Lyrics for the audio", max_length=2000)
    duration: Optional[float] = Field(default=None, description="Duration of audio in seconds", ge=1.0)
    guidance_scale: Optional[float] = Field(default=15.0, description="Guidance scale for generation", ge=1.0, le=30.0)
    num_inference_steps: Optional[int] = Field(default=60, description="Number of inference steps", ge=10)
    seed: Optional[int] = Field(default=None, description="Random seed for reproducible generation")
    scheduler: Optional[str] = Field(default="euler", description="Scheduler type")
    cfg_type: Optional[str] = Field(default="apg", description="CFG type")
    omega_scale: Optional[float] = Field(default=10.0, description="Omega scale parameter", ge=1.0, le=20.0)
    response_format: Optional[str] = Field(default="wav", description="Audio format")
    temperature: Optional[float] = Field(default=1.0, description="Sampling temperature (not used in this model)")


class SpeechGenerationRequest(BaseModel):
    """OpenAI Speech API compatible request for streaming audio generation"""
    model: str = Field(default="ace-step-v1", description="Model to use for generation")
    input: str = Field(..., description="Text prompt for audio generation", min_length=1, max_length=1000)
    voice: Optional[str] = Field(default="alloy", description="Voice type (not used in music generation)")
    response_format: Optional[str] = Field(default="mp3", description="Audio format")
    speed: Optional[float] = Field(default=1.0, description="Speed of generation (not used in music generation)")
    # Additional ACE-Step specific parameters
    lyrics: Optional[str] = Field(default="", description="Lyrics for the audio", max_length=2000)
    duration: Optional[float] = Field(default=None, description="Duration of audio in seconds", ge=1.0)
    guidance_scale: Optional[float] = Field(default=15.0, description="Guidance scale for generation", ge=1.0, le=30.0)
    num_inference_steps: Optional[int] = Field(default=60, description="Number of inference steps", ge=10)
    seed: Optional[int] = Field(default=None, description="Random seed for reproducible generation")
    scheduler: Optional[str] = Field(default="euler", description="Scheduler type")
    cfg_type: Optional[str] = Field(default="apg", description="CFG type")
    omega_scale: Optional[float] = Field(default=10.0, description="Omega scale parameter", ge=1.0, le=20.0)
    
    @validator('duration')
    def validate_duration(cls, v):
        if v is None:
            return config.default_duration
        if v > config.max_duration:
            raise ValueError(f"Duration cannot exceed {config.max_duration} seconds")
        return v
    
    @validator('num_inference_steps')
    def validate_steps(cls, v):
        if v > config.max_inference_steps:
            raise ValueError(f"Inference steps cannot exceed {config.max_inference_steps}")
        return v
    
    @validator('scheduler')
    def validate_scheduler(cls, v):
        allowed = ["euler", "heun", "pingpong"]
        if v not in allowed:
            raise ValueError(f"Scheduler must be one of: {allowed}")
        return v
    
    @validator('cfg_type')
    def validate_cfg_type(cls, v):
        allowed = ["apg", "cfg", "cfg_star"]
        if v not in allowed:
            raise ValueError(f"CFG type must be one of: {allowed}")
        return v
    
    @validator('response_format')
    def validate_format(cls, v):
        allowed = ["wav", "mp3", "ogg"]
        if v not in allowed:
            raise ValueError(f"Response format must be one of: {allowed}")
        return v


class SpeechGenerationRequest(BaseModel):
    """OpenAI Speech API compatible request for streaming audio generation"""
    model: str = Field(default="ace-step-v1", description="Model to use for generation")
    input: str = Field(..., description="Text prompt for audio generation", min_length=1, max_length=1000)
    voice: Optional[str] = Field(default="alloy", description="Voice type (not used in music generation)")
    response_format: Optional[str] = Field(default="mp3", description="Audio format")
    speed: Optional[float] = Field(default=1.0, description="Speed of generation (not used in music generation)")
    # Additional ACE-Step specific parameters
    lyrics: Optional[str] = Field(default="", description="Lyrics for the audio", max_length=2000)
    duration: Optional[float] = Field(default=None, description="Duration of audio in seconds", ge=1.0)
    guidance_scale: Optional[float] = Field(default=15.0, description="Guidance scale for generation", ge=1.0, le=30.0)
    num_inference_steps: Optional[int] = Field(default=60, description="Number of inference steps", ge=10)
    seed: Optional[int] = Field(default=None, description="Random seed for reproducible generation")
    scheduler: Optional[str] = Field(default="euler", description="Scheduler type")
    cfg_type: Optional[str] = Field(default="apg", description="CFG type")
    omega_scale: Optional[float] = Field(default=10.0, description="Omega scale parameter", ge=1.0, le=20.0)
    
    @validator('duration')
    def validate_duration(cls, v):
        if v is None:
            return config.default_duration
        if v > config.max_duration:
            raise ValueError(f"Duration cannot exceed {config.max_duration} seconds")
        return v
    
    @validator('num_inference_steps')
    def validate_steps(cls, v):
        if v > config.max_inference_steps:
            raise ValueError(f"Inference steps cannot exceed {config.max_inference_steps}")
        return v
    
    @validator('scheduler')
    def validate_scheduler(cls, v):
        allowed = ["euler", "heun", "pingpong"]
        if v not in allowed:
            raise ValueError(f"Scheduler must be one of: {allowed}")
        return v
    
    @validator('cfg_type')
    def validate_cfg_type(cls, v):
        allowed = ["apg", "cfg", "cfg_star"]
        if v not in allowed:
            raise ValueError(f"CFG type must be one of: {allowed}")
        return v
    
    @validator('response_format')
    def validate_format(cls, v):
        allowed = ["wav", "mp3", "ogg"]
        if v not in allowed:
            raise ValueError(f"Response format must be one of: {allowed}")
        return v


class AudioGenerationResponse(BaseModel):
    """OpenAI-compatible audio generation response"""
    id: str = Field(..., description="Unique identifier for the generation")
    object: str = Field(default="audio.generation", description="Object type")
    created: int = Field(..., description="Unix timestamp of creation")
    model: str = Field(..., description="Model used for generation")
    data: List[Dict[str, Any]] = Field(..., description="Generated audio data")


class SpeechStreamChunk(BaseModel):
    """Streaming audio chunk for OpenAI Speech API format"""
    audio: str = Field(..., description="Base64 encoded audio chunk")


class SpeechStreamResponse(BaseModel):
    """Streaming response for OpenAI Speech API format"""
    id: str = Field(..., description="Unique identifier for the generation")
    object: str = Field(default="audio.speech.chunk", description="Object type")
    created: int = Field(..., description="Unix timestamp of creation")
    model: str = Field(..., description="Model used for generation")
    data: SpeechStreamChunk = Field(..., description="Audio chunk data")


class AudioData(BaseModel):
    """Audio data structure"""
    url: Optional[str] = Field(None, description="URL to download the audio file")
    b64_json: Optional[str] = Field(None, description="Base64 encoded audio data")
    revised_prompt: Optional[str] = Field(None, description="Revised prompt used for generation")


class ErrorResponse(BaseModel):
    """Error response model"""
    error: Dict[str, Any] = Field(..., description="Error details")


class ModelInfo(BaseModel):
    """Model information"""
    id: str
    object: str = "model"
    created: int
    owned_by: str = "ace-step"


class ModelsResponse(BaseModel):
    """Models list response"""
    object: str = "list"
    data: List[ModelInfo]


# File management utilities
class FileManager:
    """Manages generated audio files"""
    
    def __init__(self):
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def create_output_path(self, format: str = "wav") -> Path:
        """Create a unique output path for generated audio"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"audio_{timestamp}_{uuid.uuid4().hex[:8]}.{format}"
        return self.output_dir / filename
    
    def cleanup_old_files(self):
        """Remove files older than retention period"""
        if not config.cleanup_files:
            return
        
        cutoff_time = datetime.now() - timedelta(hours=config.file_retention_hours)
        
        for file_path in self.output_dir.glob("*"):
            if file_path.is_file():
                file_time = datetime.fromtimestamp(file_path.stat().st_mtime)
                if file_time < cutoff_time:
                    try:
                        file_path.unlink()
                        print(f"Cleaned up old file: {file_path}")
                    except Exception as e:
                        print(f"Failed to cleanup file {file_path}: {e}")
    
    def get_file_info(self, file_path: Path) -> Dict[str, Any]:
        """Get file information"""
        if not file_path.exists():
            return None
        
        stat = file_path.stat()
        return {
            "path": str(file_path),
            "size": stat.st_size,
            "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
            "modified": datetime.fromtimestamp(stat.st_mtime).isoformat()
        }


# Create FastAPI app for OpenAI-compatible endpoints
openai_api_app = FastAPI(
    title="ACE-Step OpenAI Compatible API",
    description="OpenAI-compatible API for ACE-Step audio generation",
    version="1.0.0"
)

# Add CORS middleware if enabled
if config.enable_cors:
    openai_api_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# Global file manager
file_manager = FileManager()


@bentoml.service(
    name="ace-step-audio-service",
    resources={
        "gpu": 1,
        "gpu_type": "nvidia-tesla-t4",  # Adjust based on your GPU
        "memory": "8Gi",
        "cpu": "4"
    },
    traffic={
        "timeout": 300,  # 5 minutes timeout for audio generation
        "concurrency": 2
    },
    envs=[
        {"name": "CUDA_VISIBLE_DEVICES", "value": str(config.device_id)},
        {"name": "TOKENIZERS_PARALLELISM", "value": "false"}
    ],
    # Enable metrics for monitoring
    metrics={
        "enabled": True,
        "namespace": "ace_step_service"
    },
    # Enable access logging
    logging={
        "access": {
            "enabled": True,
            "request_content_length": True,
            "request_content_type": True,
            "response_content_length": True,
            "response_content_type": True
        }
    }
)
@bentoml.asgi_app(openai_api_app, path="/v1")
class acestepaudoservice:
    """BentoML service for ACE-Step audio generation with OpenAI-compatible API"""
    
    def __init__(self):
        """Initialize the service"""
        self.pipeline = None
        self.model_loaded = False
        self.file_manager = FileManager()
        
        # Start background cleanup task
        asyncio.create_task(self._periodic_cleanup())
        
        # Initialize pipeline lazily
        self._initialize_pipeline()
    
    def _initialize_pipeline(self):
        """Initialize the ACE-Step pipeline"""
        try:
            # Use detected device
            device_id = 0 if DEVICE in ["cpu", "mps"] else config.device_id
            
            self.pipeline = ACEStepPipeline(
                checkpoint_dir=config.checkpoint_path,
                dtype=config.dtype,
                torch_compile=config.torch_compile,
                device_id=device_id,
                cpu_offload=config.cpu_offload,
                overlapped_decode=config.overlapped_decode
            )
            self.model_loaded = True
            print(f"ACE-Step pipeline initialized successfully on {DEVICE}")
        except Exception as e:
            print(f"Failed to initialize pipeline: {e}")
            # Try with CPU fallback
            try:
                print("Attempting CPU fallback...")
                self.pipeline = ACEStepPipeline(
                    checkpoint_dir=config.checkpoint_path,
                    dtype=config.dtype,
                    torch_compile=False,  # Disable torch compile for CPU
                    device_id=0,
                    cpu_offload=True,
                    overlapped_decode=False
                )
                self.model_loaded = True
                print("ACE-Step pipeline initialized successfully on CPU (fallback)")
            except Exception as e2:
                print(f"CPU fallback also failed: {e2}")
                self.model_loaded = False
    
    def _ensure_model_loaded(self):
        """Ensure the model is loaded"""
        if not self.model_loaded:
            self._initialize_pipeline()
        if not self.model_loaded:
            raise bentoml.exceptions.ServiceUnavailable("Model not available - check checkpoint path and GPU availability")
    
    async def _periodic_cleanup(self):
        """Periodic cleanup of old files"""
        while True:
            try:
                await asyncio.sleep(3600)  # Run every hour
                self.file_manager.cleanup_old_files()
            except Exception as e:
                print(f"Cleanup task error: {e}")
    
    def _prepare_generation_params(self, request: AudioGenerationRequest) -> Dict[str, Any]:
        """Prepare parameters for audio generation"""
        return {
            "audio_duration": request.duration,
            "prompt": request.prompt,
            "lyrics": request.lyrics or "",
            "infer_step": request.num_inference_steps,
            "guidance_scale": request.guidance_scale,
            "scheduler_type": request.scheduler,
            "cfg_type": request.cfg_type,
            "omega_scale": request.omega_scale,
            "manual_seeds": [request.seed] if request.seed is not None else [],
            "guidance_interval": 0.5,
            "guidance_interval_decay": 0.0,
            "min_guidance_scale": 3.0,
            "use_erg_tag": True,
            "use_erg_lyric": True,
            "use_erg_diffusion": True,
            "oss_steps": [],
            "guidance_scale_text": 0.0,
            "guidance_scale_lyric": 0.0,
            "format": request.response_format
        }
    
    @bentoml.api
    async def generate_audio(
        self,
        prompt: str = Field(..., description="Text prompt for audio generation"),
        lyrics: str = Field(default="", description="Lyrics for the audio"),
        duration: float = Field(default=30.0, description="Duration in seconds"),
        guidance_scale: float = Field(default=15.0, description="Guidance scale"),
        num_inference_steps: int = Field(default=60, description="Number of inference steps"),
        seed: Optional[int] = Field(default=None, description="Random seed"),
        scheduler: str = Field(default="euler", description="Scheduler type"),
        cfg_type: str = Field(default="apg", description="CFG type"),
        omega_scale: float = Field(default=10.0, description="Omega scale"),
        response_format: str = Field(default="wav", description="Audio format")
    ) -> Path:
        """Generate audio from text prompt - BentoML native API"""
        self._ensure_model_loaded()
        
        try:
            # Create output path using file manager
            output_path = self.file_manager.create_output_path(response_format)
            
            # Prepare generation parameters
            params = {
                "audio_duration": duration,
                "prompt": prompt,
                "lyrics": lyrics,
                "infer_step": num_inference_steps,
                "guidance_scale": guidance_scale,
                "scheduler_type": scheduler,
                "cfg_type": cfg_type,
                "omega_scale": omega_scale,
                "manual_seeds": [seed] if seed is not None else [],
                "guidance_interval": 0.5,
                "guidance_interval_decay": 0.0,
                "min_guidance_scale": 3.0,
                "use_erg_tag": True,
                "use_erg_lyric": True,
                "use_erg_diffusion": True,
                "oss_steps": [],
                "guidance_scale_text": 0.0,
                "guidance_scale_lyric": 0.0,
                "save_path": str(output_path),
                "format": response_format
            }
            
            # Run generation in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            result_paths = await loop.run_in_executor(None, lambda: self.pipeline(**params))
            
            # Return the first audio file path
            if result_paths and len(result_paths) > 0:
                return Path(result_paths[0])
            else:
                raise bentoml.exceptions.InternalServerError("Failed to generate audio - no output files created")
                
        except Exception as e:
            raise bentoml.exceptions.InternalServerError(f"Audio generation failed: {str(e)}")
    
    @bentoml.api
    async def generate_audio_from_request(self, request: AudioGenerationRequest) -> Path:
        """Generate audio from AudioGenerationRequest object"""
        return await self.generate_audio(
            prompt=request.prompt,
            lyrics=request.lyrics or "",
            duration=request.duration,
            guidance_scale=request.guidance_scale,
            num_inference_steps=request.num_inference_steps,
            seed=request.seed,
            scheduler=request.scheduler,
            cfg_type=request.cfg_type,
            omega_scale=request.omega_scale,
            response_format=request.response_format
        )
    
    async def generate_audio_stream(self, request: SpeechGenerationRequest) -> AsyncGenerator[str, None]:
        """Generate audio and stream as base64 chunks"""
        self._ensure_model_loaded()
        
        try:
            # Create output path using file manager
            output_path = self.file_manager.create_output_path(request.response_format)
            
            # Prepare generation parameters
            params = {
                "audio_duration": request.duration,
                "prompt": request.input,  # Use 'input' field for OpenAI compatibility
                "lyrics": request.lyrics or "",
                "infer_step": request.num_inference_steps,
                "guidance_scale": request.guidance_scale,
                "scheduler_type": request.scheduler,
                "cfg_type": request.cfg_type,
                "omega_scale": request.omega_scale,
                "manual_seeds": [request.seed] if request.seed is not None else [],
                "guidance_interval": 0.5,
                "guidance_interval_decay": 0.0,
                "min_guidance_scale": 3.0,
                "use_erg_tag": True,
                "use_erg_lyric": True,
                "use_erg_diffusion": True,
                "oss_steps": [],
                "guidance_scale_text": 0.0,
                "guidance_scale_lyric": 0.0,
                "save_path": str(output_path),
                "format": request.response_format
            }
            
            # Run generation in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            result_paths = await loop.run_in_executor(None, lambda: self.pipeline(**params))
            
            # Get the generated audio file
            if result_paths and len(result_paths) > 0:
                audio_path = Path(result_paths[0])
                
                # Read the audio file and stream as base64 chunks
                with open(audio_path, "rb") as audio_file:
                    audio_data = audio_file.read()
                    
                    # Split into chunks for streaming
                    chunk_size = 4096  # 4KB chunks
                    for i in range(0, len(audio_data), chunk_size):
                        chunk = audio_data[i:i + chunk_size]
                        base64_chunk = base64.b64encode(chunk).decode('utf-8')
                        
                        # Create streaming response
                        stream_response = SpeechStreamResponse(
                            id=f"audio_stream_{uuid.uuid4().hex}",
                            created=int(time.time()),
                            model=request.model,
                            data=SpeechStreamChunk(audio=base64_chunk)
                        )
                        
                        # Format as Server-Sent Events
                        yield f"data: {stream_response.json()}\n\n"
                        
                        # Small delay to simulate streaming
                        await asyncio.sleep(0.01)
                
                # Send final chunk to indicate completion
                final_response = SpeechStreamResponse(
                    id=f"audio_stream_{uuid.uuid4().hex}",
                    created=int(time.time()),
                    model=request.model,
                    data=SpeechStreamChunk(audio="")  # Empty chunk to indicate end
                )
                yield f"data: {final_response.json()}\n\n"
                
            else:
                raise bentoml.exceptions.InternalServerError("Failed to generate audio - no output files created")
                
        except Exception as e:
            # Send error in streaming format
            error_response = {
                "error": {
                    "message": f"Audio generation failed: {str(e)}",
                    "type": "server_error",
                    "code": 500
                }
            }
            yield f"data: {json.dumps(error_response)}\n\n"


# OpenAI-compatible endpoints
@openai_api_app.post("/audio/generations", response_model=AudioGenerationResponse)
async def create_audio_generation(
    request: AudioGenerationRequest,
    background_tasks: BackgroundTasks,
    service: acestepaudoservice = Depends(bentoml.get_current_service)
):
    """OpenAI-compatible audio generation endpoint"""
    try:
        # Generate audio using the service
        audio_path = await service.generate_audio_from_request(request)
        
        # Create response
        generation_id = f"audio_gen_{uuid.uuid4().hex}"
        created_timestamp = int(time.time())
        
        # Create audio data with file URL
        audio_data = AudioData(
            url=f"/v1/audio/files/{audio_path.name}",
            revised_prompt=request.prompt
        )
        
        response = AudioGenerationResponse(
            id=generation_id,
            created=created_timestamp,
            model=request.model,
            data=[audio_data.dict()]
        )
        
        # Schedule cleanup of the file after some time
        background_tasks.add_task(
            lambda: asyncio.create_task(
                asyncio.sleep(config.file_retention_hours * 3600)
            )
        )
        
        return response
        
    except ValueError as e:
        raise bentoml.exceptions.BadInput(str(e))
    except bentoml.exceptions.BentoMLException:
        raise  # Re-raise BentoML exceptions as-is
    except Exception as e:
        raise bentoml.exceptions.InternalServerError(f"Internal server error: {str(e)}")


@openai_api_app.post("/audio/generations/stream")
async def create_audio_generation_stream(
    request: AudioGenerationRequest,
    service: acestepaudoservice = Depends(bentoml.get_current_service)
):
    """Streaming audio generation endpoint (placeholder for future implementation)"""
    # This would be for streaming generation progress
    # For now, just redirect to regular generation
    return await create_audio_generation(request, BackgroundTasks(), service)


@openai_api_app.post("/audio/speech")
async def create_speech_generation_stream(
    request: SpeechGenerationRequest,
    service: acestepaudoservice = Depends(bentoml.get_current_service)
):
    """OpenAI Speech API compatible streaming endpoint"""
    try:
        # Convert to streaming response
        return StreamingResponse(
            service.generate_audio_stream(request),
            media_type="text/plain",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Content-Type": "text/plain; charset=utf-8"
            }
        )
        
    except ValueError as e:
        raise bentoml.exceptions.BadInput(str(e))
    except bentoml.exceptions.BentoMLException:
        raise  # Re-raise BentoML exceptions as-is
    except Exception as e:
        raise bentoml.exceptions.InternalServerError(f"Internal server error: {str(e)}")


@openai_api_app.get("/audio/files/{filename}")
async def get_audio_file(filename: str):
    """Serve generated audio files"""
    try:
        # Security check: only allow files from our output directory
        file_path = Path(config.output_dir) / filename
        
        # Ensure the file path is within our output directory (prevent directory traversal)
        if not str(file_path.resolve()).startswith(str(Path(config.output_dir).resolve())):
            raise bentoml.exceptions.BadInput("Access denied - invalid file path")
        
        if not file_path.exists():
            raise bentoml.exceptions.NotFound("Audio file not found")
        
        # Determine media type based on file extension
        media_type_map = {
            ".wav": "audio/wav",
            ".mp3": "audio/mpeg",
            ".ogg": "audio/ogg"
        }
        media_type = media_type_map.get(file_path.suffix.lower(), "audio/wav")
        
        return FileResponse(
            path=str(file_path),
            media_type=media_type,
            filename=filename
        )
        
    except bentoml.exceptions.BentoMLException:
        raise  # Re-raise BentoML exceptions as-is
    except Exception as e:
        raise bentoml.exceptions.InternalServerError(f"Error serving file: {str(e)}")


@openai_api_app.post("/audio/upload")
async def upload_audio_file(
    file: UploadFile = File(...),
    service: acestepaudoservice = Depends(bentoml.get_current_service)
):
    """Upload audio file for processing (future feature)"""
    # This could be used for audio-to-audio generation or other features
    if not file.content_type.startswith("audio/"):
        raise bentoml.exceptions.BadInput("File must be an audio file")
    
    # For now, just return file info
    return {
        "filename": file.filename,
        "content_type": file.content_type,
        "size": file.size,
        "message": "File upload received (processing not yet implemented)"
    }


@openai_api_app.get("/models", response_model=ModelsResponse)
async def list_models():
    """List available models"""
    models = [
        ModelInfo(
            id="ace-step-v1",
            created=int(time.time()),
            owned_by="ace-step"
        )
    ]
    
    return ModelsResponse(data=models)


@openai_api_app.get("/models/{model_id}", response_model=ModelInfo)
async def get_model(model_id: str):
    """Get model information"""
    if model_id != "ace-step-v1":
        raise bentoml.exceptions.NotFound("Model not found")
    
    return ModelInfo(
        id=model_id,
        created=int(time.time()),
        owned_by="ace-step"
    )


@openai_api_app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "ace-step-audio-service"}


# Additional utility endpoints
@openai_api_app.get("/audio/status/{generation_id}")
async def get_generation_status(generation_id: str):
    """Get status of audio generation (placeholder for future async processing)"""
    # This would be useful for long-running generations
    return {
        "id": generation_id,
        "status": "completed",  # For now, all generations are synchronous
        "message": "Generation completed"
    }


@openai_api_app.delete("/audio/files/{filename}")
async def delete_audio_file(filename: str):
    """Delete a generated audio file"""
    try:
        file_path = Path(config.output_dir) / filename
        
        # Security check
        if not str(file_path.resolve()).startswith(str(Path(config.output_dir).resolve())):
            raise bentoml.exceptions.BadInput("Access denied - invalid file path")
        
        if not file_path.exists():
            raise bentoml.exceptions.NotFound("Audio file not found")
        
        file_path.unlink()
        return {"message": f"File {filename} deleted successfully"}
        
    except bentoml.exceptions.BentoMLException:
        raise  # Re-raise BentoML exceptions as-is
    except Exception as e:
        raise bentoml.exceptions.InternalServerError(f"Error deleting file: {str(e)}")


# Error handlers for OpenAI compatibility
@openai_api_app.exception_handler(bentoml.exceptions.BadInput)
async def bad_input_handler(request, exc):
    """Handle bad input errors in OpenAI format"""
    return {
        "error": {
            "message": str(exc),
            "type": "invalid_request_error",
            "code": 400
        }
    }


@openai_api_app.exception_handler(bentoml.exceptions.NotFound)
async def not_found_handler(request, exc):
    """Handle not found errors in OpenAI format"""
    return {
        "error": {
            "message": str(exc),
            "type": "invalid_request_error",
            "code": 404
        }
    }


# Note: Forbidden exception doesn't exist in this BentoML version
# Using BadInput for access control errors instead


@openai_api_app.exception_handler(bentoml.exceptions.ServiceUnavailable)
async def service_unavailable_handler(request, exc):
    """Handle service unavailable errors in OpenAI format"""
    return {
        "error": {
            "message": str(exc),
            "type": "server_error",
            "code": 503
        }
    }


@openai_api_app.exception_handler(bentoml.exceptions.InternalServerError)
async def internal_server_error_handler(request, exc):
    """Handle internal server errors in OpenAI format"""
    return {
        "error": {
            "message": str(exc),
            "type": "server_error",
            "code": 500
        }
    }


@openai_api_app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions in OpenAI format (fallback)"""
    return {
        "error": {
            "message": exc.detail,
            "type": "invalid_request_error" if exc.status_code < 500 else "server_error",
            "code": exc.status_code
        }
    }


@openai_api_app.exception_handler(ValueError)
async def validation_exception_handler(request, exc):
    """Handle validation errors in OpenAI format"""
    return {
        "error": {
            "message": str(exc),
            "type": "invalid_request_error",
            "code": 400
        }
    }


@openai_api_app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions in OpenAI format"""
    import traceback
    print(f"Unhandled exception: {exc}")
    print(traceback.format_exc())
    
    return {
        "error": {
            "message": "Internal server error",
            "type": "server_error",
            "code": 500
        }
    }
