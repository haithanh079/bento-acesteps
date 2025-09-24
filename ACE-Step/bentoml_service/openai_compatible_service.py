"""
OpenAI-Compatible BentoML Service for ACE-Steps Audio Generation
Provides OpenAI API-compatible endpoints for seamless AI Gateway integration
"""

import asyncio
import os
import time
import uuid
from pathlib import Path
from typing import Optional, Dict, Any, List, Union
import logging
import json
import base64

import bentoml
import torch
from pydantic import BaseModel

from .models.pipeline_manager import PipelineManager
from .utils.audio_utils import AudioProcessor
from .utils.config import ServiceConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Service configuration
config = ServiceConfig()

# OpenAI-compatible response models
class OpenAIError(BaseModel):
    message: str
    type: str
    code: Optional[str] = None

class OpenAIUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    generation_time: float = 0.0

class OpenAIAudioResponse(BaseModel):
    id: str
    object: str = "audio"
    created: int
    model: str
    data: List[Dict[str, Any]]
    usage: OpenAIUsage

class OpenAIModelsResponse(BaseModel):
    object: str = "list"
    data: List[Dict[str, Any]]

class OpenAIModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str = "ace-steps"
    permission: List[Dict[str, Any]] = []
    root: str = "ace-step-audio"
    parent: Optional[str] = None

@bentoml.service(
    traffic={"timeout": 300},
    resources={"gpu": 1, "memory": "8Gi"},
    name="ace-step-openai-service"
)
class ACEStepOpenAIService:
    """
    OpenAI-Compatible BentoML service for ACE-Steps audio generation
    """
    
    def __init__(self):
        self.pipeline_manager = PipelineManager()
        self.audio_processor = AudioProcessor()
        self.model_loaded = False
        self.loading_lock = asyncio.Lock()
        self.service_start_time = time.time()
        
        # OpenAI-compatible model information
        self.model_info = {
            "ace-step-audio": {
                "id": "ace-step-audio",
                "object": "model",
                "created": int(self.service_start_time),
                "owned_by": "ace-steps",
                "permission": [],
                "root": "ace-step-audio",
                "parent": None
            },
            "ace-step-music": {
                "id": "ace-step-music", 
                "object": "model",
                "created": int(self.service_start_time),
                "owned_by": "ace-steps",
                "permission": [],
                "root": "ace-step-music",
                "parent": None
            }
        }
    
    async def _ensure_model_loaded(self):
        """Ensure the model is loaded before processing requests"""
        if not self.model_loaded:
            async with self.loading_lock:
                if not self.model_loaded:
                    try:
                        await self.pipeline_manager.load_model(
                            checkpoint_path=config.DEFAULT_CHECKPOINT_PATH,
                            device_config={
                                "device": config.DEFAULT_DEVICE,
                                "dtype": config.DEFAULT_DTYPE,
                                "torch_compile": False,
                                "cpu_offload": False
                            }
                        )
                        self.model_loaded = True
                        logger.info("Model loaded successfully")
                    except Exception as e:
                        logger.error(f"Failed to load model: {str(e)}")
                        raise RuntimeError(f"Model loading failed: {str(e)}")

    @bentoml.api(route="/v1/audio/speech")
    async def create_speech(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        OpenAI-compatible speech generation endpoint
        Maps to ACE-Steps text-to-music generation
        """
        start_time = time.time()
        request_id = str(uuid.uuid4())
        
        try:
            # Validate request
            if "model" not in request:
                raise ValueError("Model is required")
            
            if "input" not in request:
                raise ValueError("Input text is required")
            
            model = request["model"]
            input_text = request["input"]
            voice = request.get("voice", "alloy")  # OpenAI voice parameter
            response_format = request.get("response_format", "mp3")
            speed = request.get("speed", 1.0)
            
            # Map OpenAI parameters to ACE-Steps parameters
            duration = int(request.get("duration", 30))  # Default 30 seconds
            guidance_scale = float(request.get("guidance_scale", 15.0))
            infer_steps = int(request.get("infer_steps", 60))
            
            # Ensure model is loaded
            await self._ensure_model_loaded()
            
            # Generate unique output path
            output_filename = f"speech_{request_id}.{response_format}"
            output_path = os.path.join(config.OUTPUT_DIR, output_filename)
            os.makedirs(config.OUTPUT_DIR, exist_ok=True)
            
            # Prepare generation parameters for ACE-Steps
            generation_params = {
                "audio_duration": duration,
                "prompt": input_text,
                "lyrics": "",  # No lyrics for speech generation
                "infer_step": infer_steps,
                "guidance_scale": guidance_scale,
                "scheduler_type": "euler",
                "cfg_type": "classifier_free",
                "omega_scale": 10.0,
                "manual_seeds": "",
                "guidance_interval": 0.1,
                "guidance_interval_decay": 0.95,
                "min_guidance_scale": 1.0,
                "use_erg_tag": False,
                "use_erg_lyric": False,
                "use_erg_diffusion": False,
                "oss_steps": "",
                "guidance_scale_text": 0.0,
                "guidance_scale_lyric": 0.0,
                "save_path": output_path
            }
            
            # Generate audio
            logger.info(f"Generating speech for: {input_text[:50]}...")
            await self.pipeline_manager.generate(generation_params)
            
            generation_time = time.time() - start_time
            
            # Read generated audio file
            if os.path.exists(output_path):
                with open(output_path, "rb") as f:
                    audio_data = f.read()
                    audio_base64 = base64.b64encode(audio_data).decode('utf-8')
            else:
                raise RuntimeError("Generated audio file not found")
            
            # Create OpenAI-compatible response
            response_data = [{
                "audio": audio_base64,
                "format": response_format,
                "duration": duration,
                "voice": voice,
                "speed": speed
            }]
            
            usage = OpenAIUsage(
                prompt_tokens=len(input_text.split()),
                completion_tokens=0,
                total_tokens=len(input_text.split()),
                generation_time=generation_time
            )
            
            return OpenAIAudioResponse(
                id=request_id,
                created=int(time.time()),
                model=model,
                data=response_data,
                usage=usage
            )
            
        except Exception as e:
            logger.error(f"Speech generation failed: {str(e)}")
            raise RuntimeError(f"Speech generation failed: {str(e)}")

    @bentoml.api(route="/v1/audio/music")
    async def create_music(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        OpenAI-compatible music generation endpoint
        Maps to ACE-Steps music generation with lyrics support
        """
        start_time = time.time()
        request_id = str(uuid.uuid4())
        
        try:
            # Validate request
            if "model" not in request:
                raise ValueError("Model is required")
            
            if "prompt" not in request:
                raise ValueError("Prompt is required")
            
            model = request["model"]
            prompt = request["prompt"]
            lyrics = request.get("lyrics", "")
            response_format = request.get("response_format", "mp3")
            duration = int(request.get("duration", 30))
            
            # Map OpenAI parameters to ACE-Steps parameters
            guidance_scale = float(request.get("guidance_scale", 15.0))
            infer_steps = int(request.get("infer_steps", 60))
            style = request.get("style", "pop")  # Music style parameter
            
            # Ensure model is loaded
            await self._ensure_model_loaded()
            
            # Generate unique output path
            output_filename = f"music_{request_id}.{response_format}"
            output_path = os.path.join(config.OUTPUT_DIR, output_filename)
            os.makedirs(config.OUTPUT_DIR, exist_ok=True)
            
            # Prepare generation parameters for ACE-Steps
            generation_params = {
                "audio_duration": duration,
                "prompt": f"{style} music: {prompt}",
                "lyrics": lyrics,
                "infer_step": infer_steps,
                "guidance_scale": guidance_scale,
                "scheduler_type": "euler",
                "cfg_type": "classifier_free",
                "omega_scale": 10.0,
                "manual_seeds": "",
                "guidance_interval": 0.1,
                "guidance_interval_decay": 0.95,
                "min_guidance_scale": 1.0,
                "use_erg_tag": False,
                "use_erg_lyric": bool(lyrics),
                "use_erg_diffusion": False,
                "oss_steps": "",
                "guidance_scale_text": 0.0,
                "guidance_scale_lyric": 5.0 if lyrics else 0.0,
                "save_path": output_path
            }
            
            # Generate audio
            logger.info(f"Generating music for: {prompt[:50]}...")
            await self.pipeline_manager.generate(generation_params)
            
            generation_time = time.time() - start_time
            
            # Read generated audio file
            if os.path.exists(output_path):
                with open(output_path, "rb") as f:
                    audio_data = f.read()
                    audio_base64 = base64.b64encode(audio_data).decode('utf-8')
            else:
                raise RuntimeError("Generated audio file not found")
            
            # Create OpenAI-compatible response
            response_data = [{
                "audio": audio_base64,
                "format": response_format,
                "duration": duration,
                "style": style,
                "has_lyrics": bool(lyrics)
            }]
            
            usage = OpenAIUsage(
                prompt_tokens=len(prompt.split()) + len(lyrics.split()),
                completion_tokens=0,
                total_tokens=len(prompt.split()) + len(lyrics.split()),
                generation_time=generation_time
            )
            
            return OpenAIAudioResponse(
                id=request_id,
                created=int(time.time()),
                model=model,
                data=response_data,
                usage=usage
            )
            
        except Exception as e:
            logger.error(f"Music generation failed: {str(e)}")
            raise RuntimeError(f"Music generation failed: {str(e)}")

    @bentoml.api(route="/v1/models")
    async def list_models(self) -> Dict[str, Any]:
        """
        OpenAI-compatible models list endpoint
        """
        try:
            models = list(self.model_info.values())
            return OpenAIModelsResponse(data=models)
        except Exception as e:
            logger.error(f"Failed to list models: {str(e)}")
            raise RuntimeError(f"Failed to list models: {str(e)}")

    @bentoml.api(route="/v1/models/{model_id}")
    async def retrieve_model(self, model_id: str) -> Dict[str, Any]:
        """
        OpenAI-compatible model retrieval endpoint
        """
        try:
            if model_id not in self.model_info:
                raise ValueError(f"Model {model_id} not found")
            
            model_data = self.model_info[model_id]
            return OpenAIModelInfo(**model_data)
        except Exception as e:
            logger.error(f"Failed to retrieve model {model_id}: {str(e)}")
            raise RuntimeError(f"Failed to retrieve model {model_id}: {str(e)}")

    @bentoml.api(route="/v1/audio/transcriptions")
    async def create_transcription(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        OpenAI-compatible transcription endpoint
        Note: ACE-Steps doesn't support transcription, returns placeholder
        """
        return {
            "text": "Transcription not supported by ACE-Steps service",
            "error": "This service only supports audio generation, not transcription"
        }

    @bentoml.api(route="/v1/audio/translations")
    async def create_translation(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        OpenAI-compatible translation endpoint
        Note: ACE-Steps doesn't support translation, returns placeholder
        """
        return {
            "text": "Translation not supported by ACE-Steps service",
            "error": "This service only supports audio generation, not translation"
        }

    @bentoml.api(route="/health")
    async def health_check(self) -> Dict[str, Any]:
        """
        Health check endpoint
        """
        try:
            gpu_available = torch.cuda.is_available()
            gpu_count = torch.cuda.device_count() if gpu_available else 0
            output_dir_exists = os.path.exists(config.OUTPUT_DIR)
            
            return {
                "status": "healthy" if (self.model_loaded and gpu_available and output_dir_exists) else "unhealthy",
                "model_loaded": self.model_loaded,
                "gpu_available": gpu_available,
                "gpu_count": gpu_count,
                "output_directory_accessible": output_dir_exists,
                "service_uptime": time.time() - self.service_start_time,
                "models_available": list(self.model_info.keys()),
                "openai_compatible": True
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "openai_compatible": True
            }

    def __post_init__(self):
        """Initialize service after creation"""
        logger.info("ACE-Steps OpenAI-Compatible service initialized")
