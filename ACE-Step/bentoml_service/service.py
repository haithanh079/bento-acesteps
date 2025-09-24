"""
BentoML Service for ACE-Steps Audio Generation
Main service implementation with API endpoints
"""

import asyncio
import os
import time
import uuid
from pathlib import Path
from typing import Optional, Dict, Any, List
import logging

import bentoml
from bentoml.io import JSON, File, Multipart
import torch
import numpy as np

from .models.pipeline_manager import PipelineManager
from .api.schemas import (
    AudioGenerationRequest,
    AudioGenerationResponse,
    MusicGenerationRequest,
    MusicGenerationResponse,
    HealthResponse,
    ErrorResponse
)
from .utils.audio_utils import AudioProcessor
from .utils.config import ServiceConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Service configuration
config = ServiceConfig()

@bentoml.service(
    traffic={"timeout": 300},  # 5-minute timeout for audio generation
    resources={"gpu": 1, "memory": "8Gi"},
    name="ace-step-audio-service"
)
class ACEStepBentoService:
    """
    BentoML service for ACE-Steps audio generation
    """
    
    def __init__(self):
        self.pipeline_manager = PipelineManager()
        self.audio_processor = AudioProcessor()
        self.model_loaded = False
        self.loading_lock = asyncio.Lock()
        
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

    @bentoml.api(
        route="/generate",
        input=JSON(pydantic_model=AudioGenerationRequest),
        output=JSON(pydantic_model=AudioGenerationResponse)
    )
    async def generate_audio(self, request: AudioGenerationRequest) -> AudioGenerationResponse:
        """
        Generate audio using ACE-Steps pipeline
        
        Args:
            request: Audio generation parameters
            
        Returns:
            AudioGenerationResponse: Generated audio information
        """
        start_time = time.time()
        
        try:
            # Ensure model is loaded
            await self._ensure_model_loaded()
            
            # Validate request parameters
            if not request.prompt and not request.lyrics:
                return AudioGenerationResponse(
                    status="error",
                    output_path="",
                    generation_time=0,
                    metadata={},
                    error_message="Either prompt or lyrics must be provided"
                )
            
            # Generate unique output path
            output_filename = f"generated_{uuid.uuid4().hex}.{request.output_format}"
            output_path = os.path.join(config.OUTPUT_DIR, output_filename)
            
            # Ensure output directory exists
            os.makedirs(config.OUTPUT_DIR, exist_ok=True)
            
            # Prepare generation parameters
            generation_params = {
                "audio_duration": request.audio_duration,
                "prompt": request.prompt,
                "lyrics": request.lyrics,
                "infer_step": request.infer_step,
                "guidance_scale": request.guidance_scale,
                "scheduler_type": request.scheduler_type,
                "cfg_type": request.cfg_type,
                "omega_scale": request.omega_scale,
                "manual_seeds": ", ".join(map(str, request.actual_seeds)) if request.actual_seeds else "",
                "guidance_interval": request.guidance_interval,
                "guidance_interval_decay": request.guidance_interval_decay,
                "min_guidance_scale": request.min_guidance_scale,
                "use_erg_tag": request.use_erg_tag,
                "use_erg_lyric": request.use_erg_lyric,
                "use_erg_diffusion": request.use_erg_diffusion,
                "oss_steps": ", ".join(map(str, request.oss_steps)) if request.oss_steps else "",
                "guidance_scale_text": request.guidance_scale_text,
                "guidance_scale_lyric": request.guidance_scale_lyric,
                "save_path": output_path
            }
            
            # Generate audio
            logger.info(f"Starting audio generation with parameters: {generation_params}")
            await self.pipeline_manager.generate(generation_params)
            
            generation_time = time.time() - start_time
            
            # Prepare metadata
            metadata = {
                "generation_time": generation_time,
                "parameters": {
                    "audio_duration": request.audio_duration,
                    "infer_step": request.infer_step,
                    "guidance_scale": request.guidance_scale,
                    "scheduler_type": request.scheduler_type
                },
                "output_format": request.output_format,
                "file_size": os.path.getsize(output_path) if os.path.exists(output_path) else 0
            }
            
            logger.info(f"Audio generation completed in {generation_time:.2f} seconds")
            
            return AudioGenerationResponse(
                status="success",
                output_path=output_path,
                generation_time=generation_time,
                metadata=metadata,
                error_message=None
            )
            
        except Exception as e:
            generation_time = time.time() - start_time
            error_msg = f"Audio generation failed: {str(e)}"
            logger.error(error_msg)
            
            return AudioGenerationResponse(
                status="error",
                output_path="",
                generation_time=generation_time,
                metadata={},
                error_message=error_msg
            )

    @bentoml.api(
        route="/music",
        input=JSON(pydantic_model=MusicGenerationRequest),
        output=JSON(pydantic_model=MusicGenerationResponse)
    )
    async def generate_music(self, request: MusicGenerationRequest) -> MusicGenerationResponse:
        """
        Generate music from text prompt
        
        Args:
            request: Music generation parameters
            
        Returns:
            MusicGenerationResponse: Generated music information
        """
        start_time = time.time()
        
        try:
            # Ensure model is loaded
            await self._ensure_model_loaded()
            
            # Convert music request to audio generation request
            audio_request = AudioGenerationRequest(
                audio_duration=request.duration,
                prompt=request.prompt,
                lyrics="",  # No lyrics for text-to-music
                infer_step=request.infer_steps,
                guidance_scale=request.guidance_scale,
                scheduler_type="euler",
                cfg_type="classifier_free",
                omega_scale=request.omega_scale,
                actual_seeds=[request.seed] if request.seed else [],
                guidance_interval=0.1,
                guidance_interval_decay=0.95,
                min_guidance_scale=1.0,
                use_erg_tag=False,
                use_erg_lyric=False,
                use_erg_diffusion=False,
                oss_steps=[],
                guidance_scale_text=0.0,
                guidance_scale_lyric=0.0,
                output_format="wav"
            )
            
            # Generate audio
            audio_response = await self.generate_audio(audio_request)
            
            if audio_response.status == "error":
                return MusicGenerationResponse(
                    status="error",
                    audio_path="",
                    prompt=request.prompt,
                    seed=request.seed or 0,
                    sample_rate=0,
                    generation_time=audio_response.generation_time,
                    error_message=audio_response.error_message
                )
            
            # Get sample rate from generated audio
            sample_rate = 44100  # Default sample rate for ACE-Steps
            
            return MusicGenerationResponse(
                status="success",
                audio_path=audio_response.output_path,
                prompt=request.prompt,
                seed=request.seed or 0,
                sample_rate=sample_rate,
                generation_time=audio_response.generation_time,
                error_message=None
            )
            
        except Exception as e:
            generation_time = time.time() - start_time
            error_msg = f"Music generation failed: {str(e)}"
            logger.error(error_msg)
            
            return MusicGenerationResponse(
                status="error",
                audio_path="",
                prompt=request.prompt,
                seed=request.seed or 0,
                sample_rate=0,
                generation_time=generation_time,
                error_message=error_msg
            )

    @bentoml.api(
        route="/health",
        output=JSON(pydantic_model=HealthResponse)
    )
    async def health_check(self) -> HealthResponse:
        """
        Health check endpoint
        
        Returns:
            HealthResponse: Service health status
        """
        try:
            # Check if model is loaded
            model_status = "loaded" if self.model_loaded else "not_loaded"
            
            # Check GPU availability
            gpu_available = torch.cuda.is_available()
            gpu_count = torch.cuda.device_count() if gpu_available else 0
            
            # Check output directory
            output_dir_exists = os.path.exists(config.OUTPUT_DIR)
            
            # Overall health status
            overall_status = "healthy" if (model_status == "loaded" and gpu_available and output_dir_exists) else "unhealthy"
            
            return HealthResponse(
                status=overall_status,
                model_loaded=self.model_loaded,
                gpu_available=gpu_available,
                gpu_count=gpu_count,
                output_directory_accessible=output_dir_exists,
                service_uptime=time.time() - getattr(self, '_start_time', time.time()),
                error_message=None if overall_status == "healthy" else "Service not fully operational"
            )
            
        except Exception as e:
            return HealthResponse(
                status="error",
                model_loaded=False,
                gpu_available=False,
                gpu_count=0,
                output_directory_accessible=False,
                service_uptime=0,
                error_message=f"Health check failed: {str(e)}"
            )

    @bentoml.api(
        route="/download",
        input=JSON(pydantic_model=dict),
        output=File
    )
    async def download_audio(self, request: Dict[str, str]) -> bytes:
        """
        Download generated audio file
        
        Args:
            request: Dictionary containing file_path
            
        Returns:
            bytes: Audio file content
        """
        try:
            file_path = request.get("file_path")
            if not file_path:
                raise ValueError("file_path is required")
            
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
            
            with open(file_path, "rb") as f:
                return f.read()
                
        except Exception as e:
            logger.error(f"Download failed: {str(e)}")
            raise RuntimeError(f"Download failed: {str(e)}")

    def __post_init__(self):
        """Initialize service after creation"""
        self._start_time = time.time()
        logger.info("ACE-Steps BentoML service initialized")
