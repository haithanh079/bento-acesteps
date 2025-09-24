"""
Pydantic schemas for ACE-Steps BentoML Service API
Request and response models
"""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, validator
import re

class AudioGenerationRequest(BaseModel):
    """Request schema for audio generation"""
    
    # Core parameters
    audio_duration: float = Field(..., ge=1.0, le=300.0, description="Audio duration in seconds")
    prompt: str = Field("", description="Text prompt for generation")
    lyrics: str = Field("", description="Lyrics for generation")
    
    # Generation parameters
    infer_step: int = Field(60, ge=1, le=200, description="Number of inference steps")
    guidance_scale: float = Field(15.0, ge=0.0, le=50.0, description="Guidance scale")
    scheduler_type: str = Field("euler", description="Scheduler type")
    cfg_type: str = Field("classifier_free", description="CFG type")
    
    # Advanced parameters
    omega_scale: float = Field(10.0, ge=0.0, le=50.0, description="Omega scale")
    guidance_interval: float = Field(0.1, ge=0.0, le=1.0, description="Guidance interval")
    guidance_interval_decay: float = Field(0.95, ge=0.0, le=1.0, description="Guidance interval decay")
    min_guidance_scale: float = Field(1.0, ge=0.0, le=10.0, description="Minimum guidance scale")
    
    # Control parameters
    use_erg_tag: bool = Field(False, description="Use ERG tag")
    use_erg_lyric: bool = Field(False, description="Use ERG lyric")
    use_erg_diffusion: bool = Field(False, description="Use ERG diffusion")
    
    # Seeds and steps
    actual_seeds: List[int] = Field(default_factory=list, description="Manual seeds")
    oss_steps: List[int] = Field(default_factory=list, description="OSS steps")
    
    # Optional parameters
    guidance_scale_text: float = Field(0.0, ge=0.0, le=50.0, description="Text guidance scale")
    guidance_scale_lyric: float = Field(0.0, ge=0.0, le=50.0, description="Lyric guidance scale")
    output_format: str = Field("wav", description="Output audio format")
    
    @validator('scheduler_type')
    def validate_scheduler_type(cls, v):
        allowed_types = ['euler', 'heun', 'pingpong']
        if v not in allowed_types:
            raise ValueError(f'scheduler_type must be one of {allowed_types}')
        return v
    
    @validator('cfg_type')
    def validate_cfg_type(cls, v):
        allowed_types = ['classifier_free', 'classifier_guidance', 'classifier_free_guidance']
        if v not in allowed_types:
            raise ValueError(f'cfg_type must be one of {allowed_types}')
        return v
    
    @validator('output_format')
    def validate_output_format(cls, v):
        allowed_formats = ['wav', 'mp3', 'flac']
        if v not in allowed_formats:
            raise ValueError(f'output_format must be one of {allowed_formats}')
        return v
    
    @validator('prompt', 'lyrics')
    def validate_text_input(cls, v):
        if v and len(v.strip()) == 0:
            return ""
        return v
    
    def __init__(self, **data):
        super().__init__(**data)
        # Ensure at least one of prompt or lyrics is provided
        if not self.prompt and not self.lyrics:
            raise ValueError("Either prompt or lyrics must be provided")

class AudioGenerationResponse(BaseModel):
    """Response schema for audio generation"""
    
    status: str = Field(..., description="Generation status")
    output_path: str = Field(..., description="Path to generated audio file")
    generation_time: float = Field(..., description="Generation time in seconds")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Generation metadata")
    error_message: Optional[str] = Field(None, description="Error message if generation failed")

class MusicGenerationRequest(BaseModel):
    """Request schema for music generation"""
    
    prompt: str = Field(..., min_length=1, max_length=1000, description="Text prompt for music generation")
    duration: int = Field(240, ge=10, le=600, description="Duration in seconds")
    infer_steps: int = Field(60, ge=1, le=200, description="Number of inference steps")
    guidance_scale: float = Field(15.0, ge=0.0, le=50.0, description="Guidance scale")
    omega_scale: float = Field(10.0, ge=0.0, le=50.0, description="Omega scale")
    seed: Optional[int] = Field(None, ge=0, le=2**32-1, description="Random seed for reproducibility")
    
    @validator('prompt')
    def validate_prompt(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError("Prompt cannot be empty")
        return v.strip()

class MusicGenerationResponse(BaseModel):
    """Response schema for music generation"""
    
    status: str = Field(..., description="Generation status")
    audio_path: str = Field(..., description="Path to generated audio file")
    prompt: str = Field(..., description="Original prompt")
    seed: int = Field(..., description="Random seed used")
    sample_rate: int = Field(..., description="Audio sample rate")
    generation_time: float = Field(..., description="Generation time in seconds")
    error_message: Optional[str] = Field(None, description="Error message if generation failed")

class HealthResponse(BaseModel):
    """Response schema for health check"""
    
    status: str = Field(..., description="Overall service status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    gpu_available: bool = Field(..., description="Whether GPU is available")
    gpu_count: int = Field(..., description="Number of available GPUs")
    output_directory_accessible: bool = Field(..., description="Whether output directory is accessible")
    service_uptime: float = Field(..., description="Service uptime in seconds")
    error_message: Optional[str] = Field(None, description="Error message if any")

class ErrorResponse(BaseModel):
    """Response schema for errors"""
    
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")

class DownloadRequest(BaseModel):
    """Request schema for audio download"""
    
    file_path: str = Field(..., description="Path to the audio file to download")
    
    @validator('file_path')
    def validate_file_path(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError("File path cannot be empty")
        return v.strip()

class BatchGenerationRequest(BaseModel):
    """Request schema for batch audio generation"""
    
    requests: List[AudioGenerationRequest] = Field(..., min_items=1, max_items=10, description="List of generation requests")
    batch_id: Optional[str] = Field(None, description="Optional batch identifier")
    
    @validator('requests')
    def validate_requests(cls, v):
        if not v:
            raise ValueError("At least one request is required")
        return v

class BatchGenerationResponse(BaseModel):
    """Response schema for batch generation"""
    
    batch_id: str = Field(..., description="Batch identifier")
    total_requests: int = Field(..., description="Total number of requests")
    successful: int = Field(..., description="Number of successful generations")
    failed: int = Field(..., description="Number of failed generations")
    results: List[AudioGenerationResponse] = Field(..., description="Individual generation results")
    total_generation_time: float = Field(..., description="Total generation time in seconds")

class ModelInfoResponse(BaseModel):
    """Response schema for model information"""
    
    model_name: str = Field(..., description="Model name")
    model_version: str = Field(..., description="Model version")
    checkpoint_path: str = Field(..., description="Checkpoint path")
    device: str = Field(..., description="Device being used")
    dtype: str = Field(..., description="Data type")
    memory_usage: Dict[str, Any] = Field(default_factory=dict, description="Memory usage information")
    supported_languages: List[str] = Field(default_factory=list, description="Supported languages")

class ConfigurationRequest(BaseModel):
    """Request schema for configuration updates"""
    
    max_concurrent_requests: Optional[int] = Field(None, ge=1, le=10, description="Maximum concurrent requests")
    request_timeout: Optional[int] = Field(None, ge=60, le=600, description="Request timeout in seconds")
    gpu_memory_limit: Optional[str] = Field(None, description="GPU memory limit")
    output_directory: Optional[str] = Field(None, description="Output directory path")
    
    @validator('gpu_memory_limit')
    def validate_gpu_memory_limit(cls, v):
        if v is not None:
            # Validate format like "8Gi", "4GB", etc.
            pattern = r'^\d+[GM]i?B?$'
            if not re.match(pattern, v):
                raise ValueError("GPU memory limit must be in format like '8Gi' or '4GB'")
        return v
