"""
Configuration settings for ACE-Step BentoML service
"""

import os
from typing import Optional
try:
    from pydantic_settings import BaseSettings
    from pydantic import Field
except ImportError:
    try:
        from pydantic import BaseSettings, Field
    except ImportError:
        # Fallback for older pydantic versions
        from pydantic import BaseModel as BaseSettings, Field


class ACEStepConfig(BaseSettings):
    """Configuration for ACE-Step service"""
    
    # Model settings
    checkpoint_path: str = Field(
        default="",
        env="ACE_STEP_CHECKPOINT_PATH",
        description="Path to ACE-Step model checkpoints"
    )
    
    device_id: int = Field(
        default=0,
        env="CUDA_VISIBLE_DEVICES",
        description="CUDA device ID to use"
    )
    
    dtype: str = Field(
        default="bfloat16",
        env="ACE_PIPELINE_DTYPE",
        description="Model dtype (bfloat16, float32)"
    )
    
    torch_compile: bool = Field(
        default=False,
        env="TORCH_COMPILE",
        description="Enable torch.compile optimization"
    )
    
    cpu_offload: bool = Field(
        default=False,
        env="CPU_OFFLOAD",
        description="Enable CPU offloading"
    )
    
    overlapped_decode: bool = Field(
        default=False,
        env="OVERLAPPED_DECODE",
        description="Enable overlapped decoding"
    )
    
    # Service settings
    max_duration: float = Field(
        default=240.0,
        env="MAX_AUDIO_DURATION",
        description="Maximum audio duration in seconds"
    )
    
    default_duration: float = Field(
        default=30.0,
        env="DEFAULT_AUDIO_DURATION",
        description="Default audio duration in seconds"
    )
    
    max_inference_steps: int = Field(
        default=200,
        env="MAX_INFERENCE_STEPS",
        description="Maximum number of inference steps"
    )
    
    default_inference_steps: int = Field(
        default=60,
        env="DEFAULT_INFERENCE_STEPS",
        description="Default number of inference steps"
    )
    
    # File storage settings
    output_dir: str = Field(
        default="/tmp/ace_step_outputs",
        env="OUTPUT_DIR",
        description="Directory for output files"
    )
    
    cleanup_files: bool = Field(
        default=True,
        env="CLEANUP_FILES",
        description="Automatically cleanup generated files"
    )
    
    file_retention_hours: int = Field(
        default=24,
        env="FILE_RETENTION_HOURS",
        description="Hours to retain generated files"
    )
    
    # API settings
    enable_cors: bool = Field(
        default=True,
        env="ENABLE_CORS",
        description="Enable CORS for API"
    )
    
    api_key: Optional[str] = Field(
        default=None,
        env="API_KEY",
        description="API key for authentication (optional)"
    )
    
    rate_limit_per_minute: int = Field(
        default=10,
        env="RATE_LIMIT_PER_MINUTE",
        description="Rate limit per minute per client"
    )
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# Global config instance
config = ACEStepConfig()
