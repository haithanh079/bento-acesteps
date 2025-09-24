"""
Configuration management for ACE-Steps BentoML Service
"""

import os
from typing import Dict, Any, Optional
from pathlib import Path

class ServiceConfig:
    """
    Configuration class for the ACE-Steps BentoML service
    """
    
    def __init__(self):
        # Model configuration
        self.DEFAULT_CHECKPOINT_PATH = os.getenv(
            "ACE_STEP_CHECKPOINT_PATH", 
            "ACE-Step/ACE-Step-v1-3.5B"
        )
        self.DEFAULT_DEVICE = os.getenv("ACE_STEP_DEVICE", "cuda")
        self.DEFAULT_DTYPE = os.getenv("ACE_STEP_DTYPE", "bfloat16")
        self.DEFAULT_DEVICE_ID = int(os.getenv("ACE_STEP_DEVICE_ID", "0"))
        
        # Resource configuration
        self.MAX_CONCURRENT_REQUESTS = int(os.getenv("MAX_CONCURRENT_REQUESTS", "2"))
        self.REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "300"))
        self.GPU_MEMORY_LIMIT = os.getenv("GPU_MEMORY_LIMIT", "8Gi")
        
        # Output configuration
        self.OUTPUT_DIR = os.getenv("OUTPUT_DIR", "/tmp/ace_step_outputs")
        self.MAX_OUTPUT_SIZE = int(os.getenv("MAX_OUTPUT_SIZE", str(100 * 1024 * 1024)))  # 100MB
        self.CLEANUP_INTERVAL_HOURS = int(os.getenv("CLEANUP_INTERVAL_HOURS", "24"))
        
        # Audio configuration
        self.DEFAULT_SAMPLE_RATE = int(os.getenv("DEFAULT_SAMPLE_RATE", "44100"))
        self.DEFAULT_AUDIO_FORMAT = os.getenv("DEFAULT_AUDIO_FORMAT", "wav")
        self.MAX_AUDIO_DURATION = int(os.getenv("MAX_AUDIO_DURATION", "600"))  # 10 minutes
        
        # Generation parameters
        self.DEFAULT_INFER_STEPS = int(os.getenv("DEFAULT_INFER_STEPS", "60"))
        self.DEFAULT_GUIDANCE_SCALE = float(os.getenv("DEFAULT_GUIDANCE_SCALE", "15.0"))
        self.DEFAULT_OMEGA_SCALE = float(os.getenv("DEFAULT_OMEGA_SCALE", "10.0"))
        
        # Logging configuration
        self.LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
        self.LOG_FORMAT = os.getenv(
            "LOG_FORMAT", 
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        
        # Security configuration
        self.ENABLE_CORS = os.getenv("ENABLE_CORS", "true").lower() == "true"
        self.ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")
        self.API_KEY_REQUIRED = os.getenv("API_KEY_REQUIRED", "false").lower() == "true"
        self.API_KEY = os.getenv("API_KEY", "")
        
        # Performance configuration
        self.ENABLE_TORCH_COMPILE = os.getenv("ENABLE_TORCH_COMPILE", "false").lower() == "true"
        self.ENABLE_CPU_OFFLOAD = os.getenv("ENABLE_CPU_OFFLOAD", "false").lower() == "true"
        self.ENABLE_OVERLAPPED_DECODE = os.getenv("ENABLE_OVERLAPPED_DECODE", "false").lower() == "true"
        
        # Monitoring configuration
        self.ENABLE_METRICS = os.getenv("ENABLE_METRICS", "true").lower() == "true"
        self.METRICS_PORT = int(os.getenv("METRICS_PORT", "9090"))
        
        # Initialize directories
        self._create_directories()
    
    def _create_directories(self):
        """Create necessary directories"""
        directories = [
            self.OUTPUT_DIR,
            os.path.join(self.OUTPUT_DIR, "temp"),
            os.path.join(self.OUTPUT_DIR, "logs")
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def get_model_config(self) -> Dict[str, Any]:
        """Get model configuration dictionary"""
        return {
            "checkpoint_path": self.DEFAULT_CHECKPOINT_PATH,
            "device": self.DEFAULT_DEVICE,
            "dtype": self.DEFAULT_DTYPE,
            "device_id": self.DEFAULT_DEVICE_ID,
            "torch_compile": self.ENABLE_TORCH_COMPILE,
            "cpu_offload": self.ENABLE_CPU_OFFLOAD,
            "overlapped_decode": self.ENABLE_OVERLAPPED_DECODE
        }
    
    def get_generation_config(self) -> Dict[str, Any]:
        """Get default generation configuration"""
        return {
            "infer_step": self.DEFAULT_INFER_STEPS,
            "guidance_scale": self.DEFAULT_GUIDANCE_SCALE,
            "omega_scale": self.DEFAULT_OMEGA_SCALE,
            "scheduler_type": "euler",
            "cfg_type": "classifier_free"
        }
    
    def get_audio_config(self) -> Dict[str, Any]:
        """Get audio processing configuration"""
        return {
            "sample_rate": self.DEFAULT_SAMPLE_RATE,
            "format": self.DEFAULT_AUDIO_FORMAT,
            "max_duration": self.MAX_AUDIO_DURATION,
            "output_dir": self.OUTPUT_DIR
        }
    
    def get_resource_config(self) -> Dict[str, Any]:
        """Get resource configuration"""
        return {
            "max_concurrent_requests": self.MAX_CONCURRENT_REQUESTS,
            "request_timeout": self.REQUEST_TIMEOUT,
            "gpu_memory_limit": self.GPU_MEMORY_LIMIT,
            "max_output_size": self.MAX_OUTPUT_SIZE
        }
    
    def get_security_config(self) -> Dict[str, Any]:
        """Get security configuration"""
        return {
            "enable_cors": self.ENABLE_CORS,
            "allowed_origins": self.ALLOWED_ORIGINS,
            "api_key_required": self.API_KEY_REQUIRED,
            "api_key": self.API_KEY
        }
    
    def get_monitoring_config(self) -> Dict[str, Any]:
        """Get monitoring configuration"""
        return {
            "enable_metrics": self.ENABLE_METRICS,
            "metrics_port": self.METRICS_PORT,
            "log_level": self.LOG_LEVEL,
            "log_format": self.LOG_FORMAT
        }
    
    def validate_config(self) -> bool:
        """Validate configuration settings"""
        try:
            # Validate paths
            if not os.path.exists(self.OUTPUT_DIR):
                os.makedirs(self.OUTPUT_DIR, exist_ok=True)
            
            # Validate numeric values
            assert self.MAX_CONCURRENT_REQUESTS > 0, "MAX_CONCURRENT_REQUESTS must be positive"
            assert self.REQUEST_TIMEOUT > 0, "REQUEST_TIMEOUT must be positive"
            assert self.MAX_AUDIO_DURATION > 0, "MAX_AUDIO_DURATION must be positive"
            assert self.DEFAULT_SAMPLE_RATE > 0, "DEFAULT_SAMPLE_RATE must be positive"
            
            # Validate device configuration
            if self.DEFAULT_DEVICE == "cuda":
                import torch
                if not torch.cuda.is_available():
                    raise ValueError("CUDA not available but device set to cuda")
            
            return True
            
        except Exception as e:
            print(f"Configuration validation failed: {str(e)}")
            return False
    
    def update_config(self, updates: Dict[str, Any]) -> None:
        """Update configuration with new values"""
        for key, value in updates.items():
            if hasattr(self, key.upper()):
                setattr(self, key.upper(), value)
            else:
                print(f"Warning: Unknown configuration key: {key}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            "model": self.get_model_config(),
            "generation": self.get_generation_config(),
            "audio": self.get_audio_config(),
            "resources": self.get_resource_config(),
            "security": self.get_security_config(),
            "monitoring": self.get_monitoring_config()
        }
    
    def __str__(self) -> str:
        """String representation of configuration"""
        return f"ServiceConfig(output_dir={self.OUTPUT_DIR}, device={self.DEFAULT_DEVICE}, dtype={self.DEFAULT_DTYPE})"

# Global configuration instance
config = ServiceConfig()
