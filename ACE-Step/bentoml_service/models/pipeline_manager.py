"""
Pipeline Manager for ACE-Steps BentoML Service
Handles model loading and audio generation
"""

import asyncio
import os
import logging
from typing import Dict, Any, Optional
import torch

# Import ACE-Steps components
from acestep.pipeline_ace_step import ACEStepPipeline
from acestep.data_sampler import DataSampler

logger = logging.getLogger(__name__)

class PipelineManager:
    """
    Manages the ACE-Steps pipeline for audio generation
    """
    
    def __init__(self):
        self.pipeline: Optional[ACEStepPipeline] = None
        self.data_sampler: Optional[DataSampler] = None
        self.model_loaded = False
        self.loading_lock = asyncio.Lock()
        
    async def load_model(
        self, 
        checkpoint_path: str, 
        device_config: Dict[str, Any]
    ) -> None:
        """
        Load the ACE-Steps model asynchronously
        
        Args:
            checkpoint_path: Path to the model checkpoint
            device_config: Device configuration dictionary
        """
        async with self.loading_lock:
            if self.model_loaded:
                logger.info("Model already loaded")
                return
                
            try:
                logger.info(f"Loading model from {checkpoint_path}")
                
                # Set CUDA device if specified
                if "device" in device_config and device_config["device"] == "cuda":
                    device_id = device_config.get("device_id", 0)
                    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
                
                # Initialize pipeline
                self.pipeline = ACEStepPipeline(
                    checkpoint_dir=checkpoint_path,
                    dtype=device_config.get("dtype", "bfloat16"),
                    torch_compile=device_config.get("torch_compile", False),
                    cpu_offload=device_config.get("cpu_offload", False),
                    overlapped_decode=device_config.get("overlapped_decode", False)
                )
                
                # Initialize data sampler
                self.data_sampler = DataSampler()
                
                self.model_loaded = True
                logger.info("Model loaded successfully")
                
            except Exception as e:
                logger.error(f"Failed to load model: {str(e)}")
                raise RuntimeError(f"Model loading failed: {str(e)}")
    
    async def generate(self, params: Dict[str, Any]) -> str:
        """
        Generate audio using the loaded pipeline
        
        Args:
            params: Generation parameters
            
        Returns:
            str: Path to the generated audio file
        """
        if not self.model_loaded or self.pipeline is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        try:
            logger.info("Starting audio generation")
            
            # Extract parameters
            audio_duration = params.get("audio_duration", 10.0)
            prompt = params.get("prompt", "")
            lyrics = params.get("lyrics", "")
            infer_step = params.get("infer_step", 60)
            guidance_scale = params.get("guidance_scale", 15.0)
            scheduler_type = params.get("scheduler_type", "euler")
            cfg_type = params.get("cfg_type", "classifier_free")
            omega_scale = params.get("omega_scale", 10.0)
            manual_seeds = params.get("manual_seeds", "")
            guidance_interval = params.get("guidance_interval", 0.1)
            guidance_interval_decay = params.get("guidance_interval_decay", 0.95)
            min_guidance_scale = params.get("min_guidance_scale", 1.0)
            use_erg_tag = params.get("use_erg_tag", False)
            use_erg_lyric = params.get("use_erg_lyric", False)
            use_erg_diffusion = params.get("use_erg_diffusion", False)
            oss_steps = params.get("oss_steps", "")
            guidance_scale_text = params.get("guidance_scale_text", 0.0)
            guidance_scale_lyric = params.get("guidance_scale_lyric", 0.0)
            save_path = params.get("save_path", "output.wav")
            
            # Run generation in a separate thread to avoid blocking
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, 
                self._run_generation,
                audio_duration,
                prompt,
                lyrics,
                infer_step,
                guidance_scale,
                scheduler_type,
                cfg_type,
                omega_scale,
                manual_seeds,
                guidance_interval,
                guidance_interval_decay,
                min_guidance_scale,
                use_erg_tag,
                use_erg_lyric,
                use_erg_diffusion,
                oss_steps,
                guidance_scale_text,
                guidance_scale_lyric,
                save_path
            )
            
            logger.info(f"Audio generation completed: {save_path}")
            return result
            
        except Exception as e:
            logger.error(f"Audio generation failed: {str(e)}")
            raise RuntimeError(f"Audio generation failed: {str(e)}")
    
    def _run_generation(
        self,
        audio_duration: float,
        prompt: str,
        lyrics: str,
        infer_step: int,
        guidance_scale: float,
        scheduler_type: str,
        cfg_type: str,
        omega_scale: float,
        manual_seeds: str,
        guidance_interval: float,
        guidance_interval_decay: float,
        min_guidance_scale: float,
        use_erg_tag: bool,
        use_erg_lyric: bool,
        use_erg_diffusion: bool,
        oss_steps: str,
        guidance_scale_text: float,
        guidance_scale_lyric: float,
        save_path: str
    ) -> str:
        """
        Run the actual generation in a separate thread
        
        Returns:
            str: Path to the generated audio file
        """
        try:
            # Call the pipeline
            self.pipeline(
                audio_duration=audio_duration,
                prompt=prompt,
                lyrics=lyrics,
                infer_step=infer_step,
                guidance_scale=guidance_scale,
                scheduler_type=scheduler_type,
                cfg_type=cfg_type,
                omega_scale=omega_scale,
                manual_seeds=manual_seeds,
                guidance_interval=guidance_interval,
                guidance_interval_decay=guidance_interval_decay,
                min_guidance_scale=min_guidance_scale,
                use_erg_tag=use_erg_tag,
                use_erg_lyric=use_erg_lyric,
                use_erg_diffusion=use_erg_diffusion,
                oss_steps=oss_steps,
                guidance_scale_text=guidance_scale_text,
                guidance_scale_lyric=guidance_scale_lyric,
                save_path=save_path
            )
            
            return save_path
            
        except Exception as e:
            logger.error(f"Generation execution failed: {str(e)}")
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model
        
        Returns:
            Dict[str, Any]: Model information
        """
        if not self.model_loaded:
            return {"status": "not_loaded"}
        
        return {
            "status": "loaded",
            "pipeline_type": type(self.pipeline).__name__,
            "device": str(self.pipeline.device) if hasattr(self.pipeline, 'device') else "unknown",
            "dtype": getattr(self.pipeline, 'dtype', 'unknown')
        }
    
    def cleanup(self) -> None:
        """
        Clean up resources
        """
        if self.pipeline is not None:
            # Clear GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            self.pipeline = None
            self.model_loaded = False
            logger.info("Pipeline cleaned up")
