"""
OpenAI API Compatibility Adapters for ACE-Steps
Provides request/response mapping and compatibility layers
"""

from typing import Dict, Any, List, Optional, Union
from pydantic import BaseModel, Field
import json
import base64
from datetime import datetime

class OpenAIRequestAdapter:
    """
    Adapter for converting OpenAI API requests to ACE-Steps format
    """
    
    @staticmethod
    def speech_to_ace_steps(request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert OpenAI speech request to ACE-Steps format
        
        OpenAI Speech API Parameters:
        - model: str (required)
        - input: str (required) - text to convert to speech
        - voice: str (optional) - voice to use
        - response_format: str (optional) - audio format
        - speed: float (optional) - speech speed
        """
        return {
            "audio_duration": request.get("duration", 30),
            "prompt": request["input"],
            "lyrics": "",
            "infer_step": request.get("infer_steps", 60),
            "guidance_scale": request.get("guidance_scale", 15.0),
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
            "output_format": request.get("response_format", "mp3")
        }
    
    @staticmethod
    def music_to_ace_steps(request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert OpenAI music request to ACE-Steps format
        
        Custom Music API Parameters:
        - model: str (required)
        - prompt: str (required) - music description
        - lyrics: str (optional) - lyrics for the music
        - style: str (optional) - music style/genre
        - duration: int (optional) - duration in seconds
        - response_format: str (optional) - audio format
        """
        prompt = request["prompt"]
        style = request.get("style", "pop")
        lyrics = request.get("lyrics", "")
        
        return {
            "audio_duration": request.get("duration", 30),
            "prompt": f"{style} music: {prompt}",
            "lyrics": lyrics,
            "infer_step": request.get("infer_steps", 60),
            "guidance_scale": request.get("guidance_scale", 15.0),
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
            "output_format": request.get("response_format", "mp3")
        }

class OpenAIResponseAdapter:
    """
    Adapter for converting ACE-Steps responses to OpenAI format
    """
    
    @staticmethod
    def ace_steps_to_speech(
        ace_response: Dict[str, Any], 
        request_id: str,
        model: str,
        generation_time: float
    ) -> Dict[str, Any]:
        """
        Convert ACE-Steps response to OpenAI speech format
        """
        return {
            "id": request_id,
            "object": "audio",
            "created": int(datetime.now().timestamp()),
            "model": model,
            "data": [{
                "audio": ace_response.get("audio_base64", ""),
                "format": ace_response.get("format", "mp3"),
                "duration": ace_response.get("duration", 30),
                "voice": ace_response.get("voice", "alloy"),
                "speed": ace_response.get("speed", 1.0)
            }],
            "usage": {
                "prompt_tokens": ace_response.get("prompt_tokens", 0),
                "completion_tokens": 0,
                "total_tokens": ace_response.get("prompt_tokens", 0),
                "generation_time": generation_time
            }
        }
    
    @staticmethod
    def ace_steps_to_music(
        ace_response: Dict[str, Any],
        request_id: str,
        model: str,
        generation_time: float
    ) -> Dict[str, Any]:
        """
        Convert ACE-Steps response to OpenAI music format
        """
        return {
            "id": request_id,
            "object": "audio",
            "created": int(datetime.now().timestamp()),
            "model": model,
            "data": [{
                "audio": ace_response.get("audio_base64", ""),
                "format": ace_response.get("format", "mp3"),
                "duration": ace_response.get("duration", 30),
                "style": ace_response.get("style", "pop"),
                "has_lyrics": ace_response.get("has_lyrics", False)
            }],
            "usage": {
                "prompt_tokens": ace_response.get("prompt_tokens", 0),
                "completion_tokens": 0,
                "total_tokens": ace_response.get("prompt_tokens", 0),
                "generation_time": generation_time
            }
        }

class OpenAIErrorAdapter:
    """
    Adapter for OpenAI-compatible error responses
    """
    
    @staticmethod
    def create_error_response(
        error_type: str,
        message: str,
        code: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create OpenAI-compatible error response
        """
        return {
            "error": {
                "message": message,
                "type": error_type,
                "code": code
            }
        }
    
    @staticmethod
    def invalid_request_error(message: str = "Invalid request") -> Dict[str, Any]:
        """Invalid request error"""
        return OpenAIErrorAdapter.create_error_response(
            "invalid_request_error",
            message,
            "invalid_request"
        )
    
    @staticmethod
    def model_not_found_error(model: str) -> Dict[str, Any]:
        """Model not found error"""
        return OpenAIErrorAdapter.create_error_response(
            "model_not_found_error",
            f"Model {model} not found",
            "model_not_found"
        )
    
    @staticmethod
    def rate_limit_error() -> Dict[str, Any]:
        """Rate limit error"""
        return OpenAIErrorAdapter.create_error_response(
            "rate_limit_error",
            "Rate limit exceeded",
            "rate_limit_exceeded"
        )
    
    @staticmethod
    def server_error(message: str = "Internal server error") -> Dict[str, Any]:
        """Server error"""
        return OpenAIErrorAdapter.create_error_response(
            "server_error",
            message,
            "internal_error"
        )

class OpenAICompatibilityLayer:
    """
    Main compatibility layer for OpenAI API integration
    """
    
    def __init__(self):
        self.request_adapter = OpenAIRequestAdapter()
        self.response_adapter = OpenAIResponseAdapter()
        self.error_adapter = OpenAIErrorAdapter()
    
    def validate_openai_request(self, request: Dict[str, Any], endpoint: str) -> bool:
        """
        Validate OpenAI-compatible request
        """
        if endpoint == "speech":
            required_fields = ["model", "input"]
        elif endpoint == "music":
            required_fields = ["model", "prompt"]
        else:
            return False
        
        for field in required_fields:
            if field not in request:
                return False
        
        return True
    
    def convert_request(self, request: Dict[str, Any], endpoint: str) -> Dict[str, Any]:
        """
        Convert OpenAI request to ACE-Steps format
        """
        if endpoint == "speech":
            return self.request_adapter.speech_to_ace_steps(request)
        elif endpoint == "music":
            return self.request_adapter.music_to_ace_steps(request)
        else:
            raise ValueError(f"Unknown endpoint: {endpoint}")
    
    def convert_response(
        self, 
        ace_response: Dict[str, Any], 
        request_id: str,
        model: str,
        generation_time: float,
        endpoint: str
    ) -> Dict[str, Any]:
        """
        Convert ACE-Steps response to OpenAI format
        """
        if endpoint == "speech":
            return self.response_adapter.ace_steps_to_speech(
                ace_response, request_id, model, generation_time
            )
        elif endpoint == "music":
            return self.response_adapter.ace_steps_to_music(
                ace_response, request_id, model, generation_time
            )
        else:
            raise ValueError(f"Unknown endpoint: {endpoint}")
    
    def create_error_response(
        self, 
        error_type: str, 
        message: str, 
        code: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create OpenAI-compatible error response
        """
        return self.error_adapter.create_error_response(error_type, message, code)

class OpenAIModelRegistry:
    """
    Registry for OpenAI-compatible models
    """
    
    def __init__(self):
        self.models = {
            "ace-step-audio": {
                "id": "ace-step-audio",
                "object": "model",
                "created": int(datetime.now().timestamp()),
                "owned_by": "ace-steps",
                "permission": [],
                "root": "ace-step-audio",
                "parent": None,
                "capabilities": ["speech_generation", "audio_generation"]
            },
            "ace-step-music": {
                "id": "ace-step-music",
                "object": "model", 
                "created": int(datetime.now().timestamp()),
                "owned_by": "ace-steps",
                "permission": [],
                "root": "ace-step-music",
                "parent": None,
                "capabilities": ["music_generation", "lyrics_generation"]
            }
        }
    
    def get_model(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Get model information"""
        return self.models.get(model_id)
    
    def list_models(self) -> List[Dict[str, Any]]:
        """List all available models"""
        return list(self.models.values())
    
    def is_model_supported(self, model_id: str) -> bool:
        """Check if model is supported"""
        return model_id in self.models
    
    def get_model_capabilities(self, model_id: str) -> List[str]:
        """Get model capabilities"""
        model = self.get_model(model_id)
        return model.get("capabilities", []) if model else []
