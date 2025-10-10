import bentoml

from fastapi import FastAPI, Request, HTTPException
from acestep.pipeline_ace_step import ACEStepPipeline
from typing import Optional, List, Dict, Any
from pydantic import BaseModel

app = FastAPI(
    title="ACE-Step Music Generation API",
    description="OpenAI-compatible text-to-music generation service",
    version="1.0.0"
)

# Pydantic models for request/response
class SpeechRequest(BaseModel):
    model: str = "acesteps"
    input: str
    voice: str = "ash"
    response_format: str = "mp3"
    speed: float = 1.0
    instructions: str = "pop, rap, electronic, blues, hip-house, rhythm and blues"

class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str = "acestep"

class ModelsResponse(BaseModel):
    object: str = "list"
    data: List[ModelInfo]

@bentoml.asgi_app(app, path="/v1")
@bentoml.service(name="acesteps_service")
class ACEStepService(bentoml.Service):

    def __init__(self, checkpoint_path: str = None, bf16: bool = True, torch_compile: bool = True, device_id: int = 0):
        super().__init__(name="acesteps_service")
        # Handle empty or None checkpoint_path - ACEStepPipeline will auto-download from HuggingFace
        if not checkpoint_path or checkpoint_path.strip() == "":
            checkpoint_path = None
        self.model = ACEStepPipeline(checkpoint_dir=checkpoint_path, dtype="bfloat16" if bf16 else "float32", torch_compile=torch_compile, device_id=device_id)

    # Health check endpoint
    @app.get("/healthz")
    async def health_check(self):
        return {"status": "healthy", "service": "acesteps"}
    
    # OpenAI-compatible models endpoint
    @app.get("/models")
    async def list_models(self):
        models = [
            ModelInfo(
                id="acesteps",
                created=1700000000,
                owned_by="acestep"
            )
        ]
        return ModelsResponse(data=models)
    
    # OpenAI-compatible speech endpoint
    @app.post("/audio/speech")
    async def create_speech(self, request: SpeechRequest):
        # Validate required fields
        if not request.input:
            raise HTTPException(status_code=400, detail="Missing required field: input")
        
        # Validate parameter values
        if request.response_format not in ["mp3", "wav"]:
            raise HTTPException(status_code=400, detail="response_format must be 'mp3' or 'wav'")
        
        if request.speed <= 0:
            raise HTTPException(status_code=400, detail="speed must be a positive number")
        
        if len(request.input.strip()) == 0:
            raise HTTPException(status_code=400, detail="input cannot be empty")
        
        # Generate audio using the model with individual parameters
        result = self.model(
            format=request.response_format,
            audio_duration=240,
            prompt=request.instructions,
            lyrics=request.input,
            infer_step=60,
            guidance_scale=15.0,
            scheduler_type="euler",
            cfg_type="apg",
            omega_scale=10.0,
            manual_seeds=[3299954530],
            guidance_interval=0.5,
            guidance_interval_decay=0.0,
            min_guidance_scale=3.0,
            use_erg_tag=True,
            use_erg_lyric=True,
            use_erg_diffusion=True,
            oss_steps="",
            guidance_scale_text=0.0,
            guidance_scale_lyric=0.0,
            save_path=None
        )
        
        # Return the generated audio (assuming the model returns audio data)
        return result