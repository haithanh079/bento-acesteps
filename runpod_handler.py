"""
ACESTEPS RunPod Serverless Worker Handler

This module implements a RunPod serverless worker for ACESTEPS music generation.
It processes job requests, generates audio using ACEStepPipeline, and uploads
results to S3 storage.
"""

import os
import uuid
import tempfile
from typing import Dict, Any, Optional
from loguru import logger

import runpod
from acestep.pipeline_ace_step import ACEStepPipeline
from runpod.serverless.utils import upload_file_to_bucket

# Global pipeline instance (initialized once)
pipeline = None


def initialize_pipeline(input_data: Dict[str, Any]) -> ACEStepPipeline:
    """
    Initialize the ACEStepPipeline with configuration from input or defaults.
    
    Args:
        input_data: Job input data containing pipeline configuration
        
    Returns:
        Initialized ACEStepPipeline instance
    """
    checkpoint_path = input_data.get("checkpoint_path", "")
    bf16 = input_data.get("bf16", True)
    torch_compile = input_data.get("torch_compile", False)
    device_id = input_data.get("device_id", 0)
    cpu_offload = input_data.get("cpu_offload", False)
    overlapped_decode = input_data.get("overlapped_decode", False)
    
    # Set CUDA device
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
    
    # Initialize pipeline
    pipeline = ACEStepPipeline(
        checkpoint_dir=checkpoint_path if checkpoint_path else None,
        dtype="bfloat16" if bf16 else "float32",
        torch_compile=torch_compile,
        cpu_offload=cpu_offload,
        overlapped_decode=overlapped_decode,
    )
    
    logger.info("ACEStepPipeline initialized successfully")
    return pipeline


def extract_inputs(job_input: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract and validate input parameters from job input.
    
    Args:
        job_input: Raw job input dictionary
        
    Returns:
        Dictionary with validated and defaulted parameters
    """
    # Required field
    if "prompt" not in job_input:
        raise ValueError("'prompt' is required in job input")
    
    # Extract parameters with defaults
    inputs = {
        "prompt": job_input["prompt"],
        "lyrics": job_input.get("lyrics", ""),
        "audio_duration": job_input.get("audio_duration", 60.0),
        "infer_step": job_input.get("infer_step", 60),
        "guidance_scale": job_input.get("guidance_scale", 15.0),
        "scheduler_type": job_input.get("scheduler_type", "euler"),
        "cfg_type": job_input.get("cfg_type", "apg"),
        "omega_scale": job_input.get("omega_scale", 10.0),
        "manual_seeds": job_input.get("actual_seeds"),
        "guidance_interval": job_input.get("guidance_interval", 0.5),
        "guidance_interval_decay": job_input.get("guidance_interval_decay", 0.0),
        "min_guidance_scale": job_input.get("min_guidance_scale", 3.0),
        "use_erg_tag": job_input.get("use_erg_tag", True),
        "use_erg_lyric": job_input.get("use_erg_lyric", True),
        "use_erg_diffusion": job_input.get("use_erg_diffusion", True),
        "oss_steps": job_input.get("oss_steps", []),
        "guidance_scale_text": job_input.get("guidance_scale_text", 0.0),
        "guidance_scale_lyric": job_input.get("guidance_scale_lyric", 0.0),
        "task": job_input.get("task", "text2music"),
    }
    
    # Convert oss_steps to string format if it's a list
    if isinstance(inputs["oss_steps"], list):
        if len(inputs["oss_steps"]) > 0:
            inputs["oss_steps"] = ", ".join(map(str, inputs["oss_steps"]))
        else:
            inputs["oss_steps"] = None
    
    # Convert manual_seeds to list format if provided
    if inputs["manual_seeds"] is not None:
        if isinstance(inputs["manual_seeds"], list):
            inputs["manual_seeds"] = inputs["manual_seeds"]
        else:
            inputs["manual_seeds"] = [inputs["manual_seeds"]]
    
    return inputs


def generate_audio(pipeline: ACEStepPipeline, inputs: Dict[str, Any], job_id: str) -> str:
    """
    Generate audio using the ACEStepPipeline.
    
    Args:
        pipeline: Initialized ACEStepPipeline instance
        inputs: Dictionary with generation parameters
        job_id: Job ID for unique filename
        
    Returns:
        Path to generated audio file
    """
    # Create temporary directory for output
    temp_dir = tempfile.gettempdir()
    output_filename = f"acestep_{job_id}_{uuid.uuid4().hex[:8]}.wav"
    output_path = os.path.join(temp_dir, output_filename)
    
    logger.info(f"Generating audio with parameters: {inputs}")
    
    # Call pipeline to generate audio
    pipeline(
        format="wav",
        audio_duration=inputs["audio_duration"],
        prompt=inputs["prompt"],
        lyrics=inputs["lyrics"],
        infer_step=inputs["infer_step"],
        guidance_scale=inputs["guidance_scale"],
        scheduler_type=inputs["scheduler_type"],
        cfg_type=inputs["cfg_type"],
        omega_scale=inputs["omega_scale"],
        manual_seeds=inputs["manual_seeds"],
        guidance_interval=inputs["guidance_interval"],
        guidance_interval_decay=inputs["guidance_interval_decay"],
        min_guidance_scale=inputs["min_guidance_scale"],
        use_erg_tag=inputs["use_erg_tag"],
        use_erg_lyric=inputs["use_erg_lyric"],
        use_erg_diffusion=inputs["use_erg_diffusion"],
        oss_steps=inputs["oss_steps"],
        guidance_scale_text=inputs["guidance_scale_text"],
        guidance_scale_lyric=inputs["guidance_scale_lyric"],
        task=inputs["task"],
        save_path=output_path,
    )
    
    logger.info(f"Audio generated successfully: {output_path}")
    return output_path


def upload_audio(file_path: str, job_id: str) -> Optional[str]:
    """
    Upload audio file to S3 and return presigned URL.
    
    Args:
        file_path: Path to local audio file
        job_id: Job ID for unique filename
        
    Returns:
        Presigned S3 URL or None if upload fails
    """
    try:
        # Get S3 bucket credentials from environment variables
        bucket_creds = {
            "endpointUrl": os.getenv("BUCKET_ENDPOINT_URL"),
            "accessId": os.getenv("BUCKET_ACCESS_KEY_ID"),
            "accessSecret": os.getenv("BUCKET_SECRET_ACCESS_KEY"),
        }
        
        # Check if bucket credentials are available
        if not all(bucket_creds.values()):
            logger.warning("S3 bucket credentials not found. Skipping upload.")
            return None
        
        # Extract filename from path
        file_name = os.path.basename(file_path)
        
        # Optional: Get bucket name and prefix from environment
        bucket_name = os.getenv("BUCKET_NAME")
        prefix = os.getenv("BUCKET_PREFIX", "acestep-outputs")
        
        # Upload file to S3
        presigned_url = upload_file_to_bucket(
            file_name=file_name,
            file_location=file_path,
            bucket_creds=bucket_creds,
            bucket_name=bucket_name,
            prefix=prefix,
        )
        
        logger.info(f"Audio uploaded to S3: {presigned_url}")
        return presigned_url
        
    except Exception as e:
        logger.error(f"Failed to upload audio to S3: {str(e)}")
        return None


def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main handler function for RunPod serverless worker.
    
    Args:
        job: Job dictionary containing 'id' and 'input' keys
        
    Returns:
        Dictionary with status, output URL, and message
    """
    global pipeline
    
    try:
        job_id = job.get("id", "unknown")
        job_input = job.get("input", {})
        
        logger.info(f"Processing job {job_id}")
        
        # Check if worker should be refreshed
        refresh_worker = job_input.get("refresh_worker", False)
        if refresh_worker:
            logger.info("Refreshing worker: reinitializing pipeline")
            pipeline = None
        
        # Initialize pipeline if needed
        if pipeline is None:
            logger.info("Initializing ACEStepPipeline...")
            pipeline = initialize_pipeline(job_input)
        
        # Extract and validate inputs
        inputs = extract_inputs(job_input)
        
        # Generate audio
        output_path = generate_audio(pipeline, inputs, job_id)
        
        # Upload to S3
        output_url = upload_audio(output_path, job_id)
        
        # Prepare response
        response = {
            "status": "success",
            "output_path": output_path,
            "message": "Audio generated successfully",
        }
        
        if output_url:
            response["output_url"] = output_url
        
        # Clean up local file if upload was successful
        if output_url and os.path.exists(output_path):
            try:
                os.remove(output_path)
                logger.info(f"Cleaned up local file: {output_path}")
            except Exception as e:
                logger.warning(f"Failed to clean up local file: {str(e)}")
        
        return response
        
    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        return {"error": f"Invalid input: {str(e)}"}
    except Exception as e:
        logger.error(f"Error processing job: {str(e)}", exc_info=True)
        return {"error": f"Error generating audio: {str(e)}"}


# Start the RunPod serverless worker
if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})

