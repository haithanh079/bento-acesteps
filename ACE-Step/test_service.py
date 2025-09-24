#!/usr/bin/env python3
"""
Test script for ACE-Step BentoML service
"""

import asyncio
import pytest
from pathlib import Path
import tempfile
import os

# Mock the dependencies for testing
class MockPipeline:
    def __call__(self, **kwargs):
        # Create a dummy audio file for testing
        output_path = kwargs.get('save_path', 'test_output.wav')
        Path(output_path).touch()
        return [output_path]

class MockConfig:
    checkpoint_path = ""
    device_id = 0
    dtype = "bfloat16"
    torch_compile = False
    cpu_offload = False
    overlapped_decode = False
    default_duration = 30.0
    max_duration = 240.0
    max_inference_steps = 200
    output_dir = tempfile.mkdtemp()
    cleanup_files = True
    file_retention_hours = 24
    enable_cors = True

# Test the service components
def test_audio_generation_request_validation():
    """Test request validation"""
    from service import AudioGenerationRequest
    
    # Valid request
    request = AudioGenerationRequest(
        prompt="Test prompt",
        duration=30.0,
        num_inference_steps=60
    )
    assert request.prompt == "Test prompt"
    assert request.duration == 30.0
    
    # Test validation
    try:
        invalid_request = AudioGenerationRequest(
            prompt="",  # Empty prompt should fail
            duration=30.0
        )
        assert False, "Should have failed validation"
    except ValueError:
        pass

def test_file_manager():
    """Test file manager functionality"""
    from service import FileManager
    
    # Mock config for testing
    import service
    service.config = MockConfig()
    
    fm = FileManager()
    
    # Test output path creation
    path = fm.create_output_path("wav")
    assert path.suffix == ".wav"
    assert path.parent == Path(MockConfig.output_dir)
    
    # Test file info
    test_file = Path(MockConfig.output_dir) / "test.wav"
    test_file.touch()
    
    info = fm.get_file_info(test_file)
    assert info is not None
    assert info["path"] == str(test_file)
    
    # Cleanup
    test_file.unlink()

async def test_service_initialization():
    """Test service initialization"""
    # This would require mocking the ACEStepPipeline
    # For now, just test that the service class can be imported
    from service import acestepaudoservice
    assert acestepaudoservice is not None

def test_openai_compatibility():
    """Test OpenAI API compatibility"""
    from service import AudioGenerationRequest, AudioGenerationResponse, AudioData
    
    # Test request structure
    request = AudioGenerationRequest(
        model="ace-step-v1",
        prompt="Test prompt",
        duration=30.0
    )
    
    # Test response structure
    audio_data = AudioData(
        url="/v1/audio/files/test.wav",
        revised_prompt="Test prompt"
    )
    
    response = AudioGenerationResponse(
        id="test_id",
        created=1234567890,
        model="ace-step-v1",
        data=[audio_data.dict()]
    )
    
    assert response.object == "audio.generation"
    assert len(response.data) == 1

if __name__ == "__main__":
    print("Running ACE-Step BentoML service tests...")
    
    try:
        test_audio_generation_request_validation()
        print("✓ Request validation test passed")
        
        test_file_manager()
        print("✓ File manager test passed")
        
        asyncio.run(test_service_initialization())
        print("✓ Service initialization test passed")
        
        test_openai_compatibility()
        print("✓ OpenAI compatibility test passed")
        
        print("\nAll tests passed! 🎉")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        exit(1)
