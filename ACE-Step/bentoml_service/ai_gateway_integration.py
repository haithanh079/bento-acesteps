"""
AI Gateway Integration for ACE-Steps OpenAI-Compatible Service
Provides testing and validation for AI Gateway routing
"""

import asyncio
import json
import time
from typing import Dict, Any, List, Optional
import httpx
import requests
from dataclasses import dataclass

@dataclass
class AIGatewayConfig:
    """AI Gateway configuration"""
    base_url: str
    api_key: str
    timeout: int = 30
    retry_attempts: int = 3
    retry_delay: float = 1.0

class AIGatewayTester:
    """
    Test AI Gateway integration with ACE-Steps service
    """
    
    def __init__(self, gateway_config: AIGatewayConfig):
        self.config = gateway_config
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {gateway_config.api_key}",
            "Content-Type": "application/json"
        })
        self.session.timeout = gateway_config.timeout
    
    def test_service_discovery(self) -> Dict[str, Any]:
        """Test if AI Gateway can discover ACE-Steps service"""
        try:
            # Test health endpoint
            response = self.session.get(f"{self.config.base_url}/health")
            if response.status_code == 200:
                return {
                    "status": "success",
                    "message": "Service discovery successful",
                    "health": response.json()
                }
            else:
                return {
                    "status": "error",
                    "message": f"Health check failed: {response.status_code}",
                    "error": response.text
                }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Service discovery failed: {str(e)}"
            }
    
    def test_model_routing(self) -> Dict[str, Any]:
        """Test model routing through AI Gateway"""
        try:
            # Test model listing
            response = self.session.get(f"{self.config.base_url}/v1/models")
            if response.status_code == 200:
                models = response.json()
                ace_models = [m for m in models.get("data", []) if "ace-step" in m.get("id", "")]
                
                return {
                    "status": "success",
                    "message": f"Found {len(ace_models)} ACE-Steps models",
                    "models": [m["id"] for m in ace_models]
                }
            else:
                return {
                    "status": "error",
                    "message": f"Model listing failed: {response.status_code}",
                    "error": response.text
                }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Model routing test failed: {str(e)}"
            }
    
    def test_speech_routing(self) -> Dict[str, Any]:
        """Test speech generation routing"""
        try:
            payload = {
                "model": "ace-step-audio",
                "input": "AI Gateway integration test",
                "voice": "alloy",
                "response_format": "mp3",
                "duration": 10,
                "infer_steps": 20
            }
            
            response = self.session.post(
                f"{self.config.base_url}/v1/audio/speech",
                json=payload
            )
            
            if response.status_code == 200:
                result = response.json()
                return {
                    "status": "success",
                    "message": "Speech routing successful",
                    "model": result.get("model"),
                    "generation_time": result.get("usage", {}).get("generation_time", 0)
                }
            else:
                return {
                    "status": "error",
                    "message": f"Speech routing failed: {response.status_code}",
                    "error": response.text
                }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Speech routing test failed: {str(e)}"
            }
    
    def test_music_routing(self) -> Dict[str, Any]:
        """Test music generation routing"""
        try:
            payload = {
                "model": "ace-step-music",
                "prompt": "A test melody for AI Gateway integration",
                "style": "pop",
                "response_format": "mp3",
                "duration": 15,
                "infer_steps": 30
            }
            
            response = self.session.post(
                f"{self.config.base_url}/v1/audio/music",
                json=payload
            )
            
            if response.status_code == 200:
                result = response.json()
                return {
                    "status": "success",
                    "message": "Music routing successful",
                    "model": result.get("model"),
                    "generation_time": result.get("usage", {}).get("generation_time", 0)
                }
            else:
                return {
                    "status": "error",
                    "message": f"Music routing failed: {response.status_code}",
                    "error": response.text
                }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Music routing test failed: {str(e)}"
            }
    
    def test_error_handling(self) -> Dict[str, Any]:
        """Test error handling and routing"""
        try:
            # Test invalid model
            payload = {
                "model": "invalid-model",
                "input": "Test error handling"
            }
            
            response = self.session.post(
                f"{self.config.base_url}/v1/audio/speech",
                json=payload
            )
            
            if response.status_code == 400 or response.status_code == 404:
                return {
                    "status": "success",
                    "message": "Error handling working correctly",
                    "status_code": response.status_code
                }
            else:
                return {
                    "status": "warning",
                    "message": f"Unexpected status code: {response.status_code}",
                    "response": response.text
                }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error handling test failed: {str(e)}"
            }
    
    def test_performance(self) -> Dict[str, Any]:
        """Test performance and latency"""
        try:
            start_time = time.time()
            
            payload = {
                "model": "ace-step-audio",
                "input": "Performance test",
                "duration": 5,
                "infer_steps": 10
            }
            
            response = self.session.post(
                f"{self.config.base_url}/v1/audio/speech",
                json=payload
            )
            
            total_time = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                generation_time = result.get("usage", {}).get("generation_time", 0)
                
                return {
                    "status": "success",
                    "message": "Performance test completed",
                    "total_time": total_time,
                    "generation_time": generation_time,
                    "overhead": total_time - generation_time
                }
            else:
                return {
                    "status": "error",
                    "message": f"Performance test failed: {response.status_code}",
                    "error": response.text
                }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Performance test failed: {str(e)}"
            }
    
    def run_full_test_suite(self) -> Dict[str, Any]:
        """Run complete test suite"""
        tests = [
            ("Service Discovery", self.test_service_discovery),
            ("Model Routing", self.test_model_routing),
            ("Speech Routing", self.test_speech_routing),
            ("Music Routing", self.test_music_routing),
            ("Error Handling", self.test_error_handling),
            ("Performance", self.test_performance)
        ]
        
        results = {}
        passed = 0
        failed = 0
        
        for test_name, test_func in tests:
            print(f"Running {test_name}...")
            result = test_func()
            results[test_name] = result
            
            if result["status"] == "success":
                passed += 1
                print(f"  ✓ {test_name}: {result['message']}")
            else:
                failed += 1
                print(f"  ✗ {test_name}: {result['message']}")
        
        return {
            "summary": {
                "total_tests": len(tests),
                "passed": passed,
                "failed": failed,
                "success_rate": f"{(passed / len(tests)) * 100:.1f}%"
            },
            "results": results
        }

class AIGatewayLoadTester:
    """
    Load testing for AI Gateway integration
    """
    
    def __init__(self, gateway_config: AIGatewayConfig):
        self.config = gateway_config
    
    async def test_concurrent_requests(self, num_requests: int = 10) -> Dict[str, Any]:
        """Test concurrent request handling"""
        async with httpx.AsyncClient() as client:
            tasks = []
            
            for i in range(num_requests):
                task = self._make_request(client, i)
                tasks.append(task)
            
            start_time = time.time()
            results = await asyncio.gather(*tasks, return_exceptions=True)
            total_time = time.time() - start_time
            
            successful = sum(1 for r in results if not isinstance(r, Exception) and r.get("status") == "success")
            failed = num_requests - successful
            
            return {
                "total_requests": num_requests,
                "successful": successful,
                "failed": failed,
                "total_time": total_time,
                "requests_per_second": num_requests / total_time,
                "average_response_time": total_time / num_requests
            }
    
    async def _make_request(self, client: httpx.AsyncClient, request_id: int) -> Dict[str, Any]:
        """Make a single request"""
        try:
            payload = {
                "model": "ace-step-audio",
                "input": f"Concurrent test request {request_id}",
                "duration": 5,
                "infer_steps": 10
            }
            
            response = await client.post(
                f"{self.config.base_url}/v1/audio/speech",
                json=payload,
                headers={"Authorization": f"Bearer {self.config.api_key}"},
                timeout=self.config.timeout
            )
            
            if response.status_code == 200:
                return {"status": "success", "request_id": request_id}
            else:
                return {"status": "error", "request_id": request_id, "error": response.text}
        except Exception as e:
            return {"status": "error", "request_id": request_id, "error": str(e)}

def main():
    """Main testing function"""
    print("AI Gateway Integration Test")
    print("=" * 40)
    
    # Configure AI Gateway
    gateway_config = AIGatewayConfig(
        base_url="http://localhost:8080",  # AI Gateway URL
        api_key="sk-test",
        timeout=60
    )
    
    # Run tests
    tester = AIGatewayTester(gateway_config)
    results = tester.run_full_test_suite()
    
    # Print summary
    print(f"\nTest Summary:")
    print(f"Total Tests: {results['summary']['total_tests']}")
    print(f"Passed: {results['summary']['passed']}")
    print(f"Failed: {results['summary']['failed']}")
    print(f"Success Rate: {results['summary']['success_rate']}")
    
    # Run load test
    print(f"\nRunning Load Test...")
    load_tester = AIGatewayLoadTester(gateway_config)
    load_results = asyncio.run(load_tester.test_concurrent_requests(5))
    
    print(f"Load Test Results:")
    print(f"Total Requests: {load_results['total_requests']}")
    print(f"Successful: {load_results['successful']}")
    print(f"Failed: {load_results['failed']}")
    print(f"Requests/Second: {load_results['requests_per_second']:.2f}")
    print(f"Average Response Time: {load_results['average_response_time']:.2f}s")

if __name__ == "__main__":
    main()
