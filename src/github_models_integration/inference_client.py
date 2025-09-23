"""
GitHub Models Inference API Client

This module provides integration with the official GitHub Models inference API
for running chat completions and accessing model catalogs.

Based on: https://docs.github.com/en/rest/models/inference
"""

import asyncio
import aiohttp
import json
import logging
from typing import Dict, List, Any, Optional, AsyncGenerator
from dataclasses import dataclass, asdict
from datetime import datetime
import os


@dataclass
class ChatMessage:
    """Represents a chat message for inference requests."""
    role: str  # "system", "user", "assistant", "developer"
    content: str


@dataclass
class InferenceRequest:
    """Represents an inference request to GitHub Models API."""
    model: str
    messages: List[ChatMessage]
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    stop: Optional[List[str]] = None
    stream: bool = False
    seed: Optional[int] = None
    response_format: Optional[Dict[str, Any]] = None
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[str] = None


@dataclass
class InferenceResponse:
    """Represents a response from GitHub Models inference API."""
    success: bool
    choices: Optional[List[Dict[str, Any]]] = None
    usage: Optional[Dict[str, Any]] = None
    model: Optional[str] = None
    error: Optional[str] = None
    rate_limit_remaining: Optional[int] = None


@dataclass
class ModelInfo:
    """Represents model information from the catalog."""
    id: str
    name: str
    publisher: str
    description: Optional[str] = None
    capabilities: Optional[List[str]] = None
    context_length: Optional[int] = None


class GitHubModelsInferenceClient:
    """
    Client for GitHub Models inference API.
    
    This client provides access to the official GitHub Models inference API
    for running chat completions with organizational attribution.
    """
    
    def __init__(self, organization: Optional[str] = None):
        """Initialize the GitHub Models inference client."""
        self.organization = organization
        self.token = self._get_auth_token()
        self.session = None
        self.logger = self._setup_logging()
        
        # GitHub Models API endpoints
        self.base_url = "https://models.github.ai"
        self.inference_url = f"{self.base_url}/inference/chat/completions"
        self.catalog_url = f"{self.base_url}/catalog/models"
        
    def _get_auth_token(self) -> Optional[str]:
        """Get authentication token from environment."""
        return os.getenv("GITHUB_MODELS_TOKEN")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for GitHub Models inference client."""
        logger = logging.getLogger("github_models_inference")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    async def initialize(self):
        """Initialize the GitHub Models inference client."""
        try:
            # Initialize HTTP session with authentication
            headers = {}
            if self.token:
                headers["Authorization"] = f"Bearer {self.token}"
            
            headers.update({
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
                "Content-Type": "application/json",
                "User-Agent": "Lenovo-AAITC-Models-Client/1.0"
            })
            
            self.session = aiohttp.ClientSession(
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=60)
            )
            
            self.logger.info("GitHub Models inference client initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize GitHub Models inference client: {e}")
            raise
    
    async def get_model_catalog(self) -> List[ModelInfo]:
        """Get the catalog of available models."""
        try:
            async with self.session.get(self.catalog_url) as response:
                if response.status == 200:
                    data = await response.json()
                    models = []
                    
                    for model_data in data.get("data", []):
                        model = ModelInfo(
                            id=model_data.get("id", ""),
                            name=model_data.get("name", ""),
                            publisher=model_data.get("publisher", ""),
                            description=model_data.get("description"),
                            capabilities=model_data.get("capabilities", []),
                            context_length=model_data.get("context_length")
                        )
                        models.append(model)
                    
                    self.logger.info(f"Retrieved {len(models)} models from catalog")
                    return models
                    
                else:
                    self.logger.error(f"Failed to get model catalog: HTTP {response.status}")
                    return []
                    
        except Exception as e:
            self.logger.error(f"Error getting model catalog: {e}")
            return []
    
    async def run_inference(
        self, 
        request: InferenceRequest,
        stream: bool = False
    ) -> InferenceResponse:
        """Run an inference request."""
        try:
            # Prepare request payload
            payload = {
                "model": request.model,
                "messages": [asdict(msg) for msg in request.messages]
            }
            
            # Add optional parameters
            if request.max_tokens is not None:
                payload["max_tokens"] = request.max_tokens
            if request.temperature is not None:
                payload["temperature"] = request.temperature
            if request.top_p is not None:
                payload["top_p"] = request.top_p
            if request.frequency_penalty is not None:
                payload["frequency_penalty"] = request.frequency_penalty
            if request.presence_penalty is not None:
                payload["presence_penalty"] = request.presence_penalty
            if request.stop is not None:
                payload["stop"] = request.stop
            if request.stream:
                payload["stream"] = True
            if request.seed is not None:
                payload["seed"] = request.seed
            if request.response_format is not None:
                payload["response_format"] = request.response_format
            if request.tools is not None:
                payload["tools"] = request.tools
            if request.tool_choice is not None:
                payload["tool_choice"] = request.tool_choice
            
            # Choose endpoint based on organization attribution
            url = self.inference_url
            if self.organization:
                url = f"{self.base_url}/inference/orgs/{self.organization}/chat/completions"
            
            async with self.session.post(url, json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    return InferenceResponse(
                        success=True,
                        choices=data.get("choices", []),
                        usage=data.get("usage"),
                        model=data.get("model"),
                        rate_limit_remaining=self._get_rate_limit_remaining(response)
                    )
                else:
                    error_text = await response.text()
                    return InferenceResponse(
                        success=False,
                        error=f"HTTP {response.status}: {error_text}"
                    )
                    
        except Exception as e:
            self.logger.error(f"Error running inference: {e}")
            return InferenceResponse(
                success=False,
                error=str(e)
            )
    
    async def run_inference_stream(
        self, 
        request: InferenceRequest
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Run a streaming inference request."""
        try:
            # Prepare request payload
            payload = {
                "model": request.model,
                "messages": [asdict(msg) for msg in request.messages],
                "stream": True
            }
            
            # Add optional parameters
            if request.max_tokens is not None:
                payload["max_tokens"] = request.max_tokens
            if request.temperature is not None:
                payload["temperature"] = request.temperature
            if request.top_p is not None:
                payload["top_p"] = request.top_p
            
            # Choose endpoint based on organization attribution
            url = self.inference_url
            if self.organization:
                url = f"{self.base_url}/inference/orgs/{self.organization}/chat/completions"
            
            async with self.session.post(url, json=payload) as response:
                if response.status == 200:
                    async for line in response.content:
                        if line:
                            line_str = line.decode('utf-8').strip()
                            if line_str.startswith('data: '):
                                data_str = line_str[6:]  # Remove 'data: ' prefix
                                if data_str == '[DONE]':
                                    break
                                try:
                                    data = json.loads(data_str)
                                    yield data
                                except json.JSONDecodeError:
                                    continue
                else:
                    error_text = await response.text()
                    yield {"error": f"HTTP {response.status}: {error_text}"}
                    
        except Exception as e:
            self.logger.error(f"Error running streaming inference: {e}")
            yield {"error": str(e)}
    
    async def test_small_models(self) -> Dict[str, Any]:
        """Test inference with small models for Phase 2."""
        try:
            # Get model catalog
            models = await self.get_model_catalog()
            
            # Filter for small models
            small_models = []
            for model in models:
                model_name_lower = model.name.lower()
                if any(indicator in model_name_lower for indicator in ["3b", "mini", "small", "mobile", "edge"]):
                    small_models.append(model)
            
            # Test inference with first available small model
            test_results = {}
            if small_models:
                test_model = small_models[0]
                
                # Create test request
                test_request = InferenceRequest(
                    model=test_model.id,
                    messages=[
                        ChatMessage(role="system", content="You are a helpful assistant optimized for mobile deployment."),
                        ChatMessage(role="user", content="Hello! Can you help me test mobile inference?")
                    ],
                    max_tokens=50,
                    temperature=0.7
                )
                
                # Run inference
                response = await self.run_inference(test_request)
                
                test_results = {
                    "model_tested": test_model.id,
                    "model_name": test_model.name,
                    "publisher": test_model.publisher,
                    "inference_success": response.success,
                    "response_preview": response.choices[0]["message"]["content"][:100] if response.success and response.choices else None,
                    "usage": response.usage,
                    "error": response.error
                }
            
            return {
                "small_models_found": len(small_models),
                "available_models": [{"id": m.id, "name": m.name, "publisher": m.publisher} for m in small_models],
                "test_result": test_results
            }
            
        except Exception as e:
            self.logger.error(f"Error testing small models: {e}")
            return {"error": str(e)}
    
    async def run_phase2_evaluation(self, model_id: str, prompt: str) -> Dict[str, Any]:
        """Run Phase 2 evaluation with a specific model."""
        try:
            # Create evaluation request
            evaluation_request = InferenceRequest(
                model=model_id,
                messages=[
                    ChatMessage(role="system", content="You are an AI assistant designed for mobile and edge deployment. Provide concise, helpful responses."),
                    ChatMessage(role="user", content=prompt)
                ],
                max_tokens=100,
                temperature=0.7,
                top_p=0.9
            )
            
            start_time = datetime.now()
            
            # Run inference
            response = await self.run_inference(evaluation_request)
            
            end_time = datetime.now()
            inference_time = (end_time - start_time).total_seconds() * 1000  # Convert to ms
            
            return {
                "model_id": model_id,
                "prompt": prompt,
                "success": response.success,
                "response": response.choices[0]["message"]["content"] if response.success and response.choices else None,
                "inference_time_ms": inference_time,
                "usage": response.usage,
                "error": response.error,
                "timestamp": end_time.isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error running Phase 2 evaluation: {e}")
            return {"error": str(e)}
    
    def _get_rate_limit_remaining(self, response: aiohttp.ClientResponse) -> Optional[int]:
        """Get remaining rate limit from response headers."""
        remaining = response.headers.get("X-RateLimit-Remaining")
        return int(remaining) if remaining else None
    
    async def shutdown(self):
        """Shutdown the GitHub Models inference client."""
        try:
            if self.session:
                await self.session.close()
            self.logger.info("GitHub Models inference client shutdown complete")
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")


# Example usage and testing
async def main():
    """Example usage of GitHub Models inference client."""
    client = GitHubModelsInferenceClient(organization="Brantwood")
    
    try:
        await client.initialize()
        
        # Test small models
        print("Testing small models...")
        test_results = await client.test_small_models()
        print(f"Small models found: {test_results.get('small_models_found', 0)}")
        
        if test_results.get("test_result"):
            result = test_results["test_result"]
            print(f"Test model: {result['model_name']}")
            print(f"Inference success: {result['inference_success']}")
            if result.get("response_preview"):
                print(f"Response preview: {result['response_preview']}")
        
        # Run Phase 2 evaluation
        if test_results.get("available_models"):
            model = test_results["available_models"][0]
            print(f"\nRunning Phase 2 evaluation with {model['name']}...")
            
            evaluation = await client.run_phase2_evaluation(
                model["id"],
                "Explain how this model is optimized for mobile deployment."
            )
            
            if evaluation.get("success"):
                print(f"Evaluation successful!")
                print(f"Inference time: {evaluation['inference_time_ms']:.1f}ms")
                print(f"Response: {evaluation['response']}")
            else:
                print(f"Evaluation failed: {evaluation.get('error')}")
        
    except Exception as e:
        print(f"Error: {e}")
    finally:
        await client.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
