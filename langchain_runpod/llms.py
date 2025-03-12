"""Wrapper around RunPod's LLM Inference API."""

import json
import logging
import os
from typing import Any, Dict, List, Optional, Union, Iterator

import httpx
from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models.llms import LLM
from langchain_core.outputs import GenerationChunk
from pydantic import Field, PrivateAttr, model_validator

logger = logging.getLogger(__name__)


class RunPod(LLM):
    """LLM model wrapper for RunPod API.

    To use, you should have the ``langchain-runpod`` package installed, and the
    environment variable ``RUNPOD_API_KEY`` set with your API key, or pass it
    as the ``api_key`` parameter.

    Example:
        .. code-block:: python

            from langchain_runpod import RunPod
            
            llm = RunPod(endpoint_id="your-endpoint-id")
            llm.invoke("Tell me a joke")
    """
    
    endpoint_id: str = Field(..., description="The RunPod endpoint ID to use.")
    
    model_name: str = Field(default="", description="Name of the model for metadata.")
    
    api_key: Optional[str] = None
    """RunPod API key. If not provided, will look for RUNPOD_API_KEY env var."""
    
    api_base: str = "https://api.runpod.ai/v2"
    """Base URL for the RunPod API."""
    
    temperature: Optional[float] = None
    """Sampling temperature."""
    
    max_tokens: Optional[int] = None
    """Maximum number of tokens to generate."""
    
    top_p: Optional[float] = None
    """Top-p sampling parameter."""
    
    top_k: Optional[int] = None
    """Top-k sampling parameter."""
    
    stop: Optional[List[str]] = None
    """List of strings to stop generation when encountered."""
    
    timeout: Optional[int] = None
    """Timeout for requests in seconds."""
    
    streaming: bool = False
    """Whether to stream the results."""
    
    _client: httpx.Client = PrivateAttr()
    _async_client: Optional[httpx.AsyncClient] = PrivateAttr(default=None)
    
    @model_validator(mode='before')
    @classmethod
    def validate_environment(cls, data: Dict) -> Dict:
        """Validate that api key exists in environment."""
        api_key = data.get("api_key")
        if api_key is None:
            api_key = os.environ.get("RUNPOD_API_KEY")
            if api_key is None:
                raise ValueError(
                    "RunPod API key must be provided either through "
                    "the api_key parameter or as the environment variable "
                    "RUNPOD_API_KEY."
                )
            data["api_key"] = api_key
        
        # If no model name was provided, use endpoint_id as a fallback
        if not data.get("model_name"):
            data["model_name"] = f"runpod-endpoint-{data.get('endpoint_id', 'unknown')}"
            
        return data
    
    def __init__(self, **kwargs: Any) -> None:
        """Initialize the RunPod instance."""
        super().__init__(**kwargs)
        self._client = httpx.Client(timeout=self.timeout or 60.0)
    
    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "runpod"
    
    def _get_params(self, stop: Optional[List[str]] = None) -> Dict[str, Any]:
        """Get the parameters to use for the request."""
        params = {}
        
        if self.temperature is not None:
            params["temperature"] = self.temperature
            
        if self.max_tokens is not None:
            params["max_tokens"] = self.max_tokens
            
        if self.top_p is not None:
            params["top_p"] = self.top_p
            
        if self.top_k is not None:
            params["top_k"] = self.top_k
            
        if stop := self.stop or stop:
            params["stop"] = stop
            
        return params
    
    def _get_ls_params(self, stop: Optional[List[str]] = None, **kwargs) -> Dict[str, Any]:
        """Get the parameters used for LangSmith tracking."""
        return {
            "ls_provider": "runpod",
            "ls_model_name": self.model_name,
            "ls_model_type": "llm",
            "ls_temperature": self.temperature,
            "ls_max_tokens": self.max_tokens,
            "ls_stop": stop or self.stop,
        }
    
    def _process_response(self, response: Dict[str, Any]) -> str:
        """Process the API response and extract the generated text."""
        logger.debug(f"Raw response: {response}")
        
        # Specific handling for the response format seen in integration tests
        if isinstance(response.get("output"), list):
            output_list = response["output"]
            
            # If the output is a list with a format like [{"choices": [{"tokens": [...]}]}]
            if output_list and isinstance(output_list[0], dict) and "choices" in output_list[0]:
                first_item = output_list[0]
                choices = first_item.get("choices", [])
                
                if choices and isinstance(choices[0], dict) and "tokens" in choices[0]:
                    tokens = choices[0]["tokens"]
                    # Handle both cases: tokens as a list or as a single string
                    if isinstance(tokens, list):
                        # Integration test expects this to be joined into a string
                        return "".join(tokens)
                    return str(tokens)
            
            # If it's a simple list of strings
            if all(isinstance(item, str) for item in output_list):
                return "".join(output_list)
                
            # Fallback for any other list format - convert to string
            return str(output_list)
        
        # Handle the case where "output" is directly a string
        if isinstance(response.get("output"), str):
            return response["output"]
            
        # Handle the case where "output" is a dict
        if isinstance(response.get("output"), dict):
            output_dict = response["output"]
            for key in ["text", "content", "message", "generated_text", "response"]:
                if key in output_dict and isinstance(output_dict[key], str):
                    return output_dict[key]
            
            # If no recognizable field, return as string
            return str(output_dict)
        
        # Ultimate fallback: return the entire response as a string if we can't extract text
        logger.warning(f"Unrecognized response format: {response}")
        if "output" in response:
            return str(response["output"])
        return str(response)
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Call the RunPod API and return the generated text."""
        # Prepare request payload
        payload = {
            "input": {
                "prompt": prompt,
                **self._get_params(stop),
            }
        }
        
        # Add any additional kwargs to the payload
        for key, value in kwargs.items():
            if key not in payload["input"]:
                payload["input"][key] = value
        
        # API request
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        
        try:
            url = f"{self.api_base}/{self.endpoint_id}/run"
            response = self._client.post(
                url,
                headers=headers,
                json=payload,
                timeout=self.timeout or 60.0,
            )
            response.raise_for_status()
            
            # Parse and process the response
            response_json = response.json()
            return self._process_response(response_json)
            
        except httpx.HTTPError as e:
            raise ValueError(f"HTTP error during RunPod API request: {e}")
        except Exception as e:
            raise ValueError(f"Error calling RunPod API: {e}")
    
    def _stream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[GenerationChunk]:
        """Stream the output of the model."""
        # For simplicity, we'll implement a basic simulated streaming
        # In a real implementation, you'd connect to RunPod's streaming endpoint
        
        full_response = self._call(prompt, stop, run_manager, **kwargs)
        
        # Simulate streaming by yielding one character at a time
        for char in full_response:
            chunk = GenerationChunk(text=char)
            
            if run_manager:
                run_manager.on_llm_new_token(char)
                
            yield chunk 