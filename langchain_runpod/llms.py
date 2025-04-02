"""Wrapper around RunPod's LLM Inference API."""

import json
import logging
import os
import time
import asyncio
from typing import Any, Dict, List, Optional, Union, Iterator, AsyncIterator

import httpx
from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models.llms import LLM
from langchain_core.outputs import GenerationChunk
from pydantic import Field, PrivateAttr, model_validator

logger = logging.getLogger(__name__)


class RunPodAPIError(Exception):
    """Custom exception for RunPod API errors."""
    pass


class RunPod(LLM):
    """LLM model wrapper for RunPod API.

    Supports both synchronous and asynchronous generation, as well as
    (simulated) streaming.

    To use, you should have the ``langchain-runpod`` package installed, and the
    environment variable ``RUNPOD_API_KEY`` set with your API key, or pass it
    as the ``api_key`` parameter using named arguments.

    Example:
        .. code-block:: python

            from langchain_runpod import RunPod

            # Synchronous call
            llm = RunPod(endpoint_id="your-endpoint-id", api_key="your-key")
            response = llm.invoke("Tell me a joke")
            print(response)

            # Async call
            # response = await llm.ainvoke("Tell me another joke")
            # print(response)

            # Streaming (synchronous)
            # for chunk in llm.stream("Why did the chicken cross the road?"): 
            #     print(chunk, end="", flush=True)
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
    
    poll_interval: float = 1.0
    """How frequently to poll for job status in seconds."""
    
    max_polling_attempts: int = 120
    """Maximum number of polling attempts for async jobs."""
    
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
        # Initialize async client if not already done (e.g., in a subclass)
        if self._async_client is None:
            self._async_client = httpx.AsyncClient(timeout=self.timeout or 60.0)
    
    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "runpod"
    
    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Get the identifying parameters."""
        return {
            "endpoint_id": self.endpoint_id,
            "model_name": self.model_name,
            "api_base": self.api_base,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "stop": self.stop,
            "timeout": self.timeout,
            "streaming": self.streaming,
            "poll_interval": self.poll_interval,
            "max_polling_attempts": self.max_polling_attempts,
        }
    
    def _get_params(self, stop: Optional[List[str]] = None) -> Dict[str, Any]:
        """Get the parameters to pass to the RunPod input object."""
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
        """Process the RunPod API response and extract the generated text.
        Handles different potential response structures and statuses.
        """
        logger.debug(f"Raw RunPod response: {response}")

        status = response.get("status")
        if status != "COMPLETED":
            error_detail = response.get("error", "No error details provided.")
            logger.error(f"RunPod job failed or did not complete. Status: {status}, Error: {error_detail}")
            # Consider raising a more specific exception type
            raise ValueError(f"RunPod job ended with status {status}. Error: {error_detail}")

        output = response.get("output")

        if output is None:
            logger.warning(f"No 'output' field found in RunPod response: {response}")
            # Fallback: attempt to return the whole response if no output
            return str(response)

        # --- Process different output structures --- 

        # 1. Output is a simple string
        if isinstance(output, str):
            return output

        # 2. Output is a dictionary
        if isinstance(output, dict):
            # Common keys containing the main text output
            common_keys = ["text", "content", "message", "generated_text", "response"]
            for key in common_keys:
                if isinstance(output.get(key), str):
                    return output[key]
            
            # Handle nested structure like {'choices': [{'text': '...', ...}]} or similar
            if isinstance(output.get("choices"), list) and output["choices"]:
                first_choice = output["choices"][0]
                if isinstance(first_choice, dict):
                    for key in ["text", "message", "content"]:
                        if isinstance(first_choice.get(key), str):
                             return first_choice[key]
                        # Deeper nesting like message: {content: '...'} 
                        if isinstance(first_choice.get(key), dict) and isinstance(first_choice[key].get("content"), str):
                            return first_choice[key]["content"]
            
            # Handle Mistral specific output format
            if "outputs" in output and isinstance(output["outputs"], list) and output["outputs"]:
                first_output = output["outputs"][0]
                if isinstance(first_output, dict) and isinstance(first_output.get("text"), str):
                     return first_output["text"]
                     
            # Handle specific format from integration tests: {'choices': [{'tokens': ['...', '...']}]}
            if isinstance(output.get("choices"), list) and output["choices"]:
                first_choice = output["choices"][0]
                if isinstance(first_choice, dict) and isinstance(first_choice.get("tokens"), list):
                    return "".join(map(str, first_choice["tokens"]))

            # Fallback for unrecognized dictionary structure
            logger.warning(f"Unrecognized dictionary structure in 'output': {output}")
            return str(output)

        # 3. Output is a list
        if isinstance(output, list):
             # Specific case from integration tests: [{'choices': [{'tokens': [...]}]}]
            if output and isinstance(output[0], dict) and isinstance(output[0].get("choices"), list):
                first_item_choices = output[0]["choices"]
                if first_item_choices and isinstance(first_item_choices[0], dict):
                    tokens = first_item_choices[0].get("tokens")
                    if isinstance(tokens, list):
                        return "".join(map(str, tokens))
                    elif isinstance(tokens, str): # Handle if 'tokens' is just a string
                         return tokens
                         
            # If it's a list of strings, join them
            if all(isinstance(item, str) for item in output):
                return "".join(output)
            
            # Fallback for unrecognized list structure
            logger.warning(f"Unrecognized list structure in 'output': {output}")
            return str(output) # Convert the list to string as a fallback

        # --- Ultimate Fallback --- 
        logger.warning(f"Unrecognized type for 'output' field: {type(output)}. Response: {response}")
        return str(output) # Return string representation if type is unexpected
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Call the RunPod API synchronously and return the generated text.

        Args:
            prompt: The prompt to send to the model.
            stop: Optional list of strings to stop generation when encountered.
            run_manager: Optional callback manager for the run.
            **kwargs: Additional parameters to pass to the RunPod input object.

        Returns:
            The generated text from the model.

        Raises:
            RunPodAPIError: If the API request fails or the job status indicates an error.
        """
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

            # --- Handle Async Job Polling --- 
            job_id = response_json.get("id")
            status = response_json.get("status")

            if job_id and status in ["IN_QUEUE", "IN_PROGRESS"]:
                logger.info(f"RunPod job {job_id} is async, polling for results...")
                if run_manager:
                    # Use on_llm_new_token to provide feedback during polling
                    run_manager.on_llm_new_token(f"\n[RunPod job {job_id} status: {status}]", verbose=True)
                response_json = self._poll_for_job_status(job_id, run_manager)
            # --- End Polling Handling ---

            return self._process_response(response_json)
            
        except httpx.TimeoutException as e:
            raise RunPodAPIError(f"RunPod API request timed out: {e}") from e
        except httpx.HTTPStatusError as e:
            # Log the response body if available for debugging
            error_body = e.response.text
            logger.error(f"RunPod API returned an error: {e.response.status_code} - {error_body}")
            raise RunPodAPIError(
                f"RunPod API request failed with status {e.response.status_code}: {error_body}"
            ) from e
        except httpx.RequestError as e:
            raise RunPodAPIError(f"Error during RunPod API request: {e}") from e
        except json.JSONDecodeError as e:
            # Handle cases where the response is not valid JSON
            logger.error(f"Failed to decode JSON response from RunPod: {response.text if response else 'No response'}")
            raise RunPodAPIError(f"Invalid JSON response from RunPod API: {e}") from e
        except Exception as e:
            # Catch-all for unexpected errors during processing
            logger.exception(f"An unexpected error occurred processing RunPod response: {e}")
            raise RunPodAPIError(f"Unexpected error processing RunPod response: {e}") from e
    
    def _stream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[GenerationChunk]:
        """Stream the output of the model.
        
        Note: This currently simulates streaming by first getting the full response
        synchronously and then yielding chunks.

        Args:
            prompt: The prompt to send to the model.
            stop: Optional list of strings to stop generation when encountered.
            run_manager: Optional callback manager for the run.
            **kwargs: Additional parameters to pass to the RunPod input object.

        Yields:
            GenerationChunk: Chunks of the generated text.

        Raises:
            RunPodAPIError: If the initial API request fails.
        """
        # For simplicity, we'll implement a basic simulated streaming
        # In a real implementation, you'd connect to RunPod's streaming endpoint
        
        full_response = self._call(prompt, stop, run_manager, **kwargs)
        
        # Simulate streaming by yielding one character at a time
        for char in full_response:
            chunk = GenerationChunk(text=char)
            
            if run_manager:
                run_manager.on_llm_new_token(char)
                
            yield chunk 

    async def _acall(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Asynchronously call the RunPod API and return the generated text.

        Args:
            prompt: The prompt to send to the model.
            stop: Optional list of strings to stop generation when encountered.
            run_manager: Optional callback manager for the run.
            **kwargs: Additional parameters to pass to the RunPod input object.

        Returns:
            The generated text from the model.

        Raises:
            RunPodAPIError: If the API request fails or the job status indicates an error.
        """
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
            if self._async_client is None:
                # Should ideally be initialized in __init__
                self._async_client = httpx.AsyncClient(timeout=self.timeout or 60.0)

            response = await self._async_client.post(
                url,
                headers=headers,
                json=payload,
                timeout=self.timeout or 60.0,
            )
            response.raise_for_status()

            # Parse and process the response
            response_json = response.json()

            # --- Handle Async Job Polling --- 
            job_id = response_json.get("id")
            status = response_json.get("status")

            if job_id and status in ["IN_QUEUE", "IN_PROGRESS"]:
                logger.info(f"RunPod job {job_id} is async, polling for results...")
                if run_manager:
                    await run_manager.on_llm_new_token(f"\n[RunPod job {job_id} status: {status}]", verbose=True)
                response_json = await self._apoll_for_job_status(job_id, run_manager)
            # --- End Polling Handling ---

            return self._process_response(response_json)

        except httpx.TimeoutException as e:
            raise RunPodAPIError(f"RunPod API async request timed out: {e}") from e
        except httpx.HTTPStatusError as e:
            error_body = e.response.text
            logger.error(f"RunPod API (async) returned an error: {e.response.status_code} - {error_body}")
            raise RunPodAPIError(
                f"RunPod API async request failed with status {e.response.status_code}: {error_body}"
            ) from e
        except httpx.RequestError as e:
             raise RunPodAPIError(f"Error during RunPod API async request: {e}") from e
        except json.JSONDecodeError as e:
             logger.error(f"Failed to decode JSON async response from RunPod: {response.text if response else 'No response'}")
             raise RunPodAPIError(f"Invalid JSON response from RunPod API (async): {e}") from e
        except Exception as e:
             logger.exception(f"An unexpected error occurred processing async RunPod response: {e}")
             raise RunPodAPIError(f"Unexpected error processing async RunPod response: {e}") from e

    async def _astream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[GenerationChunk]:
        """Stream the output of the model asynchronously.

        Note: This currently simulates streaming by first getting the full response
        asynchronously and then yielding chunks.

        Args:
            prompt: The prompt to send to the model.
            stop: Optional list of strings to stop generation when encountered.
            run_manager: Optional callback manager for the run.
            **kwargs: Additional parameters to pass to the RunPod input object.

        Yields:
            GenerationChunk: Chunks of the generated text.
        
        Raises:
            RunPodAPIError: If the initial API request fails.
        """
        # Simulate streaming by getting the full response first
        full_response = await self._acall(prompt, stop, run_manager, **kwargs)

        # Yield the response chunk by chunk (character by character)
        for char in full_response:
            chunk = GenerationChunk(text=char)
            if run_manager:
                await run_manager.on_llm_new_token(token=chunk.text, chunk=chunk)
            yield chunk 

    def _poll_for_job_status(
        self,
        job_id: str,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
    ) -> Dict[str, Any]:
        """Poll the RunPod /status endpoint until the job is completed or fails."""
        status_url = f"{self.api_base}/{self.endpoint_id}/status/{job_id}"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        
        for attempt in range(self.max_polling_attempts):
            try:
                status_response = self._client.get(
                    status_url,
                    headers=headers,
                    timeout=self.timeout or 60.0, 
                )
                status_response.raise_for_status()
                status_data = status_response.json()
                status = status_data.get("status")

                logger.debug(f"Poll attempt {attempt+1}: Job {job_id} status: {status}")
                if run_manager:
                     # Use on_llm_new_token to provide feedback during polling
                    run_manager.on_llm_new_token(f"\n[RunPod job {job_id} status: {status}]", verbose=True)

                if status == "COMPLETED":
                    return status_data
                elif status == "FAILED":
                     error_detail = status_data.get("error", "Job failed with no error details.")
                     logger.error(f"RunPod job {job_id} failed: {error_detail}")
                     # Return the failed status data for _process_response to handle
                     return status_data 
                elif status in ["IN_QUEUE", "IN_PROGRESS"]:
                    time.sleep(self.poll_interval)
                else:
                    # Unexpected status
                    logger.warning(f"RunPod job {job_id} returned unexpected status: {status}")
                    return status_data # Return what we have

            except httpx.HTTPStatusError as e:
                # Log polling error but continue polling unless it's critical (like 401/404)
                logger.error(f"HTTP error while polling job {job_id} (attempt {attempt+1}): {e}")
                if e.response.status_code in [401, 403, 404]:
                     raise RunPodAPIError(f"Fatal HTTP error {e.response.status_code} while polling job {job_id}") from e
                time.sleep(self.poll_interval)
            except httpx.RequestError as e:
                 logger.error(f"Request error while polling job {job_id} (attempt {attempt+1}): {e}")
                 time.sleep(self.poll_interval)
            except Exception as e:
                 logger.exception(f"Unexpected error while polling job {job_id} (attempt {attempt+1}): {e}")
                 time.sleep(self.poll_interval)
                 
        raise TimeoutError(
            f"RunPod job {job_id} did not complete after {self.max_polling_attempts} attempts."
        ) 

    async def _apoll_for_job_status(
        self,
        job_id: str,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
    ) -> Dict[str, Any]:
        """Poll the RunPod /status endpoint asynchronously until the job is completed or fails."""
        status_url = f"{self.api_base}/{self.endpoint_id}/status/{job_id}"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        
        if self._async_client is None:
             self._async_client = httpx.AsyncClient(timeout=self.timeout or 60.0)

        for attempt in range(self.max_polling_attempts):
            try:
                status_response = await self._async_client.get(
                    status_url,
                    headers=headers,
                    timeout=self.timeout or 60.0,
                )
                status_response.raise_for_status()
                status_data = status_response.json()
                status = status_data.get("status")

                logger.debug(f"Async poll attempt {attempt+1}: Job {job_id} status: {status}")
                if run_manager:
                     await run_manager.on_llm_new_token(f"\n[RunPod job {job_id} status: {status}]", verbose=True)

                if status == "COMPLETED":
                    return status_data
                elif status == "FAILED":
                    error_detail = status_data.get("error", "Job failed with no error details.")
                    logger.error(f"RunPod job {job_id} failed: {error_detail}")
                    return status_data
                elif status in ["IN_QUEUE", "IN_PROGRESS"]:
                    await asyncio.sleep(self.poll_interval)
                else:
                    logger.warning(f"RunPod job {job_id} returned unexpected status: {status}")
                    return status_data

            except httpx.HTTPStatusError as e:
                logger.error(f"HTTP error while polling job {job_id} (async attempt {attempt+1}): {e}")
                if e.response.status_code in [401, 403, 404]:
                    raise RunPodAPIError(f"Fatal HTTP error {e.response.status_code} while polling job {job_id} (async)") from e
                await asyncio.sleep(self.poll_interval)
            except httpx.RequestError as e:
                 logger.error(f"Request error while polling job {job_id} (async attempt {attempt+1}): {e}")
                 await asyncio.sleep(self.poll_interval)
            except Exception as e:
                 logger.exception(f"Unexpected error while polling job {job_id} (async attempt {attempt+1}): {e}")
                 await asyncio.sleep(self.poll_interval)

        raise TimeoutError(
            f"RunPod job {job_id} did not complete after {self.max_polling_attempts} async attempts."
        ) 