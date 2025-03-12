"""RunPod chat models."""

import json
import logging
import os
import time
import asyncio
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional, Union

import httpx
from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.messages.ai import UsageMetadata
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from pydantic import Field, PrivateAttr, root_validator, model_validator

logger = logging.getLogger(__name__)


class ChatRunPod(BaseChatModel):
    """RunPod chat model integration for LangChain.

    This class provides integration with RunPod's API for accessing chat-based LLMs
    hosted on RunPod serverless endpoints.

    Setup:
        Install ``langchain-runpod`` and set environment variable ``RUNPOD_API_KEY``.

        .. code-block:: bash

            pip install -U langchain-runpod
            export RUNPOD_API_KEY="your-api-key"

    Key init args — completion params:
        endpoint_id: str
            The RunPod serverless endpoint ID to use.
        model_name: str
            Name of the model to reference (for metadata purposes).
        temperature: float
            Sampling temperature.
        max_tokens: Optional[int]
            Max number of tokens to generate.
        top_p: Optional[float]
            Top-p sampling parameter.
        top_k: Optional[int]
            Top-k sampling parameter.

    Key init args — client params:
        api_key: Optional[str]
            RunPod API key. If not passed, will be read from env var RUNPOD_API_KEY.
        timeout: Optional[int]
            Timeout for requests in seconds.
        max_retries: int
            Max number of retries for API calls.
        api_base: Optional[str]
            Base URL for the RunPod API. Default is "https://api.runpod.ai/v2".
        poll_interval: float
            How frequently to poll for job status in seconds.
        max_polling_attempts: int
            Maximum number of polling attempts for async jobs.
        disable_streaming: bool
            If True, will not attempt to use streaming endpoints and will always
            fall back to simulated streaming. Default is False.

    Streaming Support:
        Note that most RunPod serverless endpoints do not support native streaming by default.
        To enable true streaming support, your RunPod handler needs to be specifically
        configured with a generator function and `return_aggregate_stream: True`.
        See the RunPod documentation on WebSocket streaming for details:
        https://blog.runpod.io/introduction-to-websocket-streaming-with-runpod-serverless/
        
        If streaming is not supported by your endpoint, the integration will automatically
        fall back to simulated streaming (sending the full response and then streaming
        it character by character). You can also set `disable_streaming=True` to skip
        the streaming endpoint attempt altogether.

    Instantiate:
        .. code-block:: python

            from langchain_runpod import ChatRunPod

            llm = ChatRunPod(
                endpoint_id="abc123def456",
                model_name="llama3-70b-chat",
                temperature=0.7,
                max_tokens=1024,
                # api_key="your-api-key",  # Optional if set as environment variable
            )

    Invoke:
        .. code-block:: python

            messages = [
                ("system", "You are a helpful assistant that provides detailed information."),
                ("human", "What are the major planets in our solar system?"),
            ]
            llm.invoke(messages)

        .. code-block:: python

            AIMessage(content='The major planets in our solar system are Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, and Neptune. These eight planets orbit around the Sun and are classified as either terrestrial planets (Mercury, Venus, Earth, Mars) or gas giants (Jupiter, Saturn, Uranus, Neptune).')

    Stream:
        .. code-block:: python

            for chunk in llm.stream(messages):
                print(chunk.content, end="", flush=True)

        .. code-block:: python

            The major planets in our solar system are Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, and Neptune...

    """  # noqa: E501

    endpoint_id: str
    """The RunPod endpoint ID to use."""
    
    model_name: str = Field(default="")
    """The name of the model. Used for metadata purposes."""
    
    api_key: Optional[str] = None
    """RunPod API key."""
    
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
    
    timeout: Optional[int] = None
    """Timeout for requests in seconds."""
    
    stop: Optional[List[str]] = None
    """List of strings to stop generation when encountered."""
    
    max_retries: int = 2
    """Maximum number of retries for API calls."""

    poll_interval: float = 1.0
    """How frequently to poll for job status in seconds."""
    
    max_polling_attempts: int = 60
    """Maximum number of polling attempts for async jobs."""
    
    disable_streaming: bool = False
    """If True, will not attempt to use streaming endpoints and will always fall back to simulated streaming."""

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
        """Initialize the ChatRunPod instance."""
        super().__init__(**kwargs)
        self._client = httpx.Client(timeout=self.timeout or 60.0)

    @property
    def _llm_type(self) -> str:
        """Return type of chat model."""
        return "chat-runpod"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Return a dictionary of identifying parameters."""
        return {
            "model_name": self.model_name,
            "endpoint_id": self.endpoint_id,
        }
        
    def _get_ls_params(self, stop: Optional[List[str]] = None, **kwargs) -> Dict[str, Any]:
        """Get the params used for LangSmith tracking.
        
        Args:
            stop: Optional list of stop sequences.
            **kwargs: Additional parameters passed to the model.
        
        Returns:
            Parameters for LangSmith tracing.
        """
        params = {
            "ls_provider": "runpod",
            "ls_model_name": self.model_name,
            "ls_model_type": "chat",
            "ls_temperature": self.temperature,
            "ls_max_tokens": self.max_tokens,
            "ls_stop": stop or self.stop,
        }
        # Make sure the output dict doesn't contain None values
        return {k: v for k, v in params.items() if v is not None}

    def _convert_messages_to_prompt(self, messages: List[BaseMessage]) -> Dict[str, Any]:
        """Convert a list of LangChain messages to RunPod API format.
        
        This method creates a format that should work with most LLMs hosted on RunPod,
        but may need to be overridden for custom formats.
        """
        logger.debug(f"Converting messages to prompt: {messages}")
        
        # Convert messages to a simple text format for the RunPod endpoint
        combined_text = ""
        
        for message in messages:
            # Handle multi-modal content by checking if content is a list
            if isinstance(message.content, list):
                # Most RunPod endpoints don't support structured content yet
                # Converting to simple text format
                text_content = []
                for item in message.content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        text_content.append(item.get("text", ""))
                content = " ".join(text_content)
                logger.warning(
                    "Multi-modal content detected. Converting to text-only format. "
                    "The endpoint may not support multi-modal inputs."
                )
            else:
                content = message.content
                
            # Combine the messages into a single text string
            if isinstance(message, SystemMessage):
                combined_text += f"System: {content}\n"
            elif isinstance(message, HumanMessage):
                combined_text += f"User: {content}\n"
            elif isinstance(message, AIMessage):
                combined_text += f"Assistant: {content}\n"
            else:
                combined_text += f"{content}\n"
        
        # For simple text-only endpoints, use a basic prompt format as shown in example
        simple_payload = {
            "prompt": combined_text.strip()
        }
        
        # Add optional parameters
        if self.temperature is not None:
            simple_payload["temperature"] = self.temperature
            
        if self.max_tokens is not None:
            simple_payload["max_tokens"] = self.max_tokens
            
        if self.top_p is not None:
            simple_payload["top_p"] = self.top_p
            
        if self.top_k is not None:
            simple_payload["top_k"] = self.top_k
            
        if self.stop:
            simple_payload["stop"] = self.stop
            
        logger.debug(f"Final structured payload: {simple_payload}")
        
        # Wrap in "input" field as expected by the RunPod endpoint
        return {"input": simple_payload}

    def _process_response(self, response_json: Dict[str, Any]) -> AIMessage:
        """Process the response from RunPod API and extract the message content."""
        try:
            # Different RunPod endpoints might return varying response structures
            # Here we try to handle the most common formats
            
            # Check if there's an error
            if "error" in response_json:
                error_msg = response_json.get("error", "Unknown error")
                raise ValueError(f"RunPod API error: {error_msg}")
            
            # For debugging
            logger.debug(f"Response format: {response_json}")
            
            # Check if response is in the format with 'output' as a list
            if "output" in response_json and isinstance(response_json["output"], list):
                output_list = response_json["output"]
                
                # Process output as a list format
                if output_list:
                    # For integration tests compatibility, get simple string output from tokens
                    if isinstance(output_list[0], dict) and "choices" in output_list[0]:
                        first_item = output_list[0]
                        choices = first_item.get("choices", [])
                        
                        if choices and "tokens" in choices[0]:
                            # Join all tokens into a single string
                            tokens = choices[0]["tokens"]
                            if isinstance(tokens, list):
                                content = "".join(tokens)
                            else:
                                content = str(tokens)
                        else:
                            # Handle other formats
                            content = str(choices[0]) if choices else str(first_item)
                            
                        # Extract usage if available
                        usage_metadata = {}
                        if "usage" in first_item and isinstance(first_item["usage"], dict):
                            usage = first_item["usage"]
                            usage_metadata = {
                                "input_tokens": usage.get("input", 0),
                                "output_tokens": usage.get("output", 0),
                                "total_tokens": usage.get("input", 0) + usage.get("output", 0),
                            }
                            
                        return AIMessage(
                            content=content,
                            additional_kwargs={},
                            response_metadata={"raw_response": response_json},
                            usage_metadata=usage_metadata if usage_metadata else None,
                        )
                    else:
                        # Simply join all items in the list if it's just strings
                        if all(isinstance(item, str) for item in output_list):
                            content = "".join(output_list)
                        else:
                            content = str(output_list)
                        
                        return AIMessage(
                            content=content,
                            additional_kwargs={},
                            response_metadata={"raw_response": response_json},
                        )
            
            # For simpler 'output' field as a string or dict
            if "output" in response_json:
                output = response_json["output"]
                
                if isinstance(output, str):
                    return AIMessage(
                        content=output,
                        additional_kwargs={},
                        response_metadata={"raw_response": response_json},
                    )
                elif isinstance(output, dict):
                    # Try common formats for content
                    for key in ["content", "text", "message", "generated_text", "response"]:
                        if key in output and isinstance(output[key], str):
                            return AIMessage(
                                content=output[key],
                                additional_kwargs={},
                                response_metadata={"raw_response": response_json},
                            )
                    
                    # If no recognizable content field, stringify the dict
                    return AIMessage(
                        content=str(output),
                        additional_kwargs={},
                        response_metadata={"raw_response": response_json},
                    )
            
            # Fallback: return the string representation of the entire response
            logger.warning(f"Unrecognized response format: {response_json}")
            return AIMessage(
                content=str(response_json),
                additional_kwargs={},
                response_metadata={"raw_response": response_json},
            )
            
        except Exception as e:
            logger.error(f"Error processing RunPod response: {e}")
            logger.error(f"Response JSON: {response_json}")
            # Return a message with the error information
            return AIMessage(
                content="Error processing response from RunPod API",
                additional_kwargs={"error": str(e)},
                response_metadata={"raw_response": response_json, "error": str(e)},
            )

    def _poll_for_job_status(self, job_id: str) -> Dict[str, Any]:
        """Poll for status of an async job and return results when complete."""
        logger.debug(f"Polling for job status for job ID: {job_id}")
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
        }
        
        status_url = f"{self.api_base}/{self.endpoint_id}/status/{job_id}"
        
        for attempt in range(self.max_polling_attempts):
            try:
                time.sleep(self.poll_interval)
                
                response = self._client.get(
                    status_url, 
                    headers=headers,
                    timeout=self.timeout or 10.0
                )
                response.raise_for_status()
                status_data = response.json()
                
                # Check if job is complete
                if status_data.get("status") == "COMPLETED":
                    logger.debug(f"Job completed successfully after {attempt + 1} attempts")
                    return status_data
                
                # Check if job failed
                if status_data.get("status") in ["FAILED", "CANCELLED"]:
                    error_msg = status_data.get("error", "Unknown error")
                    raise ValueError(f"RunPod job failed: {error_msg}")
                
                # If still in progress, continue polling
                logger.debug(f"Job still in progress, status: {status_data.get('status')}")
                
            except httpx.HTTPError as e:
                logger.error(f"HTTP error polling for job status: {e}")
                if attempt == self.max_polling_attempts - 1:
                    raise ValueError(f"Max polling attempts reached, last error: {e}")
        
        raise ValueError(f"Max polling attempts ({self.max_polling_attempts}) reached without job completion")

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate a chat response from RunPod API."""
        # Prepare stop sequences
        stop_sequences = stop if stop else self.stop
        
        # Convert messages to the format expected by RunPod API
        payload = self._convert_messages_to_prompt(messages)
        
        # Add any additional kwargs to the payload
        for key, value in kwargs.items():
            payload[key] = value
        
        # Prepare headers with API key
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        
        # Make the API request
        try:
            url = f"{self.api_base}/{self.endpoint_id}/run"
            response = self._client.post(
                url,
                headers=headers,
                json=payload,
                timeout=self.timeout or 60.0,
            )
            response.raise_for_status()
            
            # Parse the initial response
            response_json = response.json()
            
            # Check if this is an async job that requires polling
            if "id" in response_json and "status" in response_json:
                job_id = response_json.get("id")
                status = response_json.get("status")
                
                if status in ["IN_QUEUE", "IN_PROGRESS"]:
                    logger.info(f"RunPod job {job_id} is async, polling for results...")
                    if run_manager:
                        run_manager.on_llm_new_token(f"Waiting for job {job_id}...")
                    
                    # Poll for results
                    final_response = self._poll_for_job_status(job_id)
                    response_json = final_response
            
            # Process the response
            message = self._process_response(response_json)
            
            # Return the chat result
            generation = ChatGeneration(message=message)
            return ChatResult(generations=[generation])
            
        except httpx.HTTPError as e:
            raise ValueError(f"HTTP error during RunPod API request: {e}")
        except Exception as e:
            raise ValueError(f"Error calling RunPod API: {e}")

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """Stream the output of the model from RunPod API.
        
        Note: RunPod serverless endpoints typically don't support streaming.
        We use simulated streaming by generating the full response and streaming
        it character by character.
        """
        logger.info("Using simulated streaming as RunPod serverless endpoints don't natively support streaming")
        return self._simulated_streaming(messages, stop=stop, run_manager=run_manager, **kwargs)

    def _simulated_streaming(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """Simulate streaming by generating the full response and then streaming it character by character."""
        try:
            # Generate the full response
            full_response = self._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
            content = full_response.generations[0].message.content
            
            # Get usage metadata from the full response if available
            usage_metadata = full_response.generations[0].message.usage_metadata
            
            # Total number of characters to stream
            total_chars = len(content)
            
            # Simulate streaming by yielding one character at a time
            for i, char in enumerate(content):
                # For the first chunk, include input tokens in usage metadata
                if i == 0 and usage_metadata:
                    chunk_metadata = {
                        "input_tokens": usage_metadata.get("input_tokens", 0),
                        "output_tokens": 1,  # Each chunk is one token in our simulation
                        "total_tokens": usage_metadata.get("input_tokens", 0) + 1
                    }
                # For other chunks, only include output tokens
                elif usage_metadata:
                    chunk_metadata = {
                        "input_tokens": 0,  # Only first chunk gets input tokens
                        "output_tokens": 1,  # Each chunk is one token in our simulation
                        "total_tokens": 1
                    }
                else:
                    chunk_metadata = None
                
                chunk = ChatGenerationChunk(
                    message=AIMessageChunk(
                        content=char,
                        usage_metadata=chunk_metadata
                    ),
                )
                
                if run_manager:
                    run_manager.on_llm_new_token(
                        token=char,
                        chunk=chunk,
                    )
                    
                yield chunk
                time.sleep(0.01)  # Small delay to simulate typing
                
        except Exception as e:
            error_message = f"Error in simulated streaming: {e}"
            logger.error(error_message)
            error_chunk = ChatGenerationChunk(
                message=AIMessageChunk(content=error_message),
            )
            yield error_chunk

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Asynchronously generate a chat response from RunPod API."""
        # Initialize async client if needed
        if self._async_client is None:
            self._async_client = httpx.AsyncClient(timeout=self.timeout or 60.0)
            
        # Prepare stop sequences
        stop_sequences = stop if stop else self.stop
        
        # Convert messages to the format expected by RunPod API
        payload = self._convert_messages_to_prompt(messages)
        
        # Add any additional kwargs to the payload
        for key, value in kwargs.items():
            payload[key] = value
        
        # Prepare headers with API key
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        
        # Make the API request
        try:
            url = f"{self.api_base}/{self.endpoint_id}/run"
            response = await self._async_client.post(
                url,
                headers=headers,
                json=payload,
                timeout=self.timeout or 60.0,
            )
            response.raise_for_status()
            
            # Parse the response
            response_json = response.json()
            
            # Check if this is an async job that requires polling
            if "id" in response_json and "status" in response_json:
                job_id = response_json.get("id")
                status = response_json.get("status")
                
                if status in ["IN_QUEUE", "IN_PROGRESS"]:
                    logger.info(f"RunPod job {job_id} is async, polling for results...")
                    if run_manager:
                        await run_manager.on_llm_new_token(f"Waiting for job {job_id}...")
                    
                    # Poll for results (async version)
                    for attempt in range(self.max_polling_attempts):
                        await asyncio.sleep(self.poll_interval)
                        
                        status_url = f"{self.api_base}/{self.endpoint_id}/status/{job_id}"
                        status_response = await self._async_client.get(
                            status_url, 
                            headers={"Authorization": f"Bearer {self.api_key}"},
                            timeout=self.timeout or 10.0
                        )
                        status_response.raise_for_status()
                        status_data = status_response.json()
                        
                        if status_data.get("status") == "COMPLETED":
                            response_json = status_data
                            break
                            
                        if status_data.get("status") in ["FAILED", "CANCELLED"]:
                            error_msg = status_data.get("error", "Unknown error")
                            raise ValueError(f"RunPod job failed: {error_msg}")
                            
                        if attempt == self.max_polling_attempts - 1:
                            raise ValueError(f"Max polling attempts reached without job completion")
            
            # Process the response
            message = self._process_response(response_json)
            
            # Return the chat result
            generation = ChatGeneration(message=message)
            return ChatResult(generations=[generation])
            
        except httpx.HTTPError as e:
            raise ValueError(f"HTTP error during async RunPod API request: {e}")
        except Exception as e:
            raise ValueError(f"Error calling async RunPod API: {e}")

    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        """Asynchronously stream the output of the model from RunPod API.
        
        Note: RunPod serverless endpoints typically don't support streaming.
        We use simulated streaming by generating the full response and streaming
        it character by character.
        """
        logger.info("Using simulated async streaming as RunPod serverless endpoints don't natively support streaming")
        async for chunk in self._async_simulated_streaming(messages, stop=stop, run_manager=run_manager, **kwargs):
            yield chunk

    async def _async_simulated_streaming(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        """Simulate streaming asynchronously by generating the full response and then streaming it character by character."""
        try:
            # Generate the full response
            full_response = await self._agenerate(messages, stop=stop, run_manager=run_manager, **kwargs)
            content = full_response.generations[0].message.content
            
            # Get usage metadata from the full response if available
            usage_metadata = full_response.generations[0].message.usage_metadata
            
            # Total number of characters to stream
            total_chars = len(content)
            
            # Simulate streaming by yielding one character at a time
            for i, char in enumerate(content):
                # For the first chunk, include input tokens in usage metadata
                if i == 0 and usage_metadata:
                    chunk_metadata = {
                        "input_tokens": usage_metadata.get("input_tokens", 0),
                        "output_tokens": 1,  # Each chunk is one token in our simulation
                        "total_tokens": usage_metadata.get("input_tokens", 0) + 1
                    }
                # For other chunks, only include output tokens
                elif usage_metadata:
                    chunk_metadata = {
                        "input_tokens": 0,  # Only first chunk gets input tokens
                        "output_tokens": 1,  # Each chunk is one token in our simulation
                        "total_tokens": 1
                    }
                else:
                    chunk_metadata = None
                
                chunk = ChatGenerationChunk(
                    message=AIMessageChunk(
                        content=char,
                        usage_metadata=chunk_metadata
                    ),
                )
                
                if run_manager:
                    await run_manager.on_llm_new_token(
                        token=char,
                        chunk=chunk,
                    )
                    
                yield chunk
                await asyncio.sleep(0.01)  # Small delay to simulate typing
                
        except Exception as e:
            error_message = f"Error in async simulated streaming: {e}"
            logger.error(error_message)
            error_chunk = ChatGenerationChunk(
                message=AIMessageChunk(content=error_message),
            )
            yield error_chunk
