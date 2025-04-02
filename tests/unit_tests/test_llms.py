"""Custom Unit tests for the RunPod LLM class."""

import os
from unittest.mock import patch, MagicMock

import pytest
import httpx

from langchain_runpod.llms import RunPod, RunPodAPIError


@pytest.fixture
def mock_llm() -> RunPod:
    """Fixture for a RunPod instance with mock credentials."""
    return RunPod(
        endpoint_id="test-endpoint", 
        api_key="test-key", 
        temperature=0.1, 
        max_tokens=10
    )

@pytest.fixture
def mock_env_llm() -> RunPod:
    """Fixture for RunPod instance initialized via environment variable."""
    with patch.dict(os.environ, {"RUNPOD_API_KEY": "env-test-key"}):
        llm = RunPod(endpoint_id="test-env-endpoint")
        # Ensure async client is initialized for later tests
        if llm._async_client is None:
             llm._async_client = httpx.AsyncClient()
        yield llm

# --- Test Initialization --- 

def test_initialization_with_api_key(mock_llm: RunPod):
    assert mock_llm.api_key == "test-key"
    assert mock_llm.endpoint_id == "test-endpoint"
    assert mock_llm._client is not None
    assert mock_llm._async_client is not None

def test_initialization_with_env_var(mock_env_llm: RunPod):
    assert mock_env_llm.api_key == "env-test-key"
    assert mock_env_llm.endpoint_id == "test-env-endpoint"

def test_initialization_missing_api_key():
    with patch.dict(os.environ, {}, clear=True): # Ensure no env var
        with pytest.raises(ValueError, match="RunPod API key must be provided"):
            RunPod(endpoint_id="test-no-key-endpoint")

# --- Test _call --- 

@patch("httpx.Client.post")
def test_call_success(mock_post: MagicMock, mock_llm: RunPod):
    """Test successful synchronous API call."""
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "id": "test-job-id",
        "status": "COMPLETED",
        "output": "Mock LLM response."
    }
    mock_post.return_value = mock_response

    prompt = "Test prompt"
    response = mock_llm._call(prompt)

    assert response == "Mock LLM response."
    mock_post.assert_called_once()
    call_args, call_kwargs = mock_post.call_args
    assert call_args[0] == f"{mock_llm.api_base}/{mock_llm.endpoint_id}/run"
    assert call_kwargs["headers"]["Authorization"] == f"Bearer {mock_llm.api_key}"
    assert call_kwargs["json"]["input"]["prompt"] == prompt
    assert call_kwargs["json"]["input"]["temperature"] == 0.1
    assert call_kwargs["json"]["input"]["max_tokens"] == 10

@patch("httpx.Client.post")
def test_call_api_error(mock_post: MagicMock, mock_llm: RunPod):
    """Test handling of API error during synchronous call."""
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 500
    mock_response.text = "Internal Server Error"
    mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
        "Server Error", request=MagicMock(), response=mock_response
    )
    mock_post.return_value = mock_response

    with pytest.raises(RunPodAPIError, match="RunPod API request failed with status 500"):
        mock_llm._call("Test prompt")

@patch("httpx.Client.post")
def test_call_job_failed(mock_post: MagicMock, mock_llm: RunPod):
    """Test handling of failed job status during synchronous call."""
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "id": "test-job-id",
        "status": "FAILED",
        "error": "Something went wrong"
    }
    mock_post.return_value = mock_response
    
    # Expect RunPodAPIError because _call wraps the ValueError from _process_response
    with pytest.raises(RunPodAPIError, match="Unexpected error processing RunPod response: RunPod job ended with status FAILED"):
        mock_llm._call("Test prompt")

# --- Test _acall --- 

@pytest.mark.asyncio
@patch("httpx.AsyncClient.post")
async def test_acall_success(mock_post: MagicMock, mock_llm: RunPod):
    """Test successful asynchronous API call."""
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "id": "test-async-job-id",
        "status": "COMPLETED",
        "output": "Mock async LLM response."
    }
    # For async mock, return value needs to be awaitable
    async def mock_post_async(*args, **kwargs):
        return mock_response
    mock_post.side_effect = mock_post_async

    prompt = "Test async prompt"
    response = await mock_llm._acall(prompt)

    assert response == "Mock async LLM response."
    mock_post.assert_called_once()
    call_args, call_kwargs = mock_post.call_args
    assert call_args[0] == f"{mock_llm.api_base}/{mock_llm.endpoint_id}/run"
    assert call_kwargs["headers"]["Authorization"] == f"Bearer {mock_llm.api_key}"
    assert call_kwargs["json"]["input"]["prompt"] == prompt

@pytest.mark.asyncio
@patch("httpx.AsyncClient.post")
async def test_acall_api_error(mock_post: MagicMock, mock_llm: RunPod):
    """Test handling of API error during asynchronous call."""
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 401
    mock_response.text = "Unauthorized"
    mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
        "Unauthorized", request=MagicMock(), response=mock_response
    )
    async def mock_post_async(*args, **kwargs):
        return mock_response
    mock_post.side_effect = mock_post_async
    
    with pytest.raises(RunPodAPIError, match="RunPod API async request failed with status 401"):
        await mock_llm._acall("Test async prompt")

# --- Test _process_response --- 
# (Tested indirectly via _call and _acall, but can add direct tests)

def test_process_response_various_outputs(mock_llm: RunPod):
    """Test _process_response with different valid output structures."""
    assert mock_llm._process_response({"status": "COMPLETED", "output": "Simple string"}) == "Simple string"
    assert mock_llm._process_response({"status": "COMPLETED", "output": {"text": "Dict text"}}) == "Dict text"
    assert mock_llm._process_response({"status": "COMPLETED", "output": {"content": "Dict content"}}) == "Dict content"
    assert mock_llm._process_response({"status": "COMPLETED", "output": {"choices": [{"text": "Choice text"}]}}) == "Choice text"
    assert mock_llm._process_response({"status": "COMPLETED", "output": {"choices": [{"message": {"content": "Msg content"}}]}}) == "Msg content"
    assert mock_llm._process_response({"status": "COMPLETED", "output": {"outputs": [{"text": "Mistral text"}]}}) == "Mistral text"
    assert mock_llm._process_response({"status": "COMPLETED", "output": [{"choices": [{"tokens": ["List ", "tokens"]}]}]}) == "List tokens"
    assert mock_llm._process_response({"status": "COMPLETED", "output": ["List ", "of ", "strings"]}) == "List of strings"

def test_process_response_job_status(mock_llm: RunPod):
    """Test _process_response handling non-COMPLETED statuses."""
    with pytest.raises(ValueError, match="RunPod job ended with status IN_PROGRESS"):
        mock_llm._process_response({"status": "IN_PROGRESS"})
    with pytest.raises(ValueError, match="RunPod job ended with status FAILED. Error: Pod terminated"):
        mock_llm._process_response({"status": "FAILED", "error": "Pod terminated"})

def test_process_response_missing_output(mock_llm: RunPod):
    """Test _process_response handling missing output field."""
    response_dict = {"status": "COMPLETED", "id": "123"}
    # Should return string representation of the dict as fallback
    assert mock_llm._process_response(response_dict) == str(response_dict)

# --- Test Streaming Simulation --- 
# Note: These tests verify the simulation logic, not actual streaming protocols

@patch("httpx.Client.post")
def test_stream_simulation(mock_post: MagicMock, mock_llm: RunPod):
    """Test synchronous streaming simulation."""
    mock_response_text = "Stream response."
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 200
    mock_response.json.return_value = {"status": "COMPLETED", "output": mock_response_text}
    mock_post.return_value = mock_response

    chunks = list(mock_llm._stream("Test stream prompt"))
    
    assert len(chunks) == len(mock_response_text)
    assert "".join(c.text for c in chunks) == mock_response_text
    mock_post.assert_called_once() # _stream calls _call internally

@pytest.mark.asyncio
@patch("httpx.AsyncClient.post")
async def test_astream_simulation(mock_post: MagicMock, mock_llm: RunPod):
    """Test asynchronous streaming simulation."""
    mock_response_text = "Async stream response."
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 200
    mock_response.json.return_value = {"status": "COMPLETED", "output": mock_response_text}
    async def mock_post_async(*args, **kwargs):
        return mock_response
    mock_post.side_effect = mock_post_async

    stream_results = []
    async for chunk in mock_llm._astream("Test async stream prompt"):
        stream_results.append(chunk)
        
    assert len(stream_results) == len(mock_response_text)
    assert "".join(c.text for c in stream_results) == mock_response_text
    mock_post.assert_called_once() # _astream calls _acall internally 