"""Custom integration tests for RunPod LLM."""

import os
import pytest

from langchain_core.outputs import LLMResult

from langchain_runpod.llms import RunPod

# Get API key and endpoint ID from environment
RUNPOD_API_KEY = os.environ.get("RUNPOD_API_KEY")
RUNPOD_ENDPOINT_ID = os.environ.get("RUNPOD_ENDPOINT_ID")

# Skip tests if API key or endpoint ID is not set
_skip_if_credentials_not_set = pytest.mark.skipif(
    RUNPOD_API_KEY is None or RUNPOD_ENDPOINT_ID is None,
    reason="RUNPOD_API_KEY or RUNPOD_ENDPOINT_ID environment variable not set",
)


@_skip_if_credentials_not_set
@pytest.fixture
def llm() -> RunPod:
    """Fixture for a RunPod LLM instance with real credentials."""
    return RunPod(
        endpoint_id=RUNPOD_ENDPOINT_ID,  # type: ignore
        api_key=RUNPOD_API_KEY,          # type: ignore
        temperature=0.01,                # Low temp for predictable output
        max_tokens=50
    )


@_skip_if_credentials_not_set
def test_llm_invoke(llm: RunPod):
    """Test synchronous invocation."""
    prompt = "Say hello!"
    response = llm.invoke(prompt)
    assert isinstance(response, str)
    assert len(response) > 0
    print(f"Invoke Response: {response}") # Optional: print response for debugging


@_skip_if_credentials_not_set
@pytest.mark.asyncio
async def test_llm_ainvoke(llm: RunPod):
    """Test asynchronous invocation."""
    prompt = "Say goodbye!"
    response = await llm.ainvoke(prompt)
    assert isinstance(response, str)
    assert len(response) > 0
    print(f"AInvoke Response: {response}") # Optional: print response


@_skip_if_credentials_not_set
def test_llm_batch(llm: RunPod):
    """Test synchronous batch invocation."""
    prompts = ["Say hello!", "Say bonjour!"]
    result: LLMResult = llm.generate(prompts)
    assert isinstance(result, LLMResult)
    assert len(result.generations) == len(prompts)
    for generations in result.generations:
        assert len(generations) > 0
        assert isinstance(generations[0].text, str)
        assert len(generations[0].text) > 0
        print(f"Batch Response: {generations[0].text}") # Optional: print


@_skip_if_credentials_not_set
@pytest.mark.asyncio
async def test_llm_agenerate(llm: RunPod):
    """Test asynchronous generation (batch)."""
    prompts = ["Say hello async!", "Say bonjour async!"]
    result: LLMResult = await llm.agenerate(prompts)
    assert isinstance(result, LLMResult)
    assert len(result.generations) == len(prompts)
    for generations in result.generations:
        assert len(generations) > 0
        assert isinstance(generations[0].text, str)
        assert len(generations[0].text) > 0
        print(f"AGenerate Response: {generations[0].text}") # Optional: print


@_skip_if_credentials_not_set
def test_llm_stream(llm: RunPod):
    """Test synchronous streaming."""
    prompt = "Tell me a short story about a robot learning to code."
    full_response = ""
    chunk_count = 0
    for chunk in llm.stream(prompt):
        assert isinstance(chunk, str)
        assert len(chunk) > 0
        full_response += chunk
        chunk_count += 1
    
    assert chunk_count > 1 # Should receive multiple chunks
    assert len(full_response) > 0
    print(f"Stream Response: {full_response}") # Optional: print


@_skip_if_credentials_not_set
@pytest.mark.asyncio
async def test_llm_astream(llm: RunPod):
    """Test asynchronous streaming."""
    prompt = "Tell me a short story about a cat discovering gravity."
    full_response = ""
    chunk_count = 0
    async for chunk in llm.astream(prompt):
        assert isinstance(chunk, str)
        assert len(chunk) > 0
        full_response += chunk
        chunk_count += 1
        
    assert chunk_count > 1
    assert len(full_response) > 0
    print(f"AStream Response: {full_response}") # Optional: print 