"""Integration tests for the RunPod integration."""
import os
import pytest
from langchain_runpod import ChatRunPod

# Skip all tests if the API key and endpoint ID are not set
skip_integration_tests = pytest.mark.skipif(
    not os.environ.get("RUNPOD_API_KEY") or not os.environ.get("RUNPOD_ENDPOINT_ID"),
    reason="RUNPOD_API_KEY and RUNPOD_ENDPOINT_ID not set in environment variables"
)

@skip_integration_tests
def test_simple_query():
    """Test a simple query with the RunPod endpoint."""
    # Create a ChatRunPod instance
    chat = ChatRunPod(
        endpoint_id=os.environ.get("RUNPOD_ENDPOINT_ID"),
        model_name="runpod-test",
        temperature=0.7,
        max_tokens=256,
        poll_interval=1.0,
        max_polling_attempts=120,
        disable_streaming=True,
    )
    
    # Send a simple query
    response = chat.invoke("Hello World")
    
    # Basic assertions - we can't check exact content
    # since it depends on the model, but we can check that
    # the response has content and the expected structure
    assert response.content is not None
    assert isinstance(response.content, str)
    assert len(response.content) > 0
    
    # Check if response metadata exists
    assert response.response_metadata is not None
    assert "raw_response" in response.response_metadata

@skip_integration_tests
def test_complex_messages():
    """Test with more complex messages."""
    from langchain_core.messages import SystemMessage, HumanMessage
    
    # Create a ChatRunPod instance
    chat = ChatRunPod(
        endpoint_id=os.environ.get("RUNPOD_ENDPOINT_ID"),
        model_name="runpod-test",
        temperature=0.7,
        max_tokens=256,
        poll_interval=1.0,
        max_polling_attempts=120,
        disable_streaming=True,
    )
    
    # Create a message list with system and human messages
    messages = [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content="Tell me about artificial intelligence.")
    ]
    
    # Send the messages
    response = chat.invoke(messages)
    
    # Basic assertions
    assert response.content is not None
    assert isinstance(response.content, str)
    assert len(response.content) > 0

@skip_integration_tests
def test_streaming():
    """Test streaming functionality (with fallback to simulated streaming)."""
    # Create a ChatRunPod instance
    chat = ChatRunPod(
        endpoint_id=os.environ.get("RUNPOD_ENDPOINT_ID"),
        model_name="runpod-test",
        temperature=0.7,
        max_tokens=256,
        poll_interval=1.0,
        max_polling_attempts=120,
        disable_streaming=True,  # Using simulated streaming
    )
    
    # Test with streaming
    chunks = []
    for chunk in chat.stream("Write a short poem"):
        assert chunk.content is not None
        chunks.append(chunk.content)
    
    # Check that we got some chunks
    assert len(chunks) > 0
    
    # Join the chunks and check the full response
    full_response = "".join(chunks)
    assert len(full_response) > 0 