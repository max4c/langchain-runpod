# langchain-runpod

This package contains the LangChain integration with [RunPod](https://www.runpod.io).

## Installation

```bash
pip install -U langchain-runpod
```

## Authentication

Configure credentials by setting the following environment variable:

```bash
export RUNPOD_API_KEY="your-runpod-api-key"

```

You can obtain your RunPod API key from the [RunPod API Keys page](https://www.runpod.io/console/user/settings) in your account settings.

## Chat Models

`ChatRunPod` class allows you to interact with any text-based LLM running on RunPod's serverless endpoints.

```python
import os

# Set your RunPod API key
os.environ["RUNPOD_API_KEY"] = "your-runpod-api-key"  # Replace with your actual API key

from langchain_runpod import ChatRunPod

# Initialize the ChatRunPod model
# Replace "endpoint-id" with your actual RunPod endpoint ID
chat = ChatRunPod(
    endpoint_id="endpoint-id",  # Your RunPod serverless endpoint ID
    model_name="llama3-70b-chat",  # Optional - helps with identification
    temperature=0.7,  # Control randomness (0.0 to 1.0)
    max_tokens=1024,  # Maximum tokens in the response
)

# Basic invocation
response = chat.invoke("Explain how transformer models work in 3 sentences.")
print(response.content)
```

### Important Notes

1. **Endpoint Configuration**: The RunPod endpoint must be running an LLM server that accepts requests in a standard format. Common frameworks like [FastChat](https://github.com/lm-sys/FastChat), [text-generation-webui](https://github.com/oobabooga/text-generation-webui), and [vLLM](https://github.com/vllm-project/vllm) all work.

2. **Response Format**: The integration attempts to handle various response formats from different LLM serving frameworks. If you encounter issues with the response parsing, you may need to customize the `_process_response` method.

3. **Multi-Modal Content**: Currently, multi-modal inputs (images, audio, etc.) are converted to text-only format, as most RunPod endpoints don't support multi-modal inputs.

## Setting Up a RunPod Endpoint

1. Go to [RunPod Serverless](https://www.runpod.io/console/serverless) in your RunPod console
2. Click "New Endpoint"
3. Select a GPU and template (e.g., choose a template that runs vLLM, FastChat, or text-generation-webui)
4. Configure settings and deploy
5. Note the endpoint ID for use with this library
