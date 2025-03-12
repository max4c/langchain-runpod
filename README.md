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
from langchain_runpod import ChatRunPod

# Create the chat model instance
# Replace "endpoint-id" with your RunPod endpoint ID
chat = ChatRunPod(
    endpoint_id="endpoint-id",  # Your RunPod serverless endpoint ID
    model_name="llama3-70b-chat",  # Optional - for identification purposes
    temperature=0.7,
    max_tokens=1024,
    # api_key="your-runpod-api-key",  # Optional if set as environment variable
)

# Standard invoke method
response = chat.invoke("Tell me fun things to do in NYC")
print(response.content)

# With streaming
for chunk in chat.stream("Tell me fun things to do in NYC"):
    print(chunk.content, end="", flush=True)

# Using multiple messages for chat
from langchain_core.messages import HumanMessage, SystemMessage

messages = [
    SystemMessage(content="You are a helpful travel assistant."),
    HumanMessage(content="What are the must-see attractions in Paris?")
]

response = chat.invoke(messages)
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

## LLMs

`RunPodLLM` class exposes LLMs from RunPod.

```python
from langchain_runpod import RunPodLLM

llm = RunPodLLM(endpoint_id="endpoint-id")
llm.invoke("The meaning of life is")
```

## Embeddings

`RunPodEmbeddings` class exposes embeddings from RunPod.

```python
from langchain_runpod import RunPodEmbeddings

embeddings = RunPodEmbeddings(endpoint_id="endpoint-id")
embeddings.embed_query("What is the meaning of life?")
```
