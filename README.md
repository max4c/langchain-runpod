# langchain-runpod

`langchain-runpod` integrates [RunPod Serverless](https://www.runpod.io/serverless-gpu) endpoints with LangChain.

It allows you to interact with custom large language models (LLMs) and chat models deployed on RunPod's cost-effective and scalable GPU infrastructure directly within your LangChain applications.

This package provides:
- `RunPod`: For interacting with standard text-completion models.
- `ChatRunPod`: For interacting with conversational chat models.

## Installation

```bash
pip install -U langchain-runpod
```

## Authentication

To use this integration, you need a RunPod API key.

1.  Obtain your API key from the [RunPod API Keys page](https://www.runpod.io/console/user/settings).
2.  Set it as an environment variable:

```bash
export RUNPOD_API_KEY="your-runpod-api-key"
```

Alternatively, you can pass the `api_key` directly when initializing the `RunPod` or `ChatRunPod` classes, though using environment variables is recommended for security.

## Basic Usage

You will also need the **Endpoint ID** for your deployed RunPod Serverless endpoint. Find this in the RunPod console under Serverless -> Endpoints.

### LLM (`RunPod`)

Use the `RunPod` class for standard LLM interactions (text completion).

```python
import os
from langchain_runpod import RunPod

# Ensure API key is set (or pass it as api_key="...")
# os.environ["RUNPOD_API_KEY"] = "your-runpod-api-key"

llm = RunPod(
    endpoint_id="your-endpoint-id", # Replace with your actual Endpoint ID
    model_name="runpod-llm", # Optional: For metadata
    temperature=0.7,
    max_tokens=100,
)

# Synchronous call
prompt = "What is the capital of France?"
response = llm.invoke(prompt)
print(f"Sync Response: {response}")

# Async call
# response_async = await llm.ainvoke(prompt)
# print(f"Async Response: {response_async}")

# Streaming (Simulated)
# print("Streaming Response:")
# for chunk in llm.stream(prompt):
#     print(chunk, end="", flush=True)
# print()
```

### Chat Model (`ChatRunPod`)

Use the `ChatRunPod` class for conversational interactions.

```python
import os
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_runpod import ChatRunPod

# Ensure API key is set (or pass it as api_key="...")
# os.environ["RUNPOD_API_KEY"] = "your-runpod-api-key"

chat = ChatRunPod(
    endpoint_id="your-endpoint-id", # Replace with your actual Endpoint ID
    model_name="runpod-chat", # Optional: For metadata
    temperature=0.7,
    max_tokens=256,
)

messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="What are the planets in our solar system?"),
]

# Synchronous call
response = chat.invoke(messages)
print(f"Sync Response:\n{response.content}")

# Async call
# response_async = await chat.ainvoke(messages)
# print(f"Async Response:\n{response_async.content}")

# Streaming (Simulated)
# print("Streaming Response:")
# for chunk in chat.stream(messages):
#     print(chunk.content, end="", flush=True)
# print()
```

## Features and Limitations

### API Interaction
- **Asynchronous Execution**: RunPod Serverless endpoints are inherently asynchronous. This integration handles the underlying polling mechanism for the `/run` and `/status/{job_id}` endpoints automatically for both `RunPod` and `ChatRunPod` classes.
- **Synchronous Endpoint**: While RunPod offers a `/runsync` endpoint, this integration primarily uses the asynchronous `/run` -> `/status` flow for better compatibility and handling of potentially long-running jobs. Polling parameters (`poll_interval`, `max_polling_attempts`) can be configured during initialization.

### Feature Support

The level of support for advanced LLM features depends heavily on the **specific model and handler** deployed on your RunPod endpoint. The RunPod API itself provides a generic interface.

| Feature               | Support Level                                                                                               | Notes                                                                                                                                                                                                |
|-----------------------|-------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Core Invoke/Gen**   | ✅ Supported                                                                                                 | Basic text generation and chat conversations work as expected (sync & async).                                                                                                                        |
| **Streaming**         | ⚠️ Simulated                                                                                                | The `.stream()` and `.astream()` methods work by getting the full response first and then yielding it chunk by chunk. True token-level streaming requires a WebSocket-enabled RunPod endpoint handler. |
| **Tool Calling**      | ↔️ Endpoint Dependent                                                                                       | No built-in support via standardized RunPod API parameters. Depends entirely on the endpoint handler interpreting tool descriptions/schemas passed in the `input`. Standard tests skipped.       |
| **Structured Output** | ↔️ Endpoint Dependent                                                                                       | No built-in support via standardized RunPod API parameters. Depends on the endpoint handler's ability to generate structured formats (e.g., JSON) based on input instructions. Standard tests skipped. |
| **JSON Mode**         | ↔️ Endpoint Dependent                                                                                       | No dedicated `response_format` parameter at the RunPod API level. Depends on the endpoint handler. Standard tests skipped.                                                                       |
| **Token Usage**       | ❌ Not Available                                                                                            | The RunPod API does not provide standardized token usage fields. Usage metadata tests are marked `xfail`. Any token info must come from the endpoint handler's custom output.                        |
| **Logprobs**          | ❌ Not Available                                                                                            | The RunPod API does not provide logprobs.                                                                                                                                                            |
| **Image Input**       | ↔️ Endpoint Dependent                                                                                       | Standard tests pass, likely by adapting image URLs/data. Actual support depends on the endpoint handler.                                                                                             |

### Important Notes

1. **Endpoint Handler**: Ensure your RunPod endpoint runs a compatible LLM server (e.g., vLLM, TGI, FastChat, text-generation-webui) that accepts standard inputs (like `prompt` or `messages`) and returns text output in a common format (direct string, or a dictionary containing keys like `text`, `content`, `output`, `choices`, etc.). The integration attempts to parse common formats, but custom handlers might require modifications to the parsing logic (e.g., overriding `_process_response`).

## Setting Up a RunPod Endpoint

1. Go to [RunPod Serverless](https://www.runpod.io/console/serverless) in your RunPod console.
2. Click "New Endpoint".
3. Select a GPU and a suitable template (e.g., a template running vLLM, TGI, FastChat, or text-generation-webui with your desired model).
4. Configure settings (like FlashInfer, custom container image if needed) and deploy.
5. Once active, copy the **Endpoint ID** for use with this library.

For more details, refer to the [RunPod Serverless Documentation](https://docs.runpod.io/serverless/overview).
