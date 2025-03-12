"""Test ChatRunPod chat model."""

from typing import Type
import os

from langchain_runpod.chat_models import ChatRunPod
from langchain_tests.integration_tests import ChatModelIntegrationTests


class TestChatParrotLinkIntegration(ChatModelIntegrationTests):
    @property
    def chat_model_class(self) -> Type[ChatRunPod]:
        return ChatRunPod

    @property
    def chat_model_params(self) -> dict:
        # These should be parameters used to initialize your integration for testing
        return {
            "model": "bird-brain-001",
            "temperature": 0,
            "parrot_buffer_length": 50,
            "endpoint_id": os.environ.get("RUNPOD_ENDPOINT_ID", "test-endpoint-id"),
            "api_key": os.environ.get("RUNPOD_API_KEY", "fake-api-key"),
        }
