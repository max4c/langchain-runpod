"""Test chat model integration."""

from typing import Type
import os
from unittest.mock import patch

import pytest
from langchain_runpod.chat_models import ChatRunPod
from langchain_tests.unit_tests import ChatModelUnitTests


class TestChatRunPodUnit(ChatModelUnitTests):
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
            "endpoint_id": "test-endpoint-id",
            "api_key": "fake-api-key",  # Mock API key for unit tests
        }
        
    @property
    def standard_chat_model_params(self) -> dict:
        """Override standard parameters to include api_key."""
        params = super().standard_chat_model_params
        params["api_key"] = "fake-api-key"  # Mock API key for unit tests
        return params
        
    @pytest.fixture(autouse=True)
    def setup_environment(self):
        """Setup environment variables for tests."""
        # This ensures that even if the test tries to read from environment variables,
        # it will get our test values
        with patch.dict(os.environ, {"RUNPOD_API_KEY": "fake-api-key"}):
            yield
