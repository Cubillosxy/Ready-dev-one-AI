"""Tests for OpenAIChatClient."""
import pytest
from unittest.mock import MagicMock, patch
from openai import RateLimitError, APIStatusError, APIConnectionError, APITimeoutError
from rdoai.llm.openai_client import OpenAIChatClient, LlmResponse, LlmErrorInfo


@pytest.fixture
def mock_openai(mocker):
    return mocker.patch("rdoai.llm.openai_client.OpenAI")


class TestOpenAIChatClient:
    def test_init(self, mock_openai):
        client = OpenAIChatClient(
            base_url="http://localhost",
            api_key="sk-test",
            model="gpt-model",
            max_retries=3
        )
        
        mock_openai.assert_called_once_with(
            base_url="http://localhost",
            api_key="sk-test",
            max_retries=3
        )
        assert client.model == "gpt-model"

    def test_chat_success(self, mock_openai):
        mock_instance = mock_openai.return_value
        mock_resp = MagicMock()
        mock_resp.choices = [
            MagicMock(message=MagicMock(content="Artificial Intelligence is cool"))
        ]
        mock_instance.chat.completions.create.return_value = mock_resp
        
        client = OpenAIChatClient("b", "k", "m")
        messages = [{"role": "user", "content": "hi"}]
        resp, error = client.chat(messages, temperature=0.5)
        
        assert error is None
        assert isinstance(resp, LlmResponse)
        assert resp.text == "Artificial Intelligence is cool"
        
        mock_instance.chat.completions.create.assert_called_once_with(
            model="m",
            messages=messages,
            temperature=0.5
        )

    def test_chat_rate_limit_error(self, mock_openai):
        mock_instance = mock_openai.return_value
        # Simulate RateLimitError
        mock_resp = MagicMock()
        mock_resp.headers = {"retry-after": "5.5"}
        error = RateLimitError(
            message="Rate limit hit",
            response=mock_resp,
            body=None
        )
        mock_instance.chat.completions.create.side_effect = error
        
        client = OpenAIChatClient("b", "k", "m")
        resp, err_info = client.chat([{"role": "user", "content": "hi"}])
        
        assert resp is None
        assert isinstance(err_info, LlmErrorInfo)
        assert err_info.code == "rate_limit"
        assert err_info.retry_after_sec == 5.5
        assert "Rate limit hit" in err_info.message

    def test_chat_api_status_error_auth(self, mock_openai):
        mock_instance = mock_openai.return_value
        mock_resp = MagicMock()
        mock_resp.status_code = 401
        error = APIStatusError(
            message="Invalid API Key",
            response=mock_resp,
            body=None
        )
        mock_instance.chat.completions.create.side_effect = error
        
        client = OpenAIChatClient("b", "k", "m")
        resp, err_info = client.chat([])
        
        assert err_info.code == "auth_error"
        assert err_info.status_code == 401

    def test_chat_api_status_error_server(self, mock_openai):
        mock_instance = mock_openai.return_value
        mock_resp = MagicMock()
        mock_resp.status_code = 503
        error = APIStatusError(
            message="Service Unavailable",
            response=mock_resp,
            body=None
        )
        mock_instance.chat.completions.create.side_effect = error
        
        client = OpenAIChatClient("b", "k", "m")
        resp, err_info = client.chat([])
        
        assert err_info.code == "server_error"
        assert err_info.status_code == 503

    def test_chat_network_error(self, mock_openai):
        mock_instance = mock_openai.return_value
        mock_instance.chat.completions.create.side_effect = APIConnectionError(
            message="Connection failed",
            request=MagicMock()
        )
        
        client = OpenAIChatClient("b", "k", "m")
        resp, err_info = client.chat([])
        
        assert err_info.code == "network_error"
        assert err_info.status_code is None

    def test_chat_unknown_error(self, mock_openai):
        mock_instance = mock_openai.return_value
        mock_instance.chat.completions.create.side_effect = ValueError("Unexpected")
        
        client = OpenAIChatClient("b", "k", "m")
        resp, err_info = client.chat([])
        
        assert err_info.code == "unknown_error"
        assert "Unexpected" in err_info.message
