"""Tests for LLM Factory."""
from unittest.mock import MagicMock, patch
from rdoai.llm.factory import build_llm
from rdoai.config import AppConfig


class TestLlmFactory:
    @patch("rdoai.llm.factory.OpenAIChatClient")
    def test_build_llm(self, mock_client_cls):
        cfg = MagicMock(spec=AppConfig)
        cfg.llm.base_url = "https://api.openai.com"
        cfg.llm.api_key = "test-key"
        cfg.llm.model = "gpt-4"
        
        client = build_llm(cfg)
        
        mock_client_cls.assert_called_once_with(
            base_url="https://api.openai.com",
            api_key="test-key",
            model="gpt-4",
            max_retries=2
        )
        assert client == mock_client_cls.return_value
