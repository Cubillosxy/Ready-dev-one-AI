"""Tests for AssistantPipeline."""
import pytest
from unittest.mock import MagicMock, patch
from rdoai.pipeline.assistant import AssistantPipeline, AssistantResult
from rdoai.config import AppConfig


class TestAssistantPipeline:
    @patch("rdoai.pipeline.assistant.build_llm")
    @patch("rdoai.pipeline.assistant.NoopContextProvider")
    def test_init(self, mock_ctx, mock_build_llm):
        cfg = MagicMock(spec=AppConfig)
        pipeline = AssistantPipeline(cfg)
        
        assert pipeline.cfg == cfg
        mock_build_llm.assert_called_once_with(cfg)
        mock_ctx.assert_called_once()
        assert pipeline.llm == mock_build_llm.return_value
        assert pipeline.ctx_provider == mock_ctx.return_value

    @patch("rdoai.pipeline.assistant.build_llm")
    @patch("rdoai.pipeline.assistant.build_user_prompt")
    def test_answer_success(self, mock_build_prompt, mock_build_llm):
        mock_llm = mock_build_llm.return_value
        mock_resp = MagicMock()
        mock_resp.text = "AI Response"
        mock_llm.chat.return_value = (mock_resp, None)
        mock_build_prompt.return_value = "Prompt Content"
        
        cfg = MagicMock(spec=AppConfig)
        cfg.llm.temperature = 0.7
        pipeline = AssistantPipeline(cfg)
        
        result = pipeline.answer("Hello", lang="en")
        
        assert isinstance(result, AssistantResult)
        assert result.answer == "AI Response"
        assert result.error is None
        
        mock_llm.chat.assert_called_once()
        args = mock_llm.chat.call_args[0][0]
        assert args[1]["content"] == "Prompt Content"
        assert mock_llm.chat.call_args[1]["temperature"] == 0.7

    @patch("rdoai.pipeline.assistant.build_llm")
    def test_answer_error(self, mock_build_llm):
        mock_llm = mock_build_llm.return_value
        mock_llm.chat.return_value = (None, "Service Down")
        
        cfg = MagicMock(spec=AppConfig)
        pipeline = AssistantPipeline(cfg)
        
        result = pipeline.answer("Hello")
        
        assert result.answer is None
        assert result.error == "Service Down"
