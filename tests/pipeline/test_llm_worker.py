"""Tests for LlmWorker."""
import pytest
from unittest.mock import MagicMock, patch
from rdoai.pipeline.llm_worker import LlmWorker, LlmJob, LlmEvent
from rdoai.pipeline.assistant import AssistantResult


class TestLlmWorker:
    @patch("rdoai.pipeline.llm_worker.AssistantPipeline")
    def test_init(self, mock_assistant):
        publish = MagicMock()
        cfg = MagicMock()
        worker = LlmWorker(cfg, publish)
        
        assert worker.cfg == cfg
        assert worker.publish == publish
        mock_assistant.assert_called_once_with(cfg)
        assert worker._last_sent == ""

    @patch("rdoai.pipeline.llm_worker.AssistantPipeline")
    def test_handle_valid_job(self, mock_assistant_cls):
        mock_assistant = mock_assistant_cls.return_value
        mock_assistant.answer.return_value = AssistantResult(answer="AI Answer", error=None)
        
        publish = MagicMock()
        worker = LlmWorker(MagicMock(), publish)
        job = LlmJob(question="How are you?", lang="es")
        
        worker.handle(job)
        
        mock_assistant.answer.assert_called_once_with("How are you?", lang="es")
        publish.assert_called_once()
        event = publish.call_args[0][0]
        assert isinstance(event, LlmEvent)
        assert event.kind == "answer"
        assert event.answer == "AI Answer"
        assert event.question == "How are you?"
        assert worker._last_sent == "How are you?"

    @patch("rdoai.pipeline.llm_worker.AssistantPipeline")
    def test_handle_deduplication(self, mock_assistant_cls):
        mock_assistant = mock_assistant_cls.return_value
        mock_assistant.answer.return_value = AssistantResult(answer="Ans", error=None)
        
        publish = MagicMock()
        worker = LlmWorker(MagicMock(), publish)
        job = LlmJob(question="Same")
        
        worker.handle(job)
        worker.handle(job)  # Duplicate
        
        assert mock_assistant.answer.call_count == 1
        assert publish.call_count == 1

    @patch("rdoai.pipeline.llm_worker.AssistantPipeline")
    def test_handle_empty_question(self, mock_assistant_cls):
        publish = MagicMock()
        worker = LlmWorker(MagicMock(), publish)
        
        worker.handle(LlmJob(question="  "))
        
        publish.assert_not_called()

    @patch("rdoai.pipeline.llm_worker.AssistantPipeline")
    def test_handle_error(self, mock_assistant_cls):
        mock_assistant = mock_assistant_cls.return_value
        mock_assistant.answer.return_value = AssistantResult(answer=None, error="API Error")
        
        publish = MagicMock()
        worker = LlmWorker(MagicMock(), publish)
        
        worker.handle(LlmJob(question="Test Error"))
        
        publish.assert_called_once()
        event = publish.call_args[0][0]
        assert event.kind == "error"
        assert event.error == "API Error"
