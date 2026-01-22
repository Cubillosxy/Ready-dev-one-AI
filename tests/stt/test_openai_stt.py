"""Tests for OpenAIStt."""
import pytest
from unittest.mock import MagicMock, patch, mock_open
from rdoai.stt.openai_stt import OpenAIStt
from rdoai.stt.base import Transcript


@pytest.fixture
def mock_openai(mocker):
    return mocker.patch("rdoai.stt.openai_stt.OpenAI")


class TestOpenAIStt:
    def test_init(self, mock_openai):
        stt = OpenAIStt(model="whisper-1")
        mock_openai.assert_called_once()
        assert stt.model == "whisper-1"

    @patch("rdoai.stt.openai_stt.open", new_callable=mock_open, read_data=b"audio data")
    def test_transcribe_success(self, mock_file, mock_openai):
        mock_instance = mock_openai.return_value
        mock_resp = MagicMock()
        mock_resp.text = "Hello from OpenAI"
        mock_instance.audio.transcriptions.create.return_value = mock_resp
        
        stt = OpenAIStt("whisper-1")
        result = stt.transcribe("test.wav", language="en")
        
        assert isinstance(result, Transcript)
        assert result.text == "Hello from OpenAI"
        assert result.language == "en"
        
        mock_instance.audio.transcriptions.create.assert_called_once()
        kwargs = mock_instance.audio.transcriptions.create.call_args[1]
        assert kwargs["model"] == "whisper-1"
        assert kwargs["language"] == "en"

    @patch("rdoai.stt.openai_stt.open", new_callable=mock_open, read_data=b"audio data")
    def test_transcribe_fallback_text(self, mock_file, mock_openai):
        mock_instance = mock_openai.return_value
        # Simulate response object that doesn't have .text but has a good __str__
        mock_resp = "Fallback Text"
        mock_instance.audio.transcriptions.create.return_value = mock_resp
        
        stt = OpenAIStt("whisper-1")
        result = stt.transcribe("test.wav")
        
        assert result.text == "Fallback Text"
