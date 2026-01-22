"""Tests for STT Factory."""
import pytest
from unittest.mock import patch, MagicMock
from rdoai.stt.factory import build_stt
from rdoai.config import SttConfig


class TestSttFactory:
    @patch("rdoai.stt.factory.OpenAIStt")
    def test_build_openai(self, mock_openai):
        cfg = SttConfig(provider="openai", openai_model="whisper-1")
        stt = build_stt(cfg)
        
        mock_openai.assert_called_once_with(model="whisper-1")
        assert stt == mock_openai.return_value

    @patch("rdoai.stt.factory.LocalVoskStt")
    def test_build_local(self, mock_vosk):
        cfg = SttConfig(provider="local", vosk_model_path="/path/to/model")
        stt = build_stt(cfg)
        
        mock_vosk.assert_called_once_with(model_path="/path/to/model")
        assert stt == mock_vosk.return_value

    def test_build_unknown(self):
        cfg = SttConfig(provider="unknown")
        with pytest.raises(ValueError, match="Unknown STT provider"):
            build_stt(cfg)

    def test_build_empty_provider(self):
        cfg = SttConfig(provider="")
        with pytest.raises(ValueError, match="Unknown STT provider"):
            build_stt(cfg)

    @patch("rdoai.stt.factory.OpenAIStt")
    def test_build_case_insensitive(self, mock_openai):
        cfg = SttConfig(provider=" OPENAI ")
        build_stt(cfg)
        mock_openai.assert_called_once()
