"""Tests for VoskStreamingStt."""
import json
import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from rdoai.stt.vosk_streaming import VoskStreamingStt
from rdoai.stt.streaming_base import StreamingResult


@pytest.fixture
def mock_vosk(mocker):
    # Mock Model and KaldiRecognizer
    mock_model_cls = mocker.patch("rdoai.stt.vosk_streaming.Model")
    mock_rec_cls = mocker.patch("rdoai.stt.vosk_streaming.KaldiRecognizer")
    
    mock_model = MagicMock()
    mock_rec = MagicMock()
    
    mock_model_cls.return_value = mock_model
    mock_rec_cls.return_value = mock_rec
    
    return {
        "model_cls": mock_model_cls,
        "rec_cls": mock_rec_cls,
        "model": mock_model,
        "rec": mock_rec
    }


class TestVoskStreamingStt:
    def test_init(self, mock_vosk):
        stt = VoskStreamingStt(model_path="/api/vosk", sample_rate=16000)
        
        mock_vosk["model_cls"].assert_called_once_with("/api/vosk")
        mock_vosk["rec_cls"].assert_called_once_with(mock_vosk["model"], 16000)
        mock_vosk["rec"].SetWords.assert_called_once_with(True)
        assert stt.sample_rate == 16000

    def test_accept_audio_final(self, mock_vosk):
        stt = VoskStreamingStt("/path", 16000)
        mock_rec = mock_vosk["rec"]
        
        # Setup: AcceptWaveform returns True (final)
        mock_rec.AcceptWaveform.return_value = True
        mock_rec.Result.return_value = json.dumps({"text": "hello world"})
        
        audio = np.array([0.5, -0.5], dtype=np.float32)
        result = stt.accept_audio(audio)
        
        assert isinstance(result, StreamingResult)
        assert result.text == "hello world"
        assert result.is_final is True
        assert stt._last_final == "hello world"

    def test_accept_audio_partial(self, mock_vosk):
        stt = VoskStreamingStt("/path", 16000)
        mock_rec = mock_vosk["rec"]
        
        # Setup: AcceptWaveform returns False (partial)
        mock_rec.AcceptWaveform.return_value = False
        mock_rec.PartialResult.return_value = json.dumps({"partial": "hello"})
        
        audio = np.array([0.1, 0.1], dtype=np.float32)
        result = stt.accept_audio(audio)
        
        assert isinstance(result, StreamingResult)
        assert result.text == "hello"
        assert result.is_final is False
        assert stt._last_partial == "hello"

    def test_accept_audio_throttle_partial(self, mock_vosk):
        stt = VoskStreamingStt("/path", 16000)
        mock_rec = mock_vosk["rec"]
        mock_rec.AcceptWaveform.return_value = False
        
        # Too short partial
        mock_rec.PartialResult.return_value = json.dumps({"partial": "he"})
        assert stt.accept_audio(np.zeros(10)) is None
        
        # Repeated partial
        mock_rec.PartialResult.return_value = json.dumps({"partial": "hello"})
        stt._last_partial = "hello"
        assert stt.accept_audio(np.zeros(10)) is None

    def test_accept_audio_dedup_final(self, mock_vosk):
        stt = VoskStreamingStt("/path", 16000)
        mock_rec = mock_vosk["rec"]
        mock_rec.AcceptWaveform.return_value = True
        mock_rec.Result.return_value = json.dumps({"text": "duplicate"})
        
        stt._last_final = "duplicate"
        assert stt.accept_audio(np.zeros(10)) is None

    def test_flush_final(self, mock_vosk):
        stt = VoskStreamingStt("/path", 16000)
        mock_rec = mock_vosk["rec"]
        mock_rec.FinalResult.return_value = json.dumps({"text": "final flush"})
        
        # State before flush
        stt._last_partial = "p"
        stt._last_final = "f"
        
        text = stt.flush_final()
        
        assert text == "final flush"
        # Verify reset
        assert stt._last_partial == ""
        assert stt._last_final == ""
        # Recognizer should be recreated
        assert mock_vosk["rec_cls"].call_count == 2
        mock_vosk["rec"].SetWords.assert_called_with(True)

    def test_audio_conversion(self, mock_vosk):
        stt = VoskStreamingStt("/path", 16000)
        mock_rec = mock_vosk["rec"]
        mock_rec.AcceptWaveform.return_value = False
        mock_rec.PartialResult.return_value = json.dumps({"partial": ""})
        
        audio = np.array([1.0, -1.0, 0.0], dtype=np.float32)
        stt.accept_audio(audio)
        
        # Get data passed to AcceptWaveform
        args, _ = mock_rec.AcceptWaveform.call_args
        data = args[0]
        
        # Convert back from bytes to int16
        restored = np.frombuffer(data, dtype=np.int16)
        expected = np.array([32767, -32767, 0], dtype=np.int16)
        np.testing.assert_array_equal(restored, expected)
