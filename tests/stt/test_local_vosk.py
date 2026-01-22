"""Tests for LocalVoskStt."""
import json
import pytest
import wave
from unittest.mock import MagicMock, patch
from rdoai.stt.local_vosk import LocalVoskStt
from rdoai.stt.base import Transcript


@pytest.fixture
def mock_vosk(mocker):
    mock_model_cls = mocker.patch("rdoai.stt.local_vosk.Model")
    mock_rec_cls = mocker.patch("rdoai.stt.local_vosk.KaldiRecognizer")
    
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


class TestLocalVoskStt:
    def test_init(self, mock_vosk):
        stt = LocalVoskStt(model_path="/m")
        mock_vosk["model_cls"].assert_called_once_with("/m")
        assert stt.model == mock_vosk["model"]

    @patch("rdoai.stt.local_vosk.wave.open")
    def test_transcribe_success(self, mock_wave_open, mock_vosk):
        stt = LocalVoskStt("/m")
        mock_rec = mock_vosk["rec"]
        
        # Setup mock wave file
        mock_wf = MagicMock()
        mock_wf.getnchannels.return_value = 1
        mock_wf.getsampwidth.return_value = 2
        mock_wf.getframerate.return_value = 16000
        mock_wf.readframes.side_effect = [b"data1", b"data2", b""] # Emit twice then stop
        mock_wave_open.return_value.__enter__.return_value = mock_wf
        
        mock_rec.FinalResult.return_value = json.dumps({"text": "done transcribing"})
        
        result = stt.transcribe("dummy.wav", language="fr")
        
        assert isinstance(result, Transcript)
        assert result.text == "done transcribing"
        assert result.language == "fr"
        
        mock_rec.AcceptWaveform.assert_any_call(b"data1")
        mock_rec.AcceptWaveform.assert_any_call(b"data2")
        mock_vosk["rec_cls"].assert_called_once_with(mock_vosk["model"], 16000)

    @patch("rdoai.stt.local_vosk.wave.open")
    def test_transcribe_invalid_channels(self, mock_wave_open, mock_vosk):
        stt = LocalVoskStt("/m")
        mock_wf = MagicMock()
        mock_wf.getnchannels.return_value = 2 # Stereo
        mock_wf.getsampwidth.return_value = 2
        mock_wave_open.return_value.__enter__.return_value = mock_wf
        
        with pytest.raises(ValueError, match="Expected mono WAV"):
            stt.transcribe("dummy.wav")

    @patch("rdoai.stt.local_vosk.wave.open")
    def test_transcribe_invalid_width(self, mock_wave_open, mock_vosk):
        stt = LocalVoskStt("/m")
        mock_wf = MagicMock()
        mock_wf.getnchannels.return_value = 1
        mock_wf.getsampwidth.return_value = 1 # 8-bit
        mock_wave_open.return_value.__enter__.return_value = mock_wf
        
        with pytest.raises(ValueError, match="Expected 16-bit PCM WAV"):
            stt.transcribe("dummy.wav")

    @patch("rdoai.stt.local_vosk.wave.open")
    def test_transcribe_default_language(self, mock_wave_open, mock_vosk):
        stt = LocalVoskStt("/m")
        mock_wf = MagicMock()
        mock_wf.getnchannels.return_value = 1
        mock_wf.getsampwidth.return_value = 2
        mock_wf.getframerate.return_value = 16000
        mock_wf.readframes.return_value = b""
        mock_wave_open.return_value.__enter__.return_value = mock_wf
        
        mock_vosk["rec"].FinalResult.return_value = json.dumps({"text": ""})
        
        result = stt.transcribe("dummy.wav")
        assert result.language == "en"
