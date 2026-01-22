"""Tests for StreamingSttWorker."""
import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from rdoai.pipeline.streaming_stt_worker import StreamingSttWorker
from rdoai.pipeline.events import AudioChunk, SttPartial, SttFinal, ErrorEvent, FlushStt
from rdoai.stt.streaming_base import StreamingResult


class TestStreamingSttWorker:
    @patch("rdoai.pipeline.streaming_stt_worker.VoskStreamingStt")
    def test_init(self, mock_engine_cls):
        publish = MagicMock()
        worker = StreamingSttWorker(model_path="/m", sample_rate=16, publish=publish)
        
        mock_engine_cls.assert_called_once_with(model_path="/m", sample_rate=16)
        assert worker.publish == publish
        assert worker.engine == mock_engine_cls.return_value

    @patch("rdoai.pipeline.streaming_stt_worker.VoskStreamingStt")
    def test_on_audio_chunk_partial(self, mock_engine_cls):
        mock_engine = mock_engine_cls.return_value
        mock_engine.accept_audio.return_value = StreamingResult(text="part", is_final=False)
        
        publish = MagicMock()
        worker = StreamingSttWorker("/m", 16, publish)
        
        audio = np.zeros(10)
        worker.on_audio(AudioChunk(audio, ts=0.0))
        
        mock_engine.accept_audio.assert_called_once()
        publish.assert_called_once()
        event = publish.call_args[0][0]
        assert isinstance(event, SttPartial)
        assert event.text == "part"

    @patch("rdoai.pipeline.streaming_stt_worker.VoskStreamingStt")
    def test_on_audio_chunk_final(self, mock_engine_cls):
        mock_engine = mock_engine_cls.return_value
        mock_engine.accept_audio.return_value = StreamingResult(text="done", is_final=True)
        
        publish = MagicMock()
        worker = StreamingSttWorker("/m", 16, publish)
        
        worker.on_audio(AudioChunk(np.zeros(10), ts=0.0))
        
        event = publish.call_args[0][0]
        assert isinstance(event, SttFinal)
        assert event.text == "done"

    @patch("rdoai.pipeline.streaming_stt_worker.VoskStreamingStt")
    def test_on_flush(self, mock_engine_cls):
        mock_engine = mock_engine_cls.return_value
        mock_engine.flush_final.return_value = "finalized"
        
        publish = MagicMock()
        worker = StreamingSttWorker("/m", 16, publish)
        
        worker.on_audio(FlushStt(reason="test"))
        
        mock_engine.flush_final.assert_called_once()
        event = publish.call_args[0][0]
        assert isinstance(event, SttFinal)
        assert event.text == "finalized"

    @patch("rdoai.pipeline.streaming_stt_worker.VoskStreamingStt")
    def test_on_error(self, mock_engine_cls):
        mock_engine = mock_engine_cls.return_value
        mock_engine.accept_audio.side_effect = Exception("STT Fault")
        
        publish = MagicMock()
        worker = StreamingSttWorker("/m", 16, publish)
        
        worker.on_audio(AudioChunk(np.zeros(10), ts=0.0))
        
        event = publish.call_args[0][0]
        assert isinstance(event, ErrorEvent)
        assert event.source == "streaming_stt"
        assert "STT Fault" in event.message
