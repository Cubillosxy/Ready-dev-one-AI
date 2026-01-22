"""Basic tests for pipeline events."""
import numpy as np
from rdoai.pipeline.events import AudioChunk, SttPartial, SttFinal, ErrorEvent, FlushStt


class TestAudioChunk:
    """Test AudioChunk event."""

    def test_creation(self):
        """Test AudioChunk can be created."""
        audio_data = np.zeros(1024, dtype=np.float32)
        chunk = AudioChunk(audio=audio_data, ts=0.0)
        
        assert chunk.ts == 0.0
        assert isinstance(chunk.audio, np.ndarray)
        assert len(chunk.audio) == 1024

    def test_audio_attribute(self):
        """Test audio attribute is accessible."""
        audio_data = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        chunk = AudioChunk(audio=audio_data, ts=1.5)
        
        assert np.array_equal(chunk.audio, audio_data)
        assert chunk.ts == 1.5


class TestSttPartial:
    """Test SttPartial event."""

    def test_creation(self):
        """Test SttPartial can be created."""
        partial = SttPartial(text="hello world")
        assert partial.text == "hello world"

    def test_empty_text(self):
        """Test SttPartial with empty text."""
        partial = SttPartial(text="")
        assert partial.text == ""


class TestSttFinal:
    """Test SttFinal event."""

    def test_creation(self):
        """Test SttFinal can be created."""
        final = SttFinal(text="final transcription")
        assert final.text == "final transcription"

    def test_with_punctuation(self):
        """Test SttFinal with punctuation."""
        final = SttFinal(text="Hello, how are you?")
        assert final.text == "Hello, how are you?"


class TestErrorEvent:
    """Test ErrorEvent."""

    def test_creation_basic(self):
        """Test ErrorEvent can be created with basic params."""
        error = ErrorEvent(source="test", message="test error")
        assert error.source == "test"
        assert error.message == "test error"
        assert error.status_code is None

    def test_creation_with_status(self):
        """Test ErrorEvent with status code."""
        error = ErrorEvent(source="api", message="timeout", status_code=500)
        assert error.source == "api"
        assert error.message == "timeout"
        assert error.status_code == 500


class TestFlushStt:
    """Test FlushStt event."""

    def test_creation(self):
        """Test FlushStt can be created."""
        flush = FlushStt(reason="segment_end")
        assert flush.reason == "segment_end"

    def test_different_reasons(self):
        """Test FlushStt with different reasons."""
        flush1 = FlushStt(reason="manual")
        flush2 = FlushStt(reason="timeout")
        
        assert flush1.reason == "manual"
        assert flush2.reason == "timeout"
