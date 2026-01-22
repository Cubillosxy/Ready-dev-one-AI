"""Tests for LocalStubStt."""
from rdoai.stt.local_stub import LocalStubStt
from rdoai.stt.base import Transcript


class TestLocalStubStt:
    def test_transcribe(self):
        stub = LocalStubStt()
        result = stub.transcribe("test.wav", language="en")
        
        assert isinstance(result, Transcript)
        assert result.text == "[LOCAL STT STUB] test.wav"
        assert result.language == "en"

    def test_transcribe_no_lang(self):
        stub = LocalStubStt()
        result = stub.transcribe("test.wav")
        assert result.language is None
