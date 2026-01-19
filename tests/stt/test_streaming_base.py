"""Basic tests for streaming STT base classes."""
from rdoai.stt.streaming_base import StreamingResult


class TestStreamingResult:
    """Test StreamingResult dataclass."""

    def test_creation_partial(self):
        """Test creating a partial result."""
        result = StreamingResult(text="hello", is_final=False)
        
        assert result.text == "hello"
        assert result.is_final is False

    def test_creation_final(self):
        """Test creating a final result."""
        result = StreamingResult(text="hello world", is_final=True)
        
        assert result.text == "hello world"
        assert result.is_final is True

    def test_empty_text(self):
        """Test StreamingResult with empty text."""
        result = StreamingResult(text="", is_final=True)
        
        assert result.text == ""
        assert result.is_final is True

    def test_text_with_punctuation(self):
        """Test StreamingResult with punctuation."""
        result = StreamingResult(text="Hello, how are you?", is_final=False)
        
        assert result.text == "Hello, how are you?"
        assert result.is_final is False
