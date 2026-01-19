"""Basic tests for configuration module."""
import pytest
from rdoai.config import (
    AudioConfig,
    VadConfig,
    MicVadConfig,
    UiConfig,
    SttConfig,
    StreamingSttConfig,
    LlmConfig,
    AssistantConfig,
    InputConfig,
    LanguageConfig,
    AppConfig,
)


class TestAudioConfig:
    """Test AudioConfig dataclass."""

    def test_default_values(self):
        """Test that AudioConfig can be instantiated with defaults."""
        config = AudioConfig()
        assert config.sample_rate == 16000
        assert config.block_size == 1024
        assert config.channels == 1
        assert config.dtype == "float32"

    def test_custom_values(self):
        """Test AudioConfig with custom values."""
        config = AudioConfig(sample_rate=48000, channels=2)
        assert config.sample_rate == 48000
        assert config.channels == 2


class TestVadConfig:
    """Test VadConfig dataclass."""

    def test_default_values(self):
        """Test that VadConfig can be instantiated with defaults."""
        config = VadConfig()
        assert config.silence_threshold_db == -45.0
        assert config.speech_threshold_db == -40.0
        assert config.min_silence_sec == 0.6

    def test_frozen(self):
        """Test that VadConfig is immutable."""
        config = VadConfig()
        with pytest.raises(Exception):  # FrozenInstanceError
            config.silence_threshold_db = -50.0


class TestMicVadConfig:
    """Test MicVadConfig dataclass."""

    def test_default_values(self):
        """Test that MicVadConfig can be instantiated with defaults."""
        config = MicVadConfig()
        assert config.silence_threshold_db == -50.0
        assert config.speech_threshold_db == -45.0
        assert config.min_silence_sec == 2.0


class TestUiConfig:
    """Test UiConfig dataclass."""

    def test_default_values(self):
        """Test that UiConfig can be instantiated with defaults."""
        config = UiConfig()
        assert config.window_alpha == 0.88
        assert config.always_on_top is True
        assert isinstance(config.geometry, str)
        assert "x" in config.geometry


class TestAppConfig:
    """Test AppConfig dataclass."""

    def test_default_instantiation(self):
        """Test that AppConfig can be instantiated with defaults."""
        config = AppConfig()
        assert config.output_dir == "captures"
        assert isinstance(config.audio, AudioConfig)
        assert isinstance(config.vad, VadConfig)
        assert isinstance(config.mic_vad, MicVadConfig)
        assert isinstance(config.ui, UiConfig)
        assert isinstance(config.stt, SttConfig)
        assert isinstance(config.streaming_stt, StreamingSttConfig)
        assert isinstance(config.llm, LlmConfig)
        assert isinstance(config.assistant, AssistantConfig)
        assert isinstance(config.input, InputConfig)
        assert isinstance(config.lang, LanguageConfig)

    def test_nested_config_access(self):
        """Test accessing nested configuration values."""
        config = AppConfig()
        assert config.audio.sample_rate == 16000
        assert config.vad.min_silence_sec == 0.6
        assert config.ui.window_alpha == 0.88
