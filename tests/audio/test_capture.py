"""Tests for AudioCapture class."""
import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from rdoai.audio.capture import AudioCapture, AudioFrame
from rdoai.config import AudioConfig


@pytest.fixture
def audio_cfg():
    return AudioConfig(
        device_index=1,
        sample_rate=16000,
        block_size=1024,
        channels=1,
        dtype="float32"
    )


@pytest.fixture
def mock_sd(mocker):
    # Mock sounddevice module
    mock = mocker.patch("rdoai.audio.capture.sd")
    # Default behavior for query_devices
    mock.query_devices.return_value = {"max_input_channels": 2}
    return mock


class TestAudioCapture:
    def test_init(self, audio_cfg):
        capture = AudioCapture(audio_cfg)
        assert capture.cfg == audio_cfg
        assert capture.stream is None
        assert capture.on_frame is None
        assert capture.on_error is None

    def test_start_success(self, audio_cfg, mock_sd):
        capture = AudioCapture(audio_cfg)
        capture.start()

        mock_sd.query_devices.assert_called_once_with(audio_cfg.device_index)
        mock_sd.InputStream.assert_called_once()
        
        # Check InputStream args (partial check)
        args, kwargs = mock_sd.InputStream.call_args
        assert kwargs["device"] == audio_cfg.device_index
        assert kwargs["samplerate"] == audio_cfg.sample_rate
        assert kwargs["channels"] == audio_cfg.channels
        assert kwargs["callback"] == capture._callback

        assert capture.stream is not None
        capture.stream.start.assert_called_once()

    def test_start_no_input_channels(self, audio_cfg, mock_sd):
        mock_sd.query_devices.return_value = {"max_input_channels": 0}
        capture = AudioCapture(audio_cfg)
        
        # Should raise RuntimeError if no on_error set
        with pytest.raises(RuntimeError, match="has no input channels"):
            capture.start()

    def test_start_error_with_callback(self, audio_cfg, mock_sd):
        mock_sd.query_devices.side_effect = Exception("SD Error")
        capture = AudioCapture(audio_cfg)
        
        error_received = None
        def on_error(e):
            nonlocal error_received
            error_received = e
            
        capture.on_error = on_error
        capture.start()
        
        assert isinstance(error_received, Exception)
        assert str(error_received) == "SD Error"
        assert capture.stream is None

    def test_stop(self, audio_cfg, mock_sd):
        capture = AudioCapture(audio_cfg)
        capture.start()
        mock_stream = capture.stream
        
        capture.stop()
        
        mock_stream.stop.assert_called_once()
        mock_stream.close.assert_called_once()
        assert capture.stream is None

    def test_stop_no_stream(self, audio_cfg):
        capture = AudioCapture(audio_cfg)
        capture.stop()  # Should not raise anything

    def test_callback_executes_on_frame(self, audio_cfg, mock_sd):
        capture = AudioCapture(audio_cfg)
        
        received_frame = None
        def on_frame(f):
            nonlocal received_frame
            received_frame = f
            
        capture.on_frame = on_frame
        
        # Simulate callback from sounddevice
        indata = np.random.rand(1024, 1).astype(np.float32)
        capture._callback(indata, 1024, None, None)
        
        assert isinstance(received_frame, AudioFrame)
        assert received_frame.frames == 1024
        # Verify it took the first channel
        np.testing.assert_array_equal(received_frame.audio, indata[:, 0])
        assert isinstance(received_frame.timestamp, float)

    def test_callback_error_handling(self, audio_cfg):
        capture = AudioCapture(audio_cfg)
        
        # This will cause an error in _callback: indata is None
        error_received = None
        def on_error(e):
            nonlocal error_received
            error_received = e
            
        capture.on_error = on_error
        capture._callback(None, 0, None, None)
        
        assert isinstance(error_received, TypeError)


class TestAudioFrame:
    def test_audio_frame_dataclass(self):
        audio = np.array([0.1, 0.2], dtype=np.float32)
        frame = AudioFrame(audio=audio, frames=2, timestamp=123.456)
        
        assert np.array_equal(frame.audio, audio)
        assert frame.frames == 2
        assert frame.timestamp == 123.456
