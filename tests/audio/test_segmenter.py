"""Comprehensive tests for AutoSegmenter."""
import os
import tempfile
import time
import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from rdoai.config import VadConfig
from rdoai.audio.segmenter import AutoSegmenter, SegmentResult


class TestAutoSegmenterInitialization:
    """Test AutoSegmenter initialization and basic state."""

    def test_instantiation(self):
        """Test that AutoSegmenter can be instantiated."""
        vad_cfg = VadConfig()
        with tempfile.TemporaryDirectory() as tmpdir:
            segmenter = AutoSegmenter(
                vad=vad_cfg,
                output_dir=tmpdir,
                sample_rate=16000
            )
            
            assert segmenter.sample_rate == 16000
            assert segmenter.output_dir == tmpdir
            assert os.path.exists(tmpdir)

    def test_initial_state(self):
        """Test initial state of AutoSegmenter."""
        vad_cfg = VadConfig()
        with tempfile.TemporaryDirectory() as tmpdir:
            segmenter = AutoSegmenter(
                vad=vad_cfg,
                output_dir=tmpdir,
                sample_rate=16000
            )
            
            assert segmenter.state == "silence"
            assert segmenter.in_segment is False
            assert segmenter.segment_frames == []
            assert segmenter.last_pause == 0.0

    def test_vad_config_applied(self):
        """Test that VAD config values are applied."""
        vad_cfg = VadConfig(
            silence_threshold_db=-50.0,
            speech_threshold_db=-45.0,
            min_silence_sec=1.0
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            segmenter = AutoSegmenter(
                vad=vad_cfg,
                output_dir=tmpdir,
                sample_rate=16000
            )
            
            assert segmenter.silence_threshold_db == -50.0
            assert segmenter.speech_threshold_db == -45.0
            assert segmenter.min_silence_sec == 1.0

    def test_reset_segment(self):
        """Test reset_segment method."""
        vad_cfg = VadConfig()
        with tempfile.TemporaryDirectory() as tmpdir:
            segmenter = AutoSegmenter(
                vad=vad_cfg,
                output_dir=tmpdir,
                sample_rate=16000
            )
            
            # Manually set some state
            segmenter.in_segment = True
            segmenter.segment_frames = [1, 2, 3]
            
            # Reset
            segmenter.reset_segment()
            
            assert segmenter.in_segment is False
            assert segmenter.segment_frames == []


class TestAutoSegmenterSpeechDetection:
    """Test speech detection logic."""

    def test_transition_to_speech(self):
        """Test transition from silence to speech."""
        vad_cfg = VadConfig(
            silence_threshold_db=-50.0,
            speech_threshold_db=-45.0,
            min_silence_sec=0.5
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            segmenter = AutoSegmenter(vad_cfg, tmpdir, 16000)
            
            # Start in silence
            assert segmenter.state == "silence"
            
            # Send audio with level above speech threshold
            audio = np.random.random(1024).astype(np.float32)
            result = segmenter.process(audio, level_db=-40.0)
            
            # Should transition to speech
            assert segmenter.state == "speech"
            assert segmenter.in_segment is True
            assert len(segmenter.segment_frames) == 1
            assert result is None  # No segment produced yet

    def test_stays_in_silence_below_threshold(self):
        """Test that segmenter stays in silence when level is below threshold."""
        vad_cfg = VadConfig(speech_threshold_db=-45.0)
        with tempfile.TemporaryDirectory() as tmpdir:
            segmenter = AutoSegmenter(vad_cfg, tmpdir, 16000)
            
            # Send quiet audio
            audio = np.random.random(1024).astype(np.float32) * 0.001
            result = segmenter.process(audio, level_db=-60.0)
            
            assert segmenter.state == "silence"
            assert segmenter.in_segment is False
            assert result is None

    def test_accumulates_frames_during_speech(self):
        """Test that frames are accumulated during speech."""
        vad_cfg = VadConfig(
            silence_threshold_db=-50.0,
            speech_threshold_db=-45.0
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            segmenter = AutoSegmenter(vad_cfg, tmpdir, 16000)
            
            # Start speech
            audio1 = np.random.random(1024).astype(np.float32)
            segmenter.process(audio1, level_db=-40.0)
            
            # Continue speech
            audio2 = np.random.random(1024).astype(np.float32)
            segmenter.process(audio2, level_db=-40.0)
            
            audio3 = np.random.random(1024).astype(np.float32)
            segmenter.process(audio3, level_db=-40.0)
            
            # Should have accumulated 3 frames
            assert len(segmenter.segment_frames) == 3
            assert segmenter.state == "speech"


class TestAutoSegmenterSilenceDetection:
    """Test silence detection and segment finalization."""

    def test_transition_to_silence(self):
        """Test transition from speech to silence."""
        vad_cfg = VadConfig(
            silence_threshold_db=-50.0,
            speech_threshold_db=-45.0,
            min_silence_sec=0.5
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            segmenter = AutoSegmenter(vad_cfg, tmpdir, 16000)
            
            # Start speech
            audio = np.random.random(1024).astype(np.float32)
            segmenter.process(audio, level_db=-40.0)
            assert segmenter.state == "speech"
            
            # Go to silence
            segmenter.process(audio, level_db=-55.0)
            
            assert segmenter.state == "silence"
            assert segmenter.in_segment is True  # Still in segment until min_silence_sec

    @patch('rdoai.audio.segmenter.time.time')
    @patch('rdoai.audio.segmenter.write_wav_mono_int16')
    def test_segment_finalization_after_min_silence(self, mock_write, mock_time):
        """Test that segment is finalized after min_silence_sec."""
        vad_cfg = VadConfig(
            silence_threshold_db=-50.0,
            speech_threshold_db=-45.0,
            min_silence_sec=0.5  # 500ms
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            segmenter = AutoSegmenter(vad_cfg, tmpdir, 16000)
            
            # Mock time progression
            base_time = 1000.0
            mock_time.side_effect = [
                base_time,      # __init__
                base_time + 1.0,  # Start speech
                base_time + 2.0,  # Continue speech
                base_time + 3.0,  # Transition to silence
                base_time + 3.6,  # Still in silence, >= min_silence_sec
            ]
            
            audio = np.random.random(1024).astype(np.float32)
            
            # Start speech
            result = segmenter.process(audio, level_db=-40.0)
            assert result is None
            
            # Continue speech
            result = segmenter.process(audio, level_db=-40.0)
            assert result is None
            
            # Go to silence
            result = segmenter.process(audio, level_db=-55.0)
            assert result is None
            
            # Wait long enough (>= min_silence_sec)
            result = segmenter.process(audio, level_db=-55.0)
            
            # Should produce a segment
            assert result is not None
            assert isinstance(result, SegmentResult)
            assert result.duration_sec > 0
            assert os.path.basename(result.wav_path).startswith("q_")
            assert mock_write.called

    def test_no_segment_if_silence_too_short(self):
        """Test that segment is not finalized if silence duration is too short."""
        vad_cfg = VadConfig(
            silence_threshold_db=-50.0,
            speech_threshold_db=-45.0,
            min_silence_sec=2.0  # 2 seconds
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            segmenter = AutoSegmenter(vad_cfg, tmpdir, 16000)
            
            audio = np.random.random(1024).astype(np.float32)
            
            # Start speech
            segmenter.process(audio, level_db=-40.0)
            
            # Go to silence
            segmenter.process(audio, level_db=-55.0)
            
            # Wait a bit but not long enough (< 2 seconds)
            time.sleep(0.1)
            result = segmenter.process(audio, level_db=-55.0)
            
            # Should not produce segment yet
            assert result is None
            assert segmenter.in_segment is True


class TestAutoSegmenterEdgeCases:
    """Test edge cases and special scenarios."""

    def test_empty_segment_not_written(self):
        """Test that empty segments are handled gracefully."""
        vad_cfg = VadConfig(min_silence_sec=0.1)
        with tempfile.TemporaryDirectory() as tmpdir:
            segmenter = AutoSegmenter(vad_cfg, tmpdir, 16000)
            
            # Manually create an empty segment situation
            segmenter.in_segment = True
            segmenter.segment_frames = []
            segmenter.state = "silence"
            
            # This should handle empty frames gracefully
            audio = np.random.random(1024).astype(np.float32)
            result = segmenter.process(audio, level_db=-55.0)
            
            # Should not crash

    def test_last_pause_tracking(self):
        """Test that last_pause is updated correctly."""
        vad_cfg = VadConfig(speech_threshold_db=-45.0)
        with tempfile.TemporaryDirectory() as tmpdir:
            segmenter = AutoSegmenter(vad_cfg, tmpdir, 16000)
            
            # Let some time pass in silence
            time.sleep(0.1)
            
            # Transition to speech
            audio = np.random.random(1024).astype(np.float32)
            segmenter.process(audio, level_db=-40.0)
            
            # last_pause should be updated
            assert segmenter.last_pause > 0.0

    def test_segment_result_attributes(self):
        """Test that SegmentResult has correct attributes."""
        result = SegmentResult(
            wav_path="/tmp/test.wav",
            duration_sec=1.5,
            timestamp=1234567890.0
        )
        
        assert result.wav_path == "/tmp/test.wav"
        assert result.duration_sec == 1.5
        assert result.timestamp == 1234567890.0

    def test_audio_blocks_are_copied(self):
        """Test that audio blocks are copied, not referenced."""
        vad_cfg = VadConfig(speech_threshold_db=-45.0)
        with tempfile.TemporaryDirectory() as tmpdir:
            segmenter = AutoSegmenter(vad_cfg, tmpdir, 16000)
            
            # Create audio block
            audio = np.random.random(1024).astype(np.float32)
            original_values = audio.copy()
            
            # Start speech
            segmenter.process(audio, level_db=-40.0)
            
            # Modify original array
            audio *= 2.0
            
            # Segmenter should have a copy, not the modified version
            stored_audio = segmenter.segment_frames[0]
            np.testing.assert_array_equal(stored_audio, original_values)
