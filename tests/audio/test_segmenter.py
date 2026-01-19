"""Basic tests for AutoSegmenter."""
import os
import tempfile
from rdoai.config import VadConfig
from rdoai.audio.segmenter import AutoSegmenter


class TestAutoSegmenter:
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
