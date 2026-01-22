"""Comprehensive tests for rms_db function."""
import numpy as np
import pytest
from rdoai.audio.levels import rms_db


class TestRmsDb:
    """Test rms_db function for audio level calculation."""

    def test_empty_array_returns_negative_160(self):
        """Test that empty array returns -160.0 dB."""
        audio = np.array([], dtype=np.float32)
        result = rms_db(audio)
        assert result == -160.0

    def test_silence_returns_very_low_db(self):
        """Test that silence (zeros) returns very low dB value."""
        audio = np.zeros(1024, dtype=np.float32)
        result = rms_db(audio)
        # For zeros, rms is 0, clamped to 1e-8, so 20*log10(1e-8) = -160
        assert result == pytest.approx(-160.0, abs=0.1)

    def test_known_amplitude(self):
        """Test rms_db with known amplitude.
        
        For a constant signal of amplitude A, RMS = A
        dB = 20 * log10(A)
        """
        # Test with amplitude 0.1
        audio = np.full(1000, 0.1, dtype=np.float32)
        expected_db = 20.0 * np.log10(0.1)  # Should be -20 dB
        result = rms_db(audio)
        assert result == pytest.approx(expected_db, abs=0.01)

    def test_half_amplitude(self):
        """Test with 0.5 amplitude (standard test signal)."""
        audio = np.full(1000, 0.5, dtype=np.float32)
        expected_db = 20.0 * np.log10(0.5)  # Should be about -6 dB
        result = rms_db(audio)
        assert result == pytest.approx(expected_db, abs=0.01)

    def test_full_amplitude(self):
        """Test with full scale amplitude (1.0)."""
        audio = np.full(1000, 1.0, dtype=np.float32)
        expected_db = 20.0 * np.log10(1.0)  # Should be 0 dB
        result = rms_db(audio)
        assert result == pytest.approx(0.0, abs=0.01)

    def test_sine_wave(self):
        """Test with a sine wave.
        
        For a sine wave, RMS = amplitude / sqrt(2)
        """
        sample_rate = 16000
        duration = 1.0
        frequency = 440.0
        amplitude = 0.5
        
        t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)
        audio = amplitude * np.sin(2 * np.pi * frequency * t).astype(np.float32)
        
        # RMS of sine = amplitude / sqrt(2)
        expected_rms = amplitude / np.sqrt(2)
        expected_db = 20.0 * np.log10(expected_rms)
        
        result = rms_db(audio)
        assert result == pytest.approx(expected_db, abs=0.1)

    def test_very_quiet_audio(self):
        """Test with very quiet audio (near silence)."""
        audio = np.full(1000, 1e-6, dtype=np.float32)
        result = rms_db(audio)
        # Should be negative (quiet) but not hit the floor of -160
        assert result < -100
        assert result > -160

    def test_loud_audio(self):
        """Test with loud audio (clipping level)."""
        audio = np.full(1000, 0.9, dtype=np.float32)
        expected_db = 20.0 * np.log10(0.9)
        result = rms_db(audio)
        assert result == pytest.approx(expected_db, abs=0.01)

    def test_negative_samples(self):
        """Test that negative samples are handled correctly.
        
        RMS should be the same for positive and negative signals.
        """
        audio_positive = np.full(1000, 0.5, dtype=np.float32)
        audio_negative = np.full(1000, -0.5, dtype=np.float32)
        
        result_positive = rms_db(audio_positive)
        result_negative = rms_db(audio_negative)
        
        assert result_positive == pytest.approx(result_negative, abs=0.01)

    def test_mixed_positive_negative(self):
        """Test with alternating positive/negative samples."""
        # Create square wave alternating between +0.5 and -0.5
        audio = np.array([0.5, -0.5] * 500, dtype=np.float32)
        expected_db = 20.0 * np.log10(0.5)
        result = rms_db(audio)
        assert result == pytest.approx(expected_db, abs=0.01)

    def test_single_sample(self):
        """Test with single sample."""
        audio = np.array([0.5], dtype=np.float32)
        expected_db = 20.0 * np.log10(0.5)
        result = rms_db(audio)
        assert result == pytest.approx(expected_db, abs=0.01)

    def test_returns_float(self):
        """Test that function returns a float."""
        audio = np.random.random(100).astype(np.float32)
        result = rms_db(audio)
        assert isinstance(result, float)

    def test_floor_prevents_log_of_zero(self):
        """Test that the 1e-8 floor prevents log(0) errors."""
        # This test verifies the max(rms, 1e-8) line works correctly
        audio = np.zeros(100, dtype=np.float32)
        # Should not raise ValueError for log(0)
        try:
            result = rms_db(audio)
            assert result == pytest.approx(-160.0, abs=0.1)
        except ValueError:
            pytest.fail("rms_db raised ValueError (log of zero)")

    def test_random_noise(self):
        """Test with random noise."""
        np.random.seed(42)  # For reproducibility
        audio = np.random.uniform(-0.1, 0.1, 10000).astype(np.float32)
        result = rms_db(audio)
        # Random uniform noise between -0.1 and 0.1 should have RMS around 0.058
        # which is about -24.7 dB
        assert -30 < result < -20
