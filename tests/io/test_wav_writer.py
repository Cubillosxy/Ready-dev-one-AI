"""Tests for wav_writer utility."""
import os
import wave
import tempfile
import numpy as np
import pytest
from rdoai.io.wav_writer import write_wav_mono_int16


class TestWavWriter:
    def test_write_wav_mono_int16_success(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.wav")
            # Sine wave 440Hz
            sr = 16000
            t = np.linspace(0, 0.1, int(sr * 0.1), dtype=np.float32)
            audio = 0.5 * np.sin(2 * np.pi * 440.0 * t)
            
            write_wav_mono_int16(path, audio, sr)
            
            assert os.path.exists(path)
            
            # Verify with wave module
            with wave.open(path, "rb") as wf:
                assert wf.getnchannels() == 1
                assert wf.getsampwidth() == 2  # 16-bit
                assert wf.getframerate() == sr
                assert wf.getnframes() == len(audio)
                
                # Check some values (simple check)
                data = wf.readframes(wf.getnframes())
                restored = np.frombuffer(data, dtype=np.int16)
                assert len(restored) == len(audio)
                # Max value should be around 0.5 * 32767 = 16383
                assert np.max(restored) <= 16384
                assert np.min(restored) >= -16384

    def test_write_wav_clipping(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "clipped.wav")
            # Over-scale audio
            audio = np.array([2.0, -2.0, 0.5], dtype=np.float32)
            
            write_wav_mono_int16(path, audio, 16000)
            
            with wave.open(path, "rb") as wf:
                data = wf.readframes(3)
                restored = np.frombuffer(data, dtype=np.int16)
                # Should be clipped to max int16
                assert restored[0] == 32767
                assert restored[1] == -32767

    def test_write_wav_empty(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "empty.wav")
            audio = np.array([], dtype=np.float32)
            
            write_wav_mono_int16(path, audio, 16000)
            
            assert os.path.exists(path)
            with wave.open(path, "rb") as wf:
                assert wf.getnframes() == 0
