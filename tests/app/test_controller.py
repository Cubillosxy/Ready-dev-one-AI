"""Tests for AppController."""
import pytest
import numpy as np
import queue
from unittest.mock import MagicMock, patch
from rdoai.app.controller import AppController
from rdoai.pipeline.events import SttPartial, SttFinal, AudioChunk
from rdoai.pipeline.llm_worker import LlmEvent
from rdoai.audio.capture import AudioFrame


@pytest.fixture
def mock_deps(mocker):
    # Mocking external subsystems to avoid side effects (threads, UI, hardware)
    deps = {
        "OverlayWindow": mocker.patch("rdoai.app.controller.OverlayWindow"),
        "AudioCapture": mocker.patch("rdoai.app.controller.AudioCapture"),
        "Worker": mocker.patch("rdoai.app.controller.Worker"),
        "StreamingSttWorker": mocker.patch("rdoai.app.controller.StreamingSttWorker"),
        "LlmWorker": mocker.patch("rdoai.app.controller.LlmWorker"),
        "build_stt": mocker.patch("rdoai.app.controller.build_stt"),
        "AssistantPipeline": mocker.patch("rdoai.app.controller.AssistantPipeline"),
        "sd": mocker.patch("rdoai.app.controller.sd")
    }
    # Default behavior for sd
    deps["sd"].query_devices.return_value = {"name": "Mock Device"}
    return deps


@pytest.fixture
def app_cfg():
    from rdoai.config import AppConfig, AudioConfig, VadConfig, MicVadConfig, LanguageConfig, UiConfig, SttConfig, AssistantConfig, InputConfig, StreamingSttConfig
    return AppConfig(
        output_dir="test_output",
        lang=LanguageConfig(lang="en", vosk_model_en="/m/en", vosk_model_es="/m/es"),
        input=InputConfig(mode="mic", mic_device_index=1, meeting_device_index=2),
        audio=AudioConfig(sample_rate=16000, device_index=1, block_size=1024, channels=1),
        vad=VadConfig(min_silence_sec=0.5, silence_threshold_db=-60.0, speech_threshold_db=-45.0),
        mic_vad=MicVadConfig(min_silence_sec=1.0, silence_threshold_db=-55.0, speech_threshold_db=-40.0),
        ui=UiConfig(geometry="400x200", window_alpha=0.8, always_on_top=True),
        stt=SttConfig(provider="local", vosk_model_path="/m/en", language="en"),
        streaming_stt=StreamingSttConfig(provider="local", vosk_model_path="/m/en"),
        assistant=AssistantConfig(trigger="on_pause"),
        debug_print_every_sec=1.0
    )


class TestAppController:
    def test_init(self, mock_deps, app_cfg):
        controller = AppController(app_cfg)
        
        assert controller.cfg == app_cfg
        assert controller.listening is True
        assert controller.lang == "en"
        assert controller.input_mode == "mic"
        
        # Verify starts
        mock_deps["StreamingSttWorker"].assert_called_once()
        assert mock_deps["Worker"].call_count >= 2 # Streaming and LLM workers
        
        # Verify UI handlers set
        controller.window.set_handlers.assert_called_once()

    def test_toggle_listening(self, mock_deps, app_cfg):
        controller = AppController(app_cfg)
        assert controller.listening is True
        controller.toggle_listening()
        assert controller.listening is False
        controller.toggle_listening()
        assert controller.listening is True

    def test_toggle_language(self, mock_deps, app_cfg):
        controller = AppController(app_cfg)
        assert controller.lang == "en"
        
        # We need to track the worker stop call
        worker_instance = mock_deps["Worker"].return_value
        
        controller.toggle_language()
        assert controller.lang == "es"
        
        # Verify worker was restarted
        worker_instance.stop.assert_called_once()
        assert mock_deps["Worker"].call_count == 3 # Initial 2 + 1 restart

    def test_toggle_input_mode(self, mock_deps, app_cfg):
        controller = AppController(app_cfg)
        assert controller.input_mode == "mic"
        assert controller.segmenter.min_silence_sec == 1.0 # From mic_vad
        
        controller.toggle_input_mode()
        assert controller.input_mode == "meeting"
        assert controller.segmenter.min_silence_sec == 0.5 # From vad

    def test_on_audio_frame_submits_to_streaming(self, mock_deps, app_cfg):
        controller = AppController(app_cfg)
        streaming_worker = controller.streaming_worker
        
        audio = np.zeros(1024, dtype=np.float32)
        frame = AudioFrame(audio=audio, frames=1024, timestamp=1.0)
        
        controller._on_audio_frame(frame)
        
        # Check that it submitted an AudioChunk
        streaming_worker.submit.assert_called()
        args = streaming_worker.submit.call_args[0][0]
        assert isinstance(args, AudioChunk)
        assert args.ts == 1.0

    def test_tick_drains_stt_partial(self, mock_deps, app_cfg):
        controller = AppController(app_cfg)
        # Put an event in the results_q
        controller.results_q.put(SttPartial(text="hel"))
        
        controller._tick()
        
        assert controller.transcript_partial == "hel"
        # Verify render called with state
        controller.window.render.assert_called_once()
        state = controller.window.render.call_args[0][0]
        assert state.transcript_partial == "hel"

    def test_tick_drains_stt_final_triggers_llm(self, mock_deps, app_cfg):
        controller = AppController(app_cfg)
        controller.pending_llm_send = True
        
        controller.results_q.put(SttFinal(text="Hello world"))
        
        controller._tick()
        
        assert controller.transcript_final == "Hello world"
        assert controller.last_utterance_final == "Hello world"
        assert controller.pending_llm_send is False
        
        # Verify LLM job submitted
        controller.llm_worker.submit.assert_called_once()
        job = controller.llm_worker.submit.call_args[0][0]
        assert job.question == "Hello world"

    def test_tick_drains_llm_event(self, mock_deps, app_cfg):
        controller = AppController(app_cfg)
        from rdoai.pipeline.llm_worker import LlmEvent
        
        controller.results_q.put(LlmEvent(kind="answer", answer="The sky is blue"))
        
        controller._tick()
        
        assert controller.suggestion == "The sky is blue"

    def test_finalize_now_no_speech(self, mock_deps, app_cfg):
        controller = AppController(app_cfg)
        controller.segmenter.in_segment = False
        
        controller.finalize_now()
        
        assert "none" in controller.last_capture

    @patch("rdoai.app.controller.write_wav_mono_int16")
    @patch("rdoai.app.controller.time.time", return_value=123.456)
    def test_finalize_now_success(self, mock_time, mock_write_wav, mock_deps, app_cfg):
        controller = AppController(app_cfg)
        controller.segmenter.in_segment = True
        controller.segmenter.segment_frames = [np.zeros(16000, dtype=np.float32)]
        
        controller.finalize_now()
        
        mock_write_wav.assert_called_once()
        assert controller.segmenter.segment_frames == []
        assert "1.00s" in controller.last_capture
