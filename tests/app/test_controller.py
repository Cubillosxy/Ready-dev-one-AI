"""Tests for AppController."""
import pytest
import numpy as np
import queue
from unittest.mock import MagicMock, patch
from rdoai.app.controller import AppController
from rdoai.pipeline.events import SttPartial, SttFinal, AudioChunk, ErrorEvent, FlushStt
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

        controller.toggle_input_mode()
        assert controller.input_mode == "mic"
        assert controller.segmenter.min_silence_sec == 1.0 # Back to mic_vad

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

    def test_run(self, mock_deps, app_cfg):
        controller = AppController(app_cfg)
        controller.run()
        
        controller.capture.start.assert_called_once()
        controller.window.mainloop.assert_called_once()
        controller.llm_worker.stop.assert_called_once()
        controller.capture.stop.assert_called_once()

    def test_on_audio_error(self, mock_deps, app_cfg):
        controller = AppController(app_cfg)
        controller._on_audio_error(Exception("test error"))
        # Just ensures it doesn't crash and prints (captured by stdout)

    def test_tick_drains_error_event(self, mock_deps, app_cfg):
        controller = AppController(app_cfg)
        controller.results_q.put(ErrorEvent(source="test", message="msg", status_code=500))
        controller._tick()
        assert controller.last_error == "test: msg"

    def test_tick_llm_error_429(self, mock_deps, app_cfg):
        from rdoai.llm.errors import LlmErrorInfo
        controller = AppController(app_cfg)
        err = LlmErrorInfo(code="rate_limit", message="quota", status_code=429)
        controller.results_q.put(LlmEvent(kind="error", error=err))
        controller._tick()
        assert "429" in controller.last_error
        assert "moment" in controller.suggestion

    def test_tick_llm_error_401(self, mock_deps, app_cfg):
        from rdoai.llm.errors import LlmErrorInfo
        controller = AppController(app_cfg)
        err = LlmErrorInfo(code="auth", message="unauthorized", status_code=401)
        controller.results_q.put(LlmEvent(kind="error", error=err))
        controller._tick()
        assert "401" in controller.last_error
        assert "OPENAI_API_KEY" in controller.suggestion

    def test_tick_llm_error_generic(self, mock_deps, app_cfg):
        from rdoai.llm.errors import LlmErrorInfo
        controller = AppController(app_cfg)
        err = LlmErrorInfo(code="err", message="msg", status_code=500)
        controller.results_q.put(LlmEvent(kind="error", error=err))
        controller._tick()
        assert "LLM error" in controller.last_error

    def test_tick_stt_final_empty_skips_llm(self, mock_deps, app_cfg):
        controller = AppController(app_cfg)
        controller.pending_llm_send = True
        controller.results_q.put(SttFinal(text="  "))
        controller._tick()
        controller.llm_worker.submit.assert_not_called()
        assert controller.pending_llm_send is False

    def test_init_meeting_mode_vad(self, mock_deps, app_cfg):
        from dataclasses import replace
        from rdoai.config import InputConfig
        # replace to avoid frozen error
        cfg = replace(app_cfg, input=InputConfig(mode="meeting", mic_device_index=1, meeting_device_index=2))
        controller = AppController(cfg)
        assert controller.segmenter.min_silence_sec == 0.5 # Default vad

    def test_device_label_failure(self, mock_deps, app_cfg):
        mock_deps["sd"].query_devices.side_effect = Exception("failed")
        controller = AppController(app_cfg)
        assert "Input device: #" in controller.device_label

    def test_on_audio_frame_produced_segment(self, mock_deps, app_cfg):
        from rdoai.audio.segmenter import SegmentResult
        controller = AppController(app_cfg)
        # Mock produced segment
        mock_deps["write_wav_mono_int16"] = patch("rdoai.app.controller.write_wav_mono_int16").start()
        
        # force produced
        with patch.object(controller.segmenter, 'process', return_value=SegmentResult("path.wav", 1.0, 1.0)):
            audio = np.zeros(1024, dtype=np.float32)
            frame = AudioFrame(audio=audio, frames=1024, timestamp=1.0)
            controller._on_audio_frame(frame)
        
        assert "path.wav" in controller.last_capture
        controller.streaming_worker.submit.assert_any_call(FlushStt(reason="segment_end"))
        assert controller.pending_llm_send is True

    def test_restart_audio_capture_exception(self, mock_deps, app_cfg):
        controller = AppController(app_cfg)
        controller.capture.stop.side_effect = Exception("error")
        # Should not raise
        controller._restart_audio_capture()

    def test_publish_callback(self, mock_deps, app_cfg):
        controller = AppController(app_cfg)
        # The publish func is passed to LlmWorker
        publish_func = mock_deps["LlmWorker"].call_args[1]["publish"]
        publish_func("test_event")
        assert controller.results_q.get_nowait() == "test_event"
