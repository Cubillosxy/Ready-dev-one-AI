import os
import queue
import time
import sounddevice as sd
import numpy as np
from dataclasses import replace

from rdoai.config import AppConfig
from rdoai.audio.capture import AudioCapture, AudioFrame
from rdoai.audio.levels import rms_db
from rdoai.audio.segmenter import AutoSegmenter, SegmentResult
from rdoai.ui.overlay import OverlayWindow, OverlayState
from rdoai.io.wav_writer import write_wav_mono_int16
from rdoai.stt.factory import build_stt
from rdoai.stt.vosk_streaming import VoskStreamingStt
from rdoai.pipeline.assistant import AssistantPipeline
from rdoai.utils.worker import Worker
from rdoai.pipeline.llm_worker import LlmWorker, LlmJob, LlmEvent

from rdoai.pipeline.events import AudioChunk, SttPartial, SttFinal, ErrorEvent
from rdoai.pipeline.streaming_stt_worker import StreamingSttWorker

from rdoai.pipeline.events import FlushStt

ON_PAUSE = "on_pause"


class AppController:
    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        os.makedirs(self.cfg.output_dir, exist_ok=True)

        self.stt = build_stt(self.cfg.stt)
        self.lang = self.cfg.lang.lang  # "en" | "es"
        self.input_mode = self.cfg.input.mode  # "meeting" | "mic"
        self.transcript_partial = ""

        self.transcript_final = ""

        self.assistant = AssistantPipeline(self.cfg)
        self.suggestion = ""

        self.listening = True
        self.last_capture = "none"
        self.last_level_db = -160.0
        self.last_vad_state = "silence"
        self.last_pause = 0.0
        self._last_debug_print = 0.0

        self.segmenter = AutoSegmenter(
            vad=self.cfg.vad,
            output_dir=self.cfg.output_dir,
            sample_rate=self.cfg.audio.sample_rate,
        )

        self.capture = AudioCapture(self.cfg.audio)
        self.capture.on_frame = self._on_audio_frame
        self.capture.on_error = self._on_audio_error

        self.window = OverlayWindow(
            geometry=self.cfg.ui.geometry,
            alpha=self.cfg.ui.window_alpha,
            always_on_top=self.cfg.ui.always_on_top,
        )

        # Apply initial VAD settings based on mode
        if self.input_mode == "mic":
            self.segmenter.min_silence_sec = self.cfg.mic_vad.min_silence_sec
            self.segmenter.silence_threshold_db = self.cfg.mic_vad.silence_threshold_db
            self.segmenter.speech_threshold_db = self.cfg.mic_vad.speech_threshold_db
        else:
            self.segmenter.min_silence_sec = self.cfg.vad.min_silence_sec
            self.segmenter.silence_threshold_db = self.cfg.vad.silence_threshold_db
            self.segmenter.speech_threshold_db = self.cfg.vad.speech_threshold_db


        self.window.set_handlers(
            on_toggle_listening=self.toggle_listening,
            on_finalize=self.finalize_now,
            on_toggle_language=self.toggle_language,
            on_toggle_input=self.toggle_input_mode,
        )

        self.device_label = self._device_label()
        self.results_q: "queue.Queue[object]" = queue.Queue()

        #... 

        self.last_utterance_final = ""

        model_path = self.cfg.streaming_stt.vosk_model_path
        self.streaming_logic = StreamingSttWorker(
            model_path=model_path,
            sample_rate=self.cfg.audio.sample_rate,
            publish=lambda ev: self.results_q.put(ev),
        )
        self.streaming_worker = Worker(
            name="streaming-stt",
            target=self.streaming_logic.on_audio,
            maxsize=300,  # drop if saturated; keep UI realtime
        )
        self.streaming_worker.start()

        def publish(event: object) -> None:
            self.results_q.put(event)

        self.llm_worker_logic = LlmWorker(self.cfg, publish=publish)
        self.llm_worker = Worker(name="llm-worker", target=self.llm_worker_logic.handle, maxsize=50)
        self.llm_worker.start()

        # UI state for suggestion / last error
        self.suggestion = ""
        self.last_error = ""


        self.pending_llm_send = False
        self.pending_llm_set_at = 0.0
        self.pending_llm_timeout_sec = 2.0  # give STT more time to finalize

    def run(self) -> None:
        self.capture.start()
        self._tick()
        self.window.mainloop()
        self.llm_worker.stop()
        self.capture.stop()

    def toggle_listening(self) -> None:
        self.listening = not self.listening
        print(f"[UI] AI -> {'ON' if self.listening else 'OFF'}")
        # Clear suggestions and errors when toggling AI state
        if not self.listening:
            self.suggestion = ""
            self.last_error = ""

    def finalize_now(self) -> None:
        if not self.segmenter.in_segment or not self.segmenter.segment_frames:
            self.last_capture = "none (no active speech)"
            print("[MANUAL] No active speech segment to finalize.")
            return

        audio = np.concatenate(self.segmenter.segment_frames)
        dur = float(audio.size) / float(self.cfg.audio.sample_rate)

        ts = time.time()
        fname = time.strftime("manual_%Y%m%d_%H%M%S") + f"_{int(ts*1000)%1000:03d}.wav"
        path = os.path.join(self.cfg.output_dir, fname)
        write_wav_mono_int16(path, audio, self.cfg.audio.sample_rate)

        self.segmenter.reset_segment()
        self.last_capture = f"{os.path.basename(path)} ({dur:.2f}s)"
        print(f"[MANUAL] Saved {path} duration={dur:.2f}s")

    def _device_label(self) -> str:
        try:
            dev = sd.query_devices(self.cfg.audio.device_index)
            return f"Input device: #{self.cfg.audio.device_index} - {dev['name']}"
        except Exception:
            return f"Input device: #{self.cfg.audio.device_index}"

    def _on_audio_frame(self, frame: AudioFrame) -> None:
        level = rms_db(frame.audio)

        self.streaming_worker.submit(AudioChunk(audio=frame.audio, ts=frame.timestamp))

        self.last_level_db = level

        now = time.time()
        if now - self._last_debug_print >= self.cfg.debug_print_every_sec:
            self._last_debug_print = now

        produced: SegmentResult | None = None
        if self.listening:
            produced = self.segmenter.process(frame.audio, level)

        self.last_vad_state = self.segmenter.state
        self.last_pause = self.segmenter.last_pause

        if produced:
            base = os.path.basename(produced.wav_path)
            self.last_capture = f"{base} ({produced.duration_sec:.2f}s)"
            print(f"[AUTO] Saved {produced.wav_path} duration={produced.duration_sec:.2f}s")

            self.streaming_worker.submit(FlushStt(reason="segment_end"))

            # LLM step (use last_utterance_final from streaming STT FINAL)
            if self.cfg.assistant.trigger == ON_PAUSE:
                self.pending_llm_send = True
                self.pending_llm_set_at = time.time()


    def _on_audio_error(self, e: Exception) -> None:
        print("[ERROR] Audio:", e)

    def _tick(self) -> None:

        # Drain worker events without blocking
        try:
            while True:
                ev = self.results_q.get_nowait()
                if isinstance(ev, SttPartial):
                    self.transcript_partial = ev.text

                elif isinstance(ev, SttFinal):
                    txt = ev.text.strip()
                    if txt:
                        self.last_utterance_final = txt
                        self.transcript_final = txt
                    self.transcript_partial = ""
                    print(f"[STT][FINAL-] {txt}")

                    if self.pending_llm_send and self.listening:
                        q = (self.last_utterance_final or "").strip()
                        if q:
                            print("[LLM] Enqueue job (from STT FINAL)...")
                            self.llm_worker.submit(LlmJob(question=q, lang=self.lang))
                        else:
                            print("[LLM] Skip: FINAL empty.")

                        self.pending_llm_send = False
                    elif self.pending_llm_send and not self.listening:
                        print("[LLM] Skip: AI is OFF.")
                        self.pending_llm_send = False


                elif isinstance(ev, ErrorEvent):
                    print(f"[{ev.source.upper()}][ERROR] status={ev.status_code} msg={ev.message}")
                    # opcional: mostrar algo en UI
                    self.last_error = f"{ev.source}: {ev.message}"

                elif isinstance(ev, LlmEvent):
                    if ev.kind == "answer":
                        self.suggestion = ev.answer
                        self.last_error = ""
                        print("[LLM] Answer received (async).")
                    elif ev.kind == "error":
                        e = ev.error
                        # Log detailed
                        print(f"[LLM][ERROR] status={e.status_code} code={e.code} retry_after={e.retry_after_sec} msg={e.message}")

                        # UI-friendly message
                        if e.status_code == 429:
                            self.last_error = "LLM 429 (rate limit/quota)."
                            self.suggestion = "Try again in a moment."
                        elif e.status_code == 401:
                            self.last_error = "LLM 401 (bad API key)."
                            self.suggestion = "Check OPENAI_API_KEY."
                        else:
                            self.last_error = f"LLM error: {e.status_code or ''} {e.code}"
                            self.suggestion = "See logs."
        except queue.Empty:
            pass


        state = OverlayState(
            listening=self.listening,
            level_db=self.last_level_db,
            vad_state=self.last_vad_state,
            last_pause_sec=self.last_pause,
            last_capture=self.last_capture,
            device_label=self.device_label,
            hint=(self.last_error or "Buttons work always. ESC quits."),
            transcript_partial=self.transcript_partial,
            transcript_final=self.transcript_final,
            suggestion=self.suggestion,

        )
        self.window.render(state)
        self.window.every(50, self._tick)


    def _current_vosk_model_path(self) -> str:
        return self.cfg.lang.vosk_model_en if self.lang == "en" else self.cfg.lang.vosk_model_es

    def _current_device_index(self) -> int:
        return self.cfg.input.meeting_device_index if self.input_mode == "meeting" else self.cfg.input.mic_device_index

    def toggle_language(self) -> None:
        self.lang = "es" if self.lang == "en" else "en"
        print(f"[UI] Language -> {self.lang}")

        # Clear transcripts
        self.transcript_partial = ""
        self.transcript_final = ""
        self.last_utterance_final = ""

        # Restart streaming STT worker with new model
        self._restart_streaming_worker()

    def toggle_input_mode(self) -> None:
        self.input_mode = "mic" if self.input_mode == "meeting" else "meeting"
        print(f"[UI] Input mode -> {self.input_mode}")

        # Restart audio capture with new device
        self._restart_audio_capture()

        if self.input_mode == "mic":
            self.segmenter.min_silence_sec = self.cfg.mic_vad.min_silence_sec
            self.segmenter.silence_threshold_db = self.cfg.mic_vad.silence_threshold_db
            self.segmenter.speech_threshold_db = self.cfg.mic_vad.speech_threshold_db
        else:
            self.segmenter.min_silence_sec = self.cfg.vad.min_silence_sec
            self.segmenter.silence_threshold_db = self.cfg.vad.silence_threshold_db
            self.segmenter.speech_threshold_db = self.cfg.vad.speech_threshold_db





    def _restart_streaming_worker(self) -> None:
        
        self.streaming_worker.stop()
 

        model_path = self._current_vosk_model_path()
        self.streaming_logic = StreamingSttWorker(
            model_path=model_path,
            sample_rate=self.cfg.audio.sample_rate,
            publish=lambda ev: self.results_q.put(ev),
        )
        self.streaming_worker = Worker(
            name="streaming-stt",
            target=self.streaming_logic.on_audio,
            maxsize=300,
        )
        self.streaming_worker.start()
        print(f"[STT] Streaming model -> {model_path}")


    def _restart_audio_capture(self) -> None:
        # Stop current stream
        try:
            self.capture.stop()
        except Exception:
            pass

        time.sleep(0.15)

        new_index = self._current_device_index()

        # Create a NEW AudioConfig with the new device index
        new_audio_cfg = replace(self.cfg.audio, device_index=new_index)

        # Create a NEW AudioCapture using the NEW config
        self.capture = AudioCapture(new_audio_cfg)
        self.capture.on_frame = self._on_audio_frame
        self.capture.on_error = self._on_audio_error
        self.capture.start()

        # Update label shown in UI
        self.device_label = self._device_label()
