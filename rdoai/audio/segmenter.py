import os
import time
from dataclasses import dataclass
import numpy as np

from rdoai.config import VadConfig
from rdoai.io.wav_writer import write_wav_mono_int16

@dataclass
class SegmentResult:
    wav_path: str
    duration_sec: float
    timestamp: float

class AutoSegmenter:
    def __init__(self, vad: VadConfig, output_dir: str, sample_rate: int):
        self.vad = vad
        self.min_silence_sec = vad.min_silence_sec
        self.silence_threshold_db = vad.silence_threshold_db
        self.speech_threshold_db = vad.speech_threshold_db

        self.sample_rate = sample_rate
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        self.state = "silence"
        self.last_change = time.time()

        self.in_segment = False
        self.segment_frames: list[np.ndarray] = []
        self.last_pause = 0.0

    def reset_segment(self) -> None:
        self.segment_frames = []
        self.in_segment = False

    def process(self, audio_block: np.ndarray, level_db: float) -> SegmentResult | None:
        now = time.time()
        produced: SegmentResult | None = None

        if self.state == "silence":
            if level_db > self.speech_threshold_db:
                silence_dur = now - self.last_change
                self.last_pause = silence_dur

                self.state = "speech"
                self.last_change = now

                self.in_segment = True
                self.segment_frames = [audio_block.copy()]

        else:  # speech
            if self.in_segment:
                self.segment_frames.append(audio_block.copy())

            if level_db < self.silence_threshold_db:
                self.state = "silence"
                self.last_change = now

        # finalize after sustained silence
        if self.state == "silence" and self.in_segment:
            silence_dur = now - self.last_change
            if silence_dur >= self.min_silence_sec:
                audio = np.concatenate(self.segment_frames) if self.segment_frames else np.array([], dtype=np.float32)
                dur = float(audio.size) / float(self.sample_rate) if audio.size else 0.0

                ts = time.time()
                fname = time.strftime("q_%Y%m%d_%H%M%S") + f"_{int(ts*1000)%1000:03d}.wav"
                path = os.path.join(self.output_dir, fname)

                if audio.size:
                    write_wav_mono_int16(path, audio, self.sample_rate)

                produced = SegmentResult(wav_path=path, duration_sec=dur, timestamp=ts)
                self.reset_segment()

        return produced
